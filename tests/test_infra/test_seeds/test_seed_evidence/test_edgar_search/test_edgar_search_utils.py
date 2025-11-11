"""
Purpose
-------
Unit tests for `edgar_search_utils`.

Key behaviors
-------------
- Verify that `build_url`:
  - uses the validity-window end minus one day when `end_date` is None,
  - honors an explicit `end_date` and emits a debug breadcrumb.
- Confirm that `build_company_feed_url`:
  - constructs the expected EDGAR Atom query string, and
  - logs a `batch_fetch_start` debug event with the correct context.
- Ensure that `create_run_data`:
  - seeds `oldest_filing_date` with the window end when None, and
  - preserves a provided `oldest_filing_date`.
- Validate that `extract_entries_and_namespace`:
  - handles namespaced and non-namespaced Atom `<feed>` roots,
  - logs a warning and returns no entries for unexpected roots.
- Check that `find_element` and `find_all_elements`:
  - use namespace-aware lookups when bindings are present, and
  - fall back to local-name XPath when bindings are absent.
- Confirm that `set_atom_entry`:
  - resolves `id` and `title` via `find_element(...)`, and
  - persists the expected Atom metadata into `RawRecord`.

Conventions
-----------
- Logging is captured via a minimal `_DummyLogger` that mimics the relevant
  subset of the `InfraLogger` interface.
- HTTP responses are replaced with lightweight `_DummyResponse` objects
  exposing a `.content` attribute.
- Tests focus on control flow, wiring, and normalization rather than on
  XML parser internals.

Downstream usage
----------------
Run this module with `pytest` as part of the CI suite. The tests act as
executable documentation for how EDGAR search conditions gate entries,
apply validity windows, and determine when pagination should stop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, cast
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest
import requests

from infra.logging.infra_logger import InfraLogger
from infra.seeds.seed_evidence.edgar_search import edgar_search_utils as esu
from infra.seeds.seed_evidence.edgar_search.edgar_config import PAGE_SIZE
from infra.seeds.seed_evidence.records.raw_record import RawRecord
from infra.seeds.seed_evidence.seed_evidence_types import NameSpaceBindings, ValidityWindow


class _DummyLogger:
    """
    Purpose
    -------
    Minimal stand-in for `InfraLogger` for EDGAR search utility tests.

    Key behaviors
    -------------
    - Captures `.debug(...)`, `.info(...)`, and `.warning(...)` calls.
    - Stores message identifiers and structured context for assertions.

    Parameters
    ----------
    None

    Attributes
    ----------
    debugs : list[tuple[str, dict[str, Any]]]
        Recorded debug-level messages.
    infos : list[tuple[str, dict[str, Any]]]
        Recorded info-level messages.
    warnings : list[tuple[str, dict[str, Any]]]
        Recorded warning-level messages.

    Notes
    -----
    - Only the subset of the real logger interface used in `edgar_search_utils`
      is implemented.
    """

    def __init__(self) -> None:
        self.debugs: List[Tuple[str, Dict[str, Any]]] = []
        self.infos: List[Tuple[str, Dict[str, Any]]] = []
        self.warnings: List[Tuple[str, Dict[str, Any]]] = []

    def debug(self, msg: str, context: Dict[str, Any] | None = None) -> None:
        self.debugs.append((msg, context or {}))

    def info(self, *args: Any, **kwargs: Any) -> None:
        # First positional argument is the event name ("loop_break", etc.).
        event = args[0] if args else kwargs.get("event", None)
        if event is None:
            event = kwargs.get("msg", "")
        context = kwargs.get("context", {}) or {}
        self.infos.append((event, context))

    def warning(self, *args: Any, **kwargs: Any) -> None:
        # Same pattern as info: event id is first positional arg.
        event = args[0] if args else kwargs.get("event", None)
        if event is None:
            event = kwargs.get("msg", "")
        context = kwargs.get("context", {}) or {}
        self.warnings.append((event, context))


class _DummyResponse:
    """
    Purpose
    -------
    Lightweight response stand-in exposing only the `.content` attribute.

    Key behaviors
    -------------
    - Mimics the subset of `requests.Response` required by
      `extract_entries_and_namespace(...)`.

    Parameters
    ----------
    content : bytes
        Raw byte payload to expose via `.content`.

    Attributes
    ----------
    content : bytes
        Stored byte content used by XML parsers.

    Notes
    -----
    - No HTTP semantics (status, headers, URL) are modeled; this is purely
      a container for bytes.
    """

    def __init__(self, content: bytes) -> None:
        self.content = content


def test_build_url_uses_window_end_minus_one_day_when_end_date_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that `build_url` derives `dateb` from the validity-window end
    minus one day when `end_date` is None.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `build_company_feed_url` and capture its arguments.

    Returns
    -------
    None
        The test passes if `build_url(...)` calls `build_company_feed_url(...)`
        with `inclusive_window_end` equal to `(window_end - 1 day)` formatted
        as `YYYYMMDD`.

    Raises
    ------
    AssertionError
        If the derived `inclusive_window_end` or returned URL is incorrect.
    """

    logger = _DummyLogger()

    window: ValidityWindow = (
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2020-02-10", tz="UTC"),
    )

    captured: Dict[str, Any] = {}

    def fake_build_company_feed_url(
        candidate: str, inclusive_window_end: str, logger_param: InfraLogger
    ) -> str:
        captured["candidate"] = candidate
        captured["inclusive_window_end"] = inclusive_window_end
        captured["logger"] = logger_param
        return "FAKE-URL"

    monkeypatch.setattr(esu, "build_company_feed_url", fake_build_company_feed_url)

    result = esu.build_url(
        ticker="0000123456",
        validity_window=window,
        logger=cast(InfraLogger, logger),
        end_date=None,
    )

    assert result == "FAKE-URL"
    # Window end is 2020-02-10 → inclusive dateb should be 2020-02-09.
    assert captured["candidate"] == "0000123456"
    assert captured["inclusive_window_end"] == "20200209"


def test_build_url_uses_explicit_end_date_and_logs_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Check that `build_url` honors an explicit `end_date` and emits a debug breadcrumb.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `build_company_feed_url` and observe arguments.

    Returns
    -------
    None
        The test passes if:
            - `build_company_feed_url(...)` is called with `inclusive_window_end`
              equal to the explicit `end_date` formatted as `YYYYMMDD`, and
            - a debug log is recorded for the non-window end date.

    Raises
    ------
    AssertionError
        If the derived date or debug log context is incorrect.
    """

    logger = _DummyLogger()

    window: ValidityWindow = (
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2020-01-31", tz="UTC"),
    )
    explicit_end: pd.Timestamp = pd.Timestamp("2020-03-15", tz="UTC")

    captured: Dict[str, Any] = {}

    def fake_build_company_feed_url(
        candidate: str, inclusive_window_end: str, logger_param: InfraLogger
    ) -> str:
        captured["candidate"] = candidate
        captured["inclusive_window_end"] = inclusive_window_end
        captured["logger"] = logger_param
        return "FAKE-URL-EXPLICIT"

    monkeypatch.setattr(esu, "build_company_feed_url", fake_build_company_feed_url)

    result = esu.build_url(
        ticker="0000123456",
        validity_window=window,
        logger=cast(InfraLogger, logger),
        end_date=explicit_end,
    )

    assert result == "FAKE-URL-EXPLICIT"
    assert captured["candidate"] == "0000123456"
    assert captured["inclusive_window_end"] == "20200315"

    # Validate debug log.
    assert logger.debugs, "Expected a debug log for explicit end_date"
    msg, ctx = logger.debugs[0]
    assert msg == "using_non_window_end_end_date_for_url_construction"
    assert ctx["stage"] == "url_construction"
    assert ctx["ticker"] == "0000123456"
    assert ctx["end_date"] == "20200315"


def test_build_company_feed_url_constructs_query_and_logs_debug() -> None:
    """
    Ensure `build_company_feed_url` constructs the expected EDGAR Atom URL
    and logs a `batch_fetch_start` event.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if:
            - the resulting URL encodes the correct query parameters, and
            - a debug log is emitted with `stage="build_company_feed_url"` and
              `offset=0`.

    Raises
    ------
    AssertionError
        If the URL query string or debug context is incorrect.
    """

    logger = _DummyLogger()
    candidate = "0000123456"
    inclusive_end = "20200131"

    url = esu.build_company_feed_url(candidate, inclusive_end, cast(InfraLogger, logger))

    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    assert parsed.path.endswith("/cgi-bin/browse-edgar")
    assert query["action"] == ["getcompany"]
    assert query["CIK"] == [candidate]
    assert query["owner"] == ["exclude"]
    assert query["dateb"] == [inclusive_end]
    assert query["count"] == [str(PAGE_SIZE)]
    assert query["start"] == ["0"]
    assert query["output"] == ["atom"]

    assert logger.debugs, "Expected a batch_fetch_start debug log"
    msg, ctx = logger.debugs[0]
    assert msg == "batch_fetch_start"
    assert ctx["stage"] == "build_company_feed_url"
    assert ctx["candidate"] == candidate
    assert ctx["offset"] == 0


def test_create_run_data_seeds_oldest_filing_date_from_window_end() -> None:
    """
    Verify that `create_run_data` seeds `oldest_filing_date` with window end
    when the parameter is None.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if `oldest_filing_date` equals `validity_window[1]`
        when no explicit value is provided.

    Raises
    ------
    AssertionError
        If `oldest_filing_date` is not initialized to the window end.
    """

    logger = _DummyLogger()
    window: ValidityWindow = (
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2020-01-31", tz="UTC"),
    )

    run_data = esu.create_run_data(
        ticker="AAA",
        validity_window=window,
        candidate="0000123456",
        name_space_bindings=None,
        logger=cast(InfraLogger, logger),
        entry=None,
        oldest_filing_date=None,
    )

    assert run_data["ticker"] == "AAA"
    assert run_data["candidate"] == "0000123456"
    assert run_data["validity_window"] == window
    assert run_data["name_space_bindings"] is None
    assert run_data["entry"] is None
    assert run_data["oldest_filing_date"] == window[1]


def test_create_run_data_preserves_existing_oldest_filing_date() -> None:
    """
    Check that `create_run_data` preserves a provided `oldest_filing_date`.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if the returned `RunData` uses the explicit
        `oldest_filing_date` instead of the window end.

    Raises
    ------
    AssertionError
        If `oldest_filing_date` does not match the explicit input value.
    """

    logger = _DummyLogger()
    window: ValidityWindow = (
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2020-01-31", tz="UTC"),
    )
    oldest = pd.Timestamp("2019-12-15", tz="UTC")

    run_data = esu.create_run_data(
        ticker="BBB",
        validity_window=window,
        candidate="0000987654",
        name_space_bindings={"ns": "http://example.com"},
        logger=cast(InfraLogger, logger),
        entry="ENTRY",
        oldest_filing_date=oldest,
    )

    assert run_data["ticker"] == "BBB"
    assert run_data["candidate"] == "0000987654"
    assert run_data["oldest_filing_date"] == oldest


def test_extract_entries_and_namespace_handles_namespaced_feed() -> None:
    """
    Ensure `extract_entries_and_namespace` handles a namespaced Atom <feed> root.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if:
            - entries are located via the default namespace, and
            - `ns_bindings` contains the default namespace under key `"ns"`.

    Raises
    ------
    AssertionError
        If entries are not parsed or namespace bindings are incorrect.
    """

    xml = b"""
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry><id>1</id></entry>
      <entry><id>2</id></entry>
    </feed>
    """

    response = cast(requests.Response, _DummyResponse(content=xml))
    logger = _DummyLogger()

    entries, ns = esu.extract_entries_and_namespace(response, cast(InfraLogger, logger))

    assert isinstance(entries, list)
    assert len(entries) == 2
    assert ns == {"ns": "http://www.w3.org/2005/Atom"}
    assert not logger.warnings, "No warnings expected for a valid feed root"


def test_extract_entries_and_namespace_handles_non_namespaced_feed() -> None:
    """
    Confirm `extract_entries_and_namespace` supports a non-namespaced <feed>.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if entries are returned and `ns_bindings` is None
        when the feed has no default namespace.

    Raises
    ------
    AssertionError
        If entries are missing or namespace bindings are unexpectedly present.
    """

    xml = b"""
    <feed>
      <entry><id>1</id></entry>
      <entry><id>2</id></entry>
    </feed>
    """

    response = cast(requests.Response, _DummyResponse(content=xml))
    logger = _DummyLogger()

    entries, ns = esu.extract_entries_and_namespace(response, cast(InfraLogger, logger))

    assert isinstance(entries, list)
    assert len(entries) == 2
    assert ns is None
    assert not logger.warnings, "No warnings expected for a valid non-namespaced feed root"


def test_extract_entries_and_namespace_logs_warning_on_non_feed_root() -> None:
    """
    Verify that `extract_entries_and_namespace` logs a warning and returns no
    entries when the root element is not <feed>.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if:
            - an `unexpected_atom_root` warning is logged, and
            - the returned entries list is empty and namespace bindings are None.

    Raises
    ------
    AssertionError
        If the warning is not emitted or entries are incorrectly returned.
    """

    xml = b"<html><body>Not a feed</body></html>"
    response = cast(requests.Response, _DummyResponse(content=xml))
    logger = _DummyLogger()

    entries, ns = esu.extract_entries_and_namespace(response, cast(InfraLogger, logger))

    assert entries == []
    assert ns is None
    assert logger.warnings, "Expected a warning for non-feed root"
    msg, ctx = logger.warnings[0]
    assert msg == "unexpected_atom_root"
    assert ctx["stage"] == "extract_entries_and_namespace"


def test_find_element_uses_namespaced_find_when_bindings_present() -> None:
    """
    Check that `find_element` uses `.find("ns:<tag>", namespaces=...)`
    when namespace bindings are provided.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if the dummy parent's `.find(...)` is invoked with
        the expected tag and namespace map.

    Raises
    ------
    AssertionError
        If the returned node is incorrect or `.find(...)` is not used.
    """

    class DummyParent:
        def __init__(self) -> None:
            self.calls: List[Tuple[str, Dict[str, Any]]] = []

        def find(self, expr: str, namespaces: Dict[str, Any]) -> str:
            self.calls.append((expr, namespaces))
            return "FOUND"

    parent = DummyParent()
    ns_bindings: NameSpaceBindings = {"ns": "http://example.com"}

    result = esu.find_element(parent, "updated", ns_bindings)

    assert result == "FOUND"
    assert parent.calls == [("ns:updated", ns_bindings)]


def test_find_element_falls_back_to_local_name_xpath_without_namespace() -> None:
    """
    Ensure `find_element` falls back to a local-name XPath when no
    namespace bindings are provided.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if the dummy parent's `.xpath(...)` is invoked with
        the expected expression and the first result is returned.

    Raises
    ------
    AssertionError
        If `.xpath(...)` is not used or the returned node is incorrect.
    """

    class DummyParent:
        def __init__(self) -> None:
            self.calls: List[str] = []

        def xpath(self, expr: str) -> List[str]:
            self.calls.append(expr)
            return ["X-NODE"]

    parent = DummyParent()
    result = esu.find_element(parent, "title", None)

    assert result == "X-NODE"
    assert parent.calls == [".//*[local-name()='title']"]


def test_find_all_elements_uses_findall_with_namespace() -> None:
    """
    Validate that `find_all_elements` uses `.findall("ns:<tag>", namespaces=...)`
    when namespace bindings are supplied.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if the correct expression and namespaces are passed
        into `.findall(...)` and the returned list is propagated.

    Raises
    ------
    AssertionError
        If the dummy parent is not called as expected or results are wrong.
    """

    class DummyParent:
        def __init__(self) -> None:
            self.calls: List[Tuple[str, Dict[str, Any]]] = []

        def findall(self, expr: str, namespaces: Dict[str, Any]) -> List[str]:
            self.calls.append((expr, namespaces))
            return ["A", "B"]

    parent = DummyParent()
    ns_bindings: NameSpaceBindings = {"ns": "http://example.com"}

    result = esu.find_all_elements(parent, "link", ns_bindings)

    assert result == ["A", "B"]
    assert parent.calls == [("ns:link", ns_bindings)]


def test_find_all_elements_uses_xpath_without_namespace() -> None:
    """
    Confirm that `find_all_elements` falls back to local-name XPath
    when namespace bindings are absent.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if the `.xpath(...)` method is invoked with the
        expected expression and the returned list is propagated.

    Raises
    ------
    AssertionError
        If `.xpath(...)` is not called or the result list is incorrect.
    """

    class DummyParent:
        def __init__(self) -> None:
            self.calls: List[str] = []

        def xpath(self, expr: str) -> List[str]:
            self.calls.append(expr)
            return ["L1", "L2"]

    parent = DummyParent()
    result = esu.find_all_elements(parent, "id", None)

    assert result == ["L1", "L2"]
    assert parent.calls == [".//*[local-name()='id']"]


def test_set_atom_entry_persists_minimal_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Check that `set_atom_entry` resolves `id` and `title` via `find_element(...)`
    and writes the expected `atom_entry` payload into `raw_record`.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `find_element(...)` to supply deterministic id/title nodes.

    Returns
    -------
    None
        The test passes if `raw_record["atom_entry"]` contains:
            - `alternate_link`,
            - ISO-8601 `updated`,
            - `entry_id` from `<id>`, and
            - `title` from `<title>`.

    Raises
    ------
    AssertionError
        If any of the expected fields are missing or incorrect.
    """

    class DummyElement:
        def __init__(self, text: str) -> None:
            self.text = text

    # Track calls by tag to return the correct dummy element.
    def fake_find_element(
        parent: Any, tag: str, ns: NameSpaceBindings | None
    ) -> DummyElement | None:
        if tag == "id":
            return DummyElement("ID-123")
        if tag == "title":
            return DummyElement("Some Title")
        return None

    monkeypatch.setattr(esu, "find_element", fake_find_element)

    entry = object()
    link_href = "https://sec.gov/Archives/edgar/data/123/000-INDEX.html"
    updated_ts = pd.Timestamp("2020-01-15T12:34:56Z")
    raw_record: Dict[str, Any] = {}

    esu.set_atom_entry(
        entry=entry,
        link_href=link_href,
        updated_ts=updated_ts,
        raw_record=cast(RawRecord, raw_record),
        name_space_bindings=None,
    )

    assert "atom_entry" in raw_record
    atom = raw_record["atom_entry"]
    assert atom["alternate_link"] == link_href
    assert atom["updated"] == updated_ts.isoformat()
    assert atom["entry_id"] == "ID-123"
    assert atom["title"] == "Some Title"
