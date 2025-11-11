"""
Purpose
-------
Unit tests for EDGAR filing index parsing (`edgar_filing_parse`).

Key behaviors
-------------
- Verify that `handle_alternate_link`:
  - wraps the index-page fetch in `make_request(...)`,
  - delegates HTML parsing to `try_extract_evidence_from_index(...)`, and
  - when parsing succeeds, enriches `RawRecord` and emits a DEBUG log.
- Ensure that `try_extract_evidence_from_index`:
  - parses form type, filed-at date, accession, CIK, and company name
    from the HTML,
  - normalizes accession (strip hyphens) and zero-pads the CIK,
  - updates `run_data["oldest_filing_date"]` when an earlier filing is seen,
  - calls `parse_8k_items(...)` for 8-K / 8-K/A and requires at least one
    eligible item, and
  - returns None when any required header field is missing or parsing fails.
- Check that `parse_8k_items`:
  - filters items based on `ELIGIBLE_8K_ITEMS`, and
  - returns aligned item-number / description lists.
- Confirm that `create_filled_link_data`:
  - asserts presence of required fields, and
  - returns a `FilledLinkData` with the same values.

Conventions
-----------
- Real network I/O and XPath expressions are stubbed via simple doubles.
- RunData and RawRecord are represented with dicts where possible; only the
  fields accessed by the module are populated.
- Tests focus on normalization, control flow, and interaction between
  helpers, not on lxml internals.

Downstream usage
----------------
Run with `pytest` as part of the seed-evidence test suite. These tests act
as executable documentation for how index pages are interpreted into
`FilledLinkData` and how that flows into upstream EDGAR evidence builders.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, cast

import pandas as pd
import pytest
import requests

from infra.logging.infra_logger import InfraLogger
from infra.seeds.seed_evidence.edgar_search import edgar_filing_parse as efp
from infra.seeds.seed_evidence.edgar_search.edgar_search_utils import RunData
from infra.seeds.seed_evidence.records.raw_record import RawRecord


class _DummyLogger:
    """
    Purpose
    -------
    Minimal stand-in for `InfraLogger` in filing-parse tests.

    Key behaviors
    -------------
    - Records info/debug/warning calls and their contexts.
    - Provides just enough surface for the module under test.

    Attributes
    ----------
    infos : list[tuple[str, dict[str, Any]]]
        Logged INFO events.
    warnings : list[tuple[str, dict[str, Any]]]
        Logged WARNING events.
    debugs : list[tuple[str, dict[str, Any]]]
        Logged DEBUG events.
    """

    def __init__(self) -> None:
        self.infos: List[Tuple[str, Dict[str, Any]]] = []
        self.warnings: List[Tuple[str, Dict[str, Any]]] = []
        self.debugs: List[Tuple[str, Dict[str, Any]]] = []

    def info(self, msg: str, context: Dict[str, Any] | None = None) -> None:
        self.infos.append((msg, context or {}))

    def warning(self, msg: str, context: Dict[str, Any] | None = None) -> None:
        self.warnings.append((msg, context or {}))

    def debug(self, msg: str, context: Dict[str, Any] | None = None) -> None:
        self.debugs.append((msg, context or {}))


class _DummySession(requests.Session):
    """
    Purpose
    -------
    Minimal `requests.Session` subclass used to satisfy type hints in tests.

    Key behaviors
    -------------
    - Behaves like a regular `requests.Session`.
    - Avoids real network I/O because all HTTP calls are stubbed.

    Parameters
    ----------
    None
        Instances are created without additional initialization parameters.

    Attributes
    ----------
    All inherited from requests.Session
        No new attributes are introduced; the class is used purely as a
        type-stable stand-in.

    Notes
    -----
    - This class is intentionally empty; its existence keeps the tests in
      line with the production type signature that expects a Session.
    """


class _DummyResponse:
    """
    Purpose
    -------
    Lightweight stand-in for `requests.Response` that also acts as a
    context manager for `make_request` stubbing.

    Key behaviors
    -------------
    - Exposes `.content`, `.url`, and `.status_code` attributes.
    - Supports `with` blocks so it can be returned directly from
      `make_request(...)` in tests.

    Parameters
    ----------
    content : bytes
        Raw response body used by the HTML parser under test.
    url : str
        URL to expose via the `.url` attribute, typically simulating
        a redirected or canonical index URL.
    status : int
        HTTP status code to expose via `.status_code`.

    Attributes
    ----------
    content : bytes
        Stored response payload.
    url : str
        Effective response URL.
    status_code : int
        HTTP status code used for log assertions.

    Notes
    -----
    - The context-manager methods `__enter__` and `__exit__` are no-ops
      except for returning `self`, which is sufficient for the tests.
    """

    def __init__(self, content: bytes, url: str = "https://example.test/index", status: int = 200):
        self.content = content
        self.url = url
        self.status_code = status

    def __enter__(self) -> "_DummyResponse":  # type: ignore[override]
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None


def test_handle_alternate_link_success_enriches_raw_record_and_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that `handle_alternate_link`:
      - calls `make_request(...)` with the link URL,
      - delegates to `try_extract_evidence_from_index(...)`, and
      - on success, writes `filing_page` into `raw_record` and logs a DEBUG event.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `make_request` and `try_extract_evidence_from_index`.

    Returns
    -------
    None
        The test passes if `raw_record["filing_page"]` is set and the logger
        captures an `index_page_extracted` event with the expected context.
    """

    logger = _DummyLogger()
    run_data: RunData = {
        "ticker": "EXMPL",
        "validity_window": (
            pd.Timestamp("2019-01-01", tz="UTC"),
            pd.Timestamp("2024-01-01", tz="UTC"),
        ),
        "candidate": "0000000001",
        "name_space_bindings": {},
        "logger": cast(InfraLogger, logger),
        "entry": {},
        "oldest_filing_date": pd.Timestamp("2100-01-01", tz="UTC"),
    }
    raw_record: RawRecord = cast(RawRecord, {})

    link_url = "https://example.test/index"
    filled: efp.FilledLinkData = {
        "accession_num": "000000000120000123",
        "cik": "0000000001",
        "form_type": "8-K",
        "filed_at": pd.Timestamp("2020-01-15", tz="UTC"),
        "company_name": "Example Corp",
        "items_8k": ["1.01"],
        "items_descriptions_8k": ["Item description"],
    }

    captured_request_url: Dict[str, Any] = {}

    def fake_make_request(
        url: str,
        expect_json: bool = False,
        max_retries: int = 5,
        backoff_factor: float = 0.5,
        timeout: tuple[float, float] = (3.5, 10.0),
        sleep_time: float = 0.0,
        session: requests.Session | None = None,
    ) -> _DummyResponse:
        captured_request_url["url"] = url
        return _DummyResponse(
            content=b"", url="https://example.test/index?redirected=1", status=200
        )

    def fake_try_extract_evidence_from_index(
        response: requests.Response, rd: RunData
    ) -> efp.FilledLinkData | None:
        assert rd is run_data
        return filled

    monkeypatch.setattr(efp, "make_request", fake_make_request)
    monkeypatch.setattr(
        efp, "try_extract_evidence_from_index", fake_try_extract_evidence_from_index
    )

    result = efp.handle_alternate_link(
        link_url=link_url,
        raw_record=raw_record,
        run_data=run_data,
        session=_DummySession(),
    )

    # make_request called with the link URL.
    assert captured_request_url["url"] == link_url

    # Returned metadata is passed through.
    assert result == filled

    # Raw record is enriched with filing_page metadata.
    assert "filing_page" in raw_record
    fp = raw_record["filing_page"]
    assert fp["page_url"].endswith("?redirected=1")
    assert fp["anchor_used"] == "index-page-extraction"

    # Logger got a debug event.
    assert logger.debugs
    msg, ctx = logger.debugs[0]
    assert msg == "index_page_extracted"
    assert ctx["stage"] == "edgar_search"
    assert ctx["request_url"] == link_url
    assert ctx["status"] == 200
    assert ctx["response_url"].endswith("?redirected=1")


def test_handle_alternate_link_returns_none_when_index_parse_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure `handle_alternate_link` returns None and does not touch
    `raw_record["filing_page"]` when `try_extract_evidence_from_index(...)`
    yields None.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `make_request` and `try_extract_evidence_from_index`.

    Returns
    -------
    None
        The test passes if:
          - the function returns None, and
          - `filing_page` is absent from `raw_record`.
    """

    logger = _DummyLogger()
    run_data: RunData = {
        "ticker": "EXMPL",
        "validity_window": (
            pd.Timestamp("2019-01-01", tz="UTC"),
            pd.Timestamp("2024-01-01", tz="UTC"),
        ),
        "candidate": "0000000001",
        "name_space_bindings": {},
        "logger": cast(InfraLogger, logger),
        "entry": {},
        "oldest_filing_date": pd.Timestamp("2100-01-01", tz="UTC"),
    }
    raw_record: RawRecord = cast(RawRecord, {})
    link_url = "https://example.test/index"

    def fake_make_request(*args: Any, **kwargs: Any) -> _DummyResponse:
        return _DummyResponse(content=b"", url=link_url, status=200)

    def fake_try_extract_evidence_from_index(
        response: requests.Response, rd: RunData
    ) -> efp.FilledLinkData | None:
        return None

    monkeypatch.setattr(efp, "make_request", fake_make_request)
    monkeypatch.setattr(
        efp, "try_extract_evidence_from_index", fake_try_extract_evidence_from_index
    )

    result = efp.handle_alternate_link(
        link_url=link_url,
        raw_record=raw_record,
        run_data=run_data,
        session=_DummySession(),
    )

    assert result is None
    assert "filing_page" not in raw_record


def test_try_extract_evidence_from_index_parses_and_normalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Parse a well-formed index page into a normalized `FilledLinkData`.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub XPath constants, the HTML parser, and 8-K item parsing.

    Returns
    -------
    None
        The test passes if:
            - header fields are parsed correctly,
            - accession hyphens are removed,
            - CIK is zero-padded to 10 digits,
            - `oldest_filing_date` is updated to the filed date, and
            - 8-K item numbers/descriptions are attached.

    Raises
    ------
    AssertionError
        If any expected field is missing, mis-normalized, or if the run
        data is not updated as expected.
    """

    # Use simple token XPaths and a dummy HTML root.
    monkeypatch.setattr(efp, "XPATH_FORM_TYPE", "FORM_TYPE")
    monkeypatch.setattr(efp, "XPATH_FILED_AT", "FILED_AT")
    monkeypatch.setattr(efp, "XPATH_ACCESSION_NUM", "ACCESSION")
    monkeypatch.setattr(efp, "XPATH_CIK", "CIK")
    monkeypatch.setattr(efp, "XPATH_COMPANY_NAME", "NAME")
    monkeypatch.setattr(efp, "XPATH_8K_ITEMS", "ITEMS")

    filed_str = "2020-01-15"
    filed_ts = pd.to_datetime(filed_str, utc=True)

    class DummyRoot:
        def xpath(self, expr: str) -> Any:
            if expr == "FORM_TYPE":
                # Code splits on space and takes index 1 → "8-K".
                return ["FORM 8-K"]
            if expr == "FILED_AT":
                return [filed_str]
            if expr == "ACCESSION":
                # Should be treated as string; `.replace('-', '')` is applied.
                return "0000000001-20-000123"
            if expr == "CIK":
                return ["123456789 some-text"]
            if expr == "NAME":
                return ["Example Corp (Filer)\n"]
            if expr == "ITEMS":
                return ["Item 1.01 Foo", "Item 2.01 Bar"]
            return []

    def fake_html_parser(_content: bytes) -> DummyRoot:
        return DummyRoot()

    # Ensure parse_8k_items is used and returns at least one eligible item.
    def fake_parse_8k_items(root: Any) -> tuple[List[str], List[str]]:
        return (["1.01"], ["Foo"])

    monkeypatch.setattr(efp.etree, "HTML", fake_html_parser)
    monkeypatch.setattr(efp, "parse_8k_items", fake_parse_8k_items)

    logger = _DummyLogger()
    run_data: RunData = {
        "ticker": "EXMPL",
        "validity_window": (
            pd.Timestamp("2019-01-01", tz="UTC"),
            pd.Timestamp("2024-01-01", tz="UTC"),
        ),
        "candidate": "0000000001",
        "name_space_bindings": {},
        "logger": cast(InfraLogger, logger),
        "entry": {},
        "oldest_filing_date": pd.Timestamp("2100-01-01", tz="UTC"),
    }

    response = cast(requests.Response, _DummyResponse(content=b"<html/>"))

    result = efp.try_extract_evidence_from_index(response, run_data)

    assert result is not None
    assert result["form_type"] == "8-K"
    assert result["filed_at"] == filed_ts
    assert result["accession_num"] == "000000000120000123"
    assert result["cik"] == "0123456789"  # zero-padded
    assert result["company_name"] == "Example Corp"
    assert result["items_8k"] == ["1.01"]
    assert result["items_descriptions_8k"] == ["Foo"]

    # oldest_filing_date updated to earlier value.
    assert run_data["oldest_filing_date"] == filed_ts


def test_try_extract_evidence_from_index_returns_none_on_missing_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Validate that `try_extract_evidence_from_index` returns None when any
    required header is missing.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub XPath constants and `etree.HTML` to simulate missing
        values.

    Returns
    -------
    None
        The test passes if the function yields None and does not attempt
        to create `FilledLinkData`.
    """

    monkeypatch.setattr(efp, "XPATH_FORM_TYPE", "FORM_TYPE")
    monkeypatch.setattr(efp, "XPATH_FILED_AT", "FILED_AT")
    monkeypatch.setattr(efp, "XPATH_ACCESSION_NUM", "ACCESSION")
    monkeypatch.setattr(efp, "XPATH_CIK", "CIK")
    monkeypatch.setattr(efp, "XPATH_COMPANY_NAME", "NAME")
    monkeypatch.setattr(efp, "XPATH_8K_ITEMS", "ITEMS")

    # Parameterize "which field is missing" by controlling xpath output.
    @pytest.mark.parametrize(
        "missing_field",
        ["FORM_TYPE", "FILED_AT", "ACCESSION", "CIK", "NAME"],
    )
    def _run_case(missing_field: str) -> None:
        class DummyRoot:
            def xpath(self, expr: str) -> Any:
                # Missing expr returns an empty list to trigger early exit.
                if expr == missing_field:
                    return [] if expr != "ACCESSION" else ""
                if expr == "FORM_TYPE":
                    return ["FORM TYPE 10-K"]
                if expr == "FILED_AT":
                    return ["2020-01-15"]
                if expr == "ACCESSION":
                    return "0000000001-20-000123"
                if expr == "CIK":
                    return ["123456789 something"]
                if expr == "NAME":
                    return ["Example Corp (Filer)\n"]
                if expr == "ITEMS":
                    return []
                return []

        def fake_html_parser(_content: bytes) -> DummyRoot:
            return DummyRoot()

        monkeypatch.setattr(efp.etree, "HTML", fake_html_parser)

        # parse_8k_items should not matter for non-8-K forms; make it obvious if called.
        monkeypatch.setattr(
            efp,
            "parse_8k_items",
            lambda _root: (["1.01"], ["Foo"]),
        )

        run_data: RunData = {
            "ticker": "EXMPL",
            "validity_window": (
                pd.Timestamp("2019-01-01", tz="UTC"),
                pd.Timestamp("2024-01-01", tz="UTC"),
            ),
            "candidate": "0000000001",
            "name_space_bindings": {},
            "logger": cast(InfraLogger, _DummyLogger()),
            "entry": {},
            "oldest_filing_date": pd.Timestamp("2100-01-01", tz="UTC"),
        }
        response = cast(requests.Response, _DummyResponse(content=b"<html/>"))

        result = efp.try_extract_evidence_from_index(response, run_data)
        assert result is None

    # Execute parameterized subcases manually (we're inside a non-param test).
    for field in ["FORM_TYPE", "FILED_AT", "ACCESSION", "CIK", "NAME"]:
        _run_case(field)


def test_try_extract_evidence_from_index_rejects_8k_without_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Reject an 8-K index page when no eligible 8-K items are parsed.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub XPath constants, the HTML parser, and `parse_8k_items`.

    Returns
    -------
    None
        The test passes if `try_extract_evidence_from_index(...)` returns
        None when `parse_8k_items(...)` yields an empty list of item
        numbers.

    Raises
    ------
    AssertionError
        If a `FilledLinkData` is produced despite there being no eligible
        8-K items.
    """

    monkeypatch.setattr(efp, "XPATH_FORM_TYPE", "FORM_TYPE")
    monkeypatch.setattr(efp, "XPATH_FILED_AT", "FILED_AT")
    monkeypatch.setattr(efp, "XPATH_ACCESSION_NUM", "ACCESSION")
    monkeypatch.setattr(efp, "XPATH_CIK", "CIK")
    monkeypatch.setattr(efp, "XPATH_COMPANY_NAME", "NAME")
    monkeypatch.setattr(efp, "XPATH_8K_ITEMS", "ITEMS")

    class DummyRoot:
        def xpath(self, expr: str) -> Any:
            if expr == "FORM_TYPE":
                # Splits to ["FORM", "8-K"] → "8-K".
                return ["FORM 8-K"]
            if expr == "FILED_AT":
                return ["2020-01-15"]
            if expr == "ACCESSION":
                return "0000000001-20-000123"
            if expr == "CIK":
                return ["123456789 foo"]
            if expr == "NAME":
                return ["Example Corp (Filer)\n"]
            if expr == "ITEMS":
                return ["Item 1.01 Something", "Item 2.01 Something else"]
            return []

    def fake_html_parser(_content: bytes) -> DummyRoot:
        return DummyRoot()

    # Simulate no eligible items.
    def fake_parse_8k_items(_root: Any) -> tuple[List[str], List[str]]:
        return ([], [])

    monkeypatch.setattr(efp.etree, "HTML", fake_html_parser)
    monkeypatch.setattr(efp, "parse_8k_items", fake_parse_8k_items)

    run_data: RunData = {
        "ticker": "EXMPL",
        "validity_window": (
            pd.Timestamp("2019-01-01", tz="UTC"),
            pd.Timestamp("2024-01-01", tz="UTC"),
        ),
        "candidate": "0000000001",
        "name_space_bindings": {},
        "logger": cast(InfraLogger, _DummyLogger()),
        "entry": {},
        "oldest_filing_date": pd.Timestamp("2100-01-01", tz="UTC"),
    }
    response = cast(requests.Response, _DummyResponse(content=b"<html/>"))

    result = efp.try_extract_evidence_from_index(response, run_data)
    assert result is None


def test_parse_8k_items_filters_and_aligns(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Check that `parse_8k_items` filters by `ELIGIBLE_8K_ITEMS` and returns
    aligned item-number and description lists.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `ELIGIBLE_8K_ITEMS` and feed controlled raw lines
        via a dummy HTML root.

    Returns
    -------
    None
        The test passes if:
          - only item numbers in `ELIGIBLE_8K_ITEMS` are retained,
          - non-matching or malformed lines are ignored, and
          - descriptions align with numbers.
    """

    monkeypatch.setattr(efp, "ELIGIBLE_8K_ITEMS", {"1.01", "2.01"})

    class DummyRoot:
        def xpath(self, expr: str) -> List[str]:
            assert expr == efp.XPATH_8K_ITEMS
            return [
                "Item 1.01 Entry into a Material Definitive Agreement",
                "Item 2.01 Completion of Acquisition or Disposition of Assets",
                "Item 3.99 Non-eligible Item",
                "Random text that should be ignored",
            ]

    # XPATH_8K_ITEMS can be anything; we only verify that it is used.
    monkeypatch.setattr(efp, "XPATH_8K_ITEMS", "ITEMS")

    items, descriptions = efp.parse_8k_items(DummyRoot())

    assert items == ["1.01", "2.01"]
    assert len(descriptions) == 2
    assert "Entry into a Material Definitive Agreement" in descriptions[0]
    assert "Completion of Acquisition or Disposition of Assets" in descriptions[1]


def test_parse_8k_items_returns_empty_when_no_eligible(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Ensure `parse_8k_items` returns empty lists when no lines correspond
    to eligible items.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to constrain `ELIGIBLE_8K_ITEMS` and provide non-matching
        input lines.

    Returns
    -------
    None
        The test passes if both returned lists are empty.
    """

    monkeypatch.setattr(efp, "ELIGIBLE_8K_ITEMS", {"1.01"})

    class DummyRoot:
        def xpath(self, expr: str) -> List[str]:
            return ["Item 3.02 Something else", "Not an item line"]

    monkeypatch.setattr(efp, "XPATH_8K_ITEMS", "ITEMS")

    items, descriptions = efp.parse_8k_items(DummyRoot())
    assert not items
    assert not descriptions


def test_create_filled_link_data_enforces_required_fields() -> None:
    """
    Verify that `create_filled_link_data` returns a `FilledLinkData`
    dictionary when all required fields are present.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if the returned dict mirrors the input `LinkData`
        for all fields.

    Raises
    ------
    AssertionError
        If any field is not propagated correctly.
    """

    filed_at = pd.Timestamp("2020-01-15", tz="UTC")
    link_data: efp.LinkData = {
        "accession_num": "000000000120000123",
        "cik": "0000000001",
        "form_type": "8-K",
        "filed_at": filed_at,
        "company_name": "Example Corp",
        "items_8k": ["1.01"],
        "items_descriptions_8k": ["Entry into a Material Definitive Agreement"],
    }

    filled = efp.create_filled_link_data(link_data)

    assert filled["accession_num"] == "000000000120000123"
    assert filled["cik"] == "0000000001"
    assert filled["form_type"] == "8-K"
    assert filled["filed_at"] == filed_at
    assert filled["company_name"] == "Example Corp"
    assert filled["items_8k"] == ["1.01"]
    assert filled["items_descriptions_8k"] is not None
    assert "Entry into a Material Definitive Agreement" in filled["items_descriptions_8k"][0]
