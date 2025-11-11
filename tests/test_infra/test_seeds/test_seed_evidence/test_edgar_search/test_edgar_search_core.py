"""
Purpose
-------
Unit tests for the EDGAR Atom company-feed harvesting core.

Key behaviors
-------------
- Verify that `init_run_data` defers to `extract_entries_and_namespace(...)`
  and `create_run_data(...)` with the expected arguments.
- Ensure that `append_collected_evidence`:
  - appends a `MappingEvidence` object and emits a DEBUG log when evidence is
    present, and behaves as a no-op when `gathered_data` is `None`.
- Validate that `extract_data_from_links`:
  - locates `rel="alternate"` links via `find_all_elements(...)`,
  - calls `set_atom_entry(...)` with the correct parameters, and
  - returns the `FilledLinkData` produced by `handle_alternate_link(...)`, or
    `None` when no suitable link exists.
- Confirm that `extract_entry_data`:
  - calls `extract_data_from_links(...)` and `within_validity_window(...)`,
  - only builds `MappingEvidence` when the filing date lies inside the
    validity window, and surfaces the constructed evidence object.
- Check that `handle_entry`:
  - passes the same buffer to `extract_entry_data(...)` and `append_collected_evidence(...)`,
    and threads candidate and logger through correctly.
- Verify that `fetch_edgar_evidence`:
  - builds a URL once per page via `build_url(...)`,
  - wraps the request in `make_request(...)`,
  - initializes run-data with `init_run_data(...)`,
  - iterates entries, calling `handle_entry(...)` for eligible ones, and
  - terminates cleanly when `evaluate_page_break_conditions(...)` returns True.

Conventions
-----------
- External network and XML parsing are stubbed with lightweight doubles.
- Logging is captured via a minimal `_DummyLogger` that mimics the subset of
  `InfraLogger` methods used in the core module.
- Tests deliberately avoid exercising the infinite `while True` loop by
  forcing `evaluate_page_break_conditions(...)` to return True after a
  single page.

Downstream usage
----------------
Run with `pytest` as part of the CI suite. These tests document how the
EDGAR Atom harvesting core composes the paginator, entry-level processing,
and evidence accumulation API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, cast

import pandas as pd
import pytest
import requests

from infra.logging.infra_logger import InfraLogger
from infra.seeds.seed_evidence.edgar_search import edgar_search_core as esc
from infra.seeds.seed_evidence.records.evidence_record import MappingEvidence
from infra.seeds.seed_evidence.records.raw_record import RawRecord
from infra.seeds.seed_evidence.seed_evidence_types import Entries, ValidityWindow


class _DummyLogger:
    """
    Purpose
    -------
    Minimal stand-in for `InfraLogger` for EDGAR core tests.

    Key behaviors
    -------------
    - Collects `.info(...)`, `.warning(...)`, and `.debug(...)` calls.
    - Stores message identifiers and structured contexts for assertions.

    Parameters
    ----------
    None

    Attributes
    ----------
    infos : list[tuple[str, dict[str, Any]]]
        Recorded info-level messages.
    warnings : list[tuple[str, dict[str, Any]]]
        Recorded warning-level messages.
    debugs : list[tuple[str, dict[str, Any]]]
        Recorded debug-level messages.

    Notes
    -----
    - Only implements the subset of the real logger interface required
      by the module under test.
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


class _DummyEvidence:
    """
    Purpose
    -------
    Lightweight stand-in for `MappingEvidence` for append/record tests.

    Key behaviors
    -------------
    - Exposes the minimal attribute surface used by `append_collected_evidence`.
    - Does not depend on the real `MappingEvidence` constructor.

    Parameters
    ----------
    evidence_id : str
        Identifier used as a dict key in accumulators.
    form_type : str
        SEC form type string (e.g., "8-K").
    candidate_cik : str
        Candidate CIK or ticker associated with the evidence.
    filed_at : pandas.Timestamp
        Filing date used in debug logging.
    accession_num : str
        EDGAR accession number for the filing.
    """

    def __init__(
        self,
        evidence_id: str,
        form_type: str,
        candidate_cik: str,
        filed_at: pd.Timestamp,
        accession_num: str,
    ) -> None:
        self.evidence_id = evidence_id
        self.form_type = form_type
        self.candidate_cik = candidate_cik
        self.filed_at = filed_at
        self.accession_num = accession_num


class _DummySession(requests.Session):
    """
    Purpose
    -------
    Minimal `requests.Session` subclass for type-safe tests.

    Key behaviors
    -------------
    - Inherits from `requests.Session` so mypy accepts it.
    - No overridden behavior; HTTP methods are never exercised.
    """


@pytest.mark.parametrize("has_evidence", [True, False])
def test_append_collected_evidence_updates_buffer_and_logs(has_evidence: bool) -> None:
    """
    Ensure that `append_collected_evidence` behaves correctly with and
    without gathered evidence.

    Parameters
    ----------
    has_evidence : bool
        Controls whether a non-None `MappingEvidence` is passed to the
        function under test.

    Returns
    -------
    None
        The test passes if:
            - when `has_evidence` is True, the evidence is added to the
              accumulator and a DEBUG log is emitted, and
            - when `has_evidence` is False, neither the accumulator nor
              the logger are modified.

    Raises
    ------
    AssertionError
        If the accumulator or debug logging does not match expectations.
    """

    logger = _DummyLogger()
    buffer: Dict[str, MappingEvidence] = {}

    evidence: MappingEvidence | None
    if has_evidence:
        dummy = _DummyEvidence(
            evidence_id="ev-1",
            form_type="8-K",
            candidate_cik="0000000001",
            filed_at=pd.Timestamp("2020-01-10", tz="UTC"),
            accession_num="0000000001-20-000001",
        )
        evidence = cast(MappingEvidence, dummy)
    else:
        evidence = None

    esc.append_collected_evidence(
        candidate="0000000001",
        logger=cast(InfraLogger, logger),
        gathered_data=evidence,
        current_collected_evidence=buffer,
    )

    if has_evidence:
        assert "ev-1" in buffer
        assert buffer["ev-1"] is evidence
        assert logger.debugs, "Expected a debug log when evidence is recorded"
        msg, ctx = logger.debugs[0]
        assert msg == "evidence_recorded"
        assert ctx["candidate_or_ticker"] == "0000000001"
        assert ctx["form_type"] == "8-K"
        assert ctx["cik"] == "0000000001"
        assert ctx["accession"] == "0000000001-20-000001"
    else:
        assert not buffer
        assert not logger.debugs


def test_extract_data_from_links_finds_alternate_and_calls_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Check that `extract_data_from_links` finds the `rel="alternate"` link,
    persists Atom metadata, and calls `handle_alternate_link(...)`.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `find_all_elements`, `set_atom_entry`, and
        `handle_alternate_link`.

    Returns
    -------
    None
        The test passes if:
            - the first `rel="alternate"` link with a non-None href is used,
            - `set_atom_entry(...)` is called with that href and the updated
              timestamp, and the value returned by `handle_alternate_link(...)` is surfaced.

    Raises
    ------
    AssertionError
        If the wrong link is used or either helper is not invoked correctly.
    """

    class DummyLink:
        def __init__(self, rel: str | None, href: str | None) -> None:
            self._rel = rel
            self._href = href

        def get(self, key: str, default: Any = None) -> Any:
            if key == "rel":
                return self._rel
            if key == "href":
                return self._href
            return default

    entry = object()
    ns_bindings = {"ns": "urn:test"}
    run_data: esc.RunData = {
        "entry": entry,
        "name_space_bindings": ns_bindings,
        "ticker": "AAA",
        "candidate": "0000000001",
        "validity_window": (
            pd.Timestamp("2020-01-01", tz="UTC"),
            pd.Timestamp("2020-02-01", tz="UTC"),
        ),
        "logger": cast(InfraLogger, _DummyLogger()),
        "oldest_filing_date": pd.Timestamp("2100-01-01", tz="UTC"),
    }
    raw_record: RawRecord = cast(RawRecord, object())
    updated_ts = pd.Timestamp("2020-01-15", tz="UTC")

    links = [
        DummyLink(rel="other", href="https://ignore.test"),
        DummyLink(rel="alternate", href="https://example.test/filing"),
    ]

    def fake_find_all_elements(entry_arg: Any, tag: str, ns: Any) -> List[Any]:
        assert entry_arg is entry
        assert tag == "link"
        assert ns is ns_bindings
        return links

    captured_set_atom: Dict[str, Any] = {}

    def fake_set_atom_entry(
        entry_arg: Any,
        href: str,
        updated: pd.Timestamp,
        record: RawRecord,
        ns: Any,
    ) -> None:
        captured_set_atom.update(
            {
                "entry": entry_arg,
                "href": href,
                "updated": updated,
                "record": record,
                "ns": ns,
            }
        )

    expected_link_data = cast(
        esc.FilledLinkData,
        {
            "cik": "0000000001",
            "form_type": "8-K",
            "filed_at": pd.Timestamp("2020-01-10", tz="UTC"),
            "accession_num": "0000000001-20-000001",
            "company_name": "Example Corp",
            "items_8k": [],
            "items_descriptions_8k": [],
        },
    )

    def fake_handle_alternate_link(
        href: str, record: RawRecord, rd: esc.RunData, session: requests.Session
    ) -> esc.FilledLinkData:
        assert href == "https://example.test/filing"
        assert rd is run_data
        return expected_link_data

    monkeypatch.setattr(esc, "find_all_elements", fake_find_all_elements)
    monkeypatch.setattr(esc, "set_atom_entry", fake_set_atom_entry)
    monkeypatch.setattr(esc, "handle_alternate_link", fake_handle_alternate_link)

    result = esc.extract_data_from_links(
        run_data=run_data,
        raw_record=raw_record,
        updated_ts=updated_ts,
        session=_DummySession(),
    )

    assert result is expected_link_data
    assert captured_set_atom["entry"] is entry
    assert captured_set_atom["href"] == "https://example.test/filing"
    assert captured_set_atom["updated"] == updated_ts
    assert captured_set_atom["record"] is raw_record
    assert captured_set_atom["ns"] is ns_bindings


def test_extract_data_from_links_returns_none_when_no_alternate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Validate that `extract_data_from_links` returns None when no suitable
    `rel="alternate"` link exists.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `find_all_elements` so that only non-alternate or
        href-less links are returned.

    Returns
    -------
    None
        The test passes if the function returns None and neither
        `set_atom_entry(...)` nor `handle_alternate_link(...)` are invoked.

    Raises
    ------
    AssertionError
        If any helper is unexpectedly called or a non-None value is returned.
    """

    class DummyLink:
        def __init__(self, rel: str | None, href: str | None) -> None:
            self._rel = rel
            self._href = href

        def get(self, key: str, default: Any = None) -> Any:
            if key == "rel":
                return self._rel
            if key == "href":
                return self._href
            return default

    run_data: esc.RunData = {
        "entry": object(),
        "name_space_bindings": {"ns": "urn:test"},
        "ticker": "AAA",
        "candidate": "0000000001",
        "validity_window": (
            pd.Timestamp("2020-01-01", tz="UTC"),
            pd.Timestamp("2020-02-01", tz="UTC"),
        ),
        "logger": cast(InfraLogger, _DummyLogger()),
        "oldest_filing_date": pd.Timestamp("2100-01-01", tz="UTC"),
    }

    raw_record: RawRecord = cast(RawRecord, object())
    updated_ts = pd.Timestamp("2020-01-15", tz="UTC")

    def fake_find_all_elements(entry: Any, tag: str, ns: Any) -> List[Any]:
        return [
            DummyLink(rel="other", href="https://ignore.test"),
            DummyLink(rel="alternate", href=None),
        ]

    monkeypatch.setattr(esc, "find_all_elements", fake_find_all_elements)

    set_called = {"count": 0}
    handle_called = {"count": 0}

    def fake_set_atom_entry(*args: Any, **kwargs: Any) -> None:
        set_called["count"] += 1

    def fake_handle_alternate_link(*args: Any, **kwargs: Any) -> esc.FilledLinkData:
        handle_called["count"] += 1
        return cast(esc.FilledLinkData, {})

    monkeypatch.setattr(esc, "set_atom_entry", fake_set_atom_entry)
    monkeypatch.setattr(esc, "handle_alternate_link", fake_handle_alternate_link)

    result = esc.extract_data_from_links(
        run_data=run_data,
        raw_record=raw_record,
        updated_ts=updated_ts,
        session=_DummySession(),
    )

    assert result is None
    assert set_called["count"] == 0
    assert handle_called["count"] == 0


def test_extract_entry_data_builds_evidence_when_in_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Check that `extract_entry_data` builds and returns `MappingEvidence`
    when the parsed filing date lies inside the validity window.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `extract_data_from_links`, `within_validity_window`,
        and `build_mapping_evidence`.

    Returns
    -------
    None
        The test passes if:
            - a non-None `FilledLinkData` is returned from
              `extract_data_from_links(...)`,
            - `within_validity_window(...)` returns True, and
            - the `MappingEvidence` constructed by `build_mapping_evidence(...)`
              is returned.

    Raises
    ------
    AssertionError
        If `build_mapping_evidence(...)` is not called with the expected
        arguments or the result is not surfaced correctly.
    """

    window: ValidityWindow = (
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2020-02-01", tz="UTC"),
    )
    run_data: esc.RunData = {
        "ticker": "AAA",
        "candidate": "0000000001",
        "validity_window": window,
        "name_space_bindings": {"ns": "urn"},
        "logger": cast(InfraLogger, _DummyLogger()),
        "entry": object(),
        "oldest_filing_date": pd.Timestamp("2100-01-01", tz="UTC"),
    }
    updated_ts = pd.Timestamp("2020-01-10", tz="UTC")
    raw_record: RawRecord = cast(RawRecord, object())

    link_data: esc.FilledLinkData = cast(
        esc.FilledLinkData,
        {
            "cik": "0000000001",
            "form_type": "8-K",
            "filed_at": pd.Timestamp("2020-01-15", tz="UTC"),
            "accession_num": "0000000001-20-000001",
            "company_name": "Example Corp",
            "items_8k": [],
            "items_descriptions_8k": [],
        },
    )

    def fake_extract_data_from_links(
        rd: esc.RunData, rr: RawRecord, upd: pd.Timestamp, sess: requests.Session
    ) -> esc.FilledLinkData | None:
        assert rd is run_data
        assert rr is raw_record
        assert upd == updated_ts
        return link_data

    def fake_within_validity_window(dt_val: pd.Timestamp, vw: ValidityWindow) -> bool:
        assert vw == window
        return True

    expected_evidence = cast(
        MappingEvidence,
        _DummyEvidence(
            evidence_id="ev-123",
            form_type="8-K",
            candidate_cik="0000000001",
            filed_at=link_data["filed_at"],
            accession_num=link_data["accession_num"],
        ),
    )

    captured_build_args: Dict[str, Any] = {}

    def fake_build_mapping_evidence(
        ticker: str,
        cik: str,
        filed_at: pd.Timestamp,
        vw: ValidityWindow,
        source: str,
        record: RawRecord,
        form_type: str,
        accession_num: str,
        company_name: str,
        items_8k: List[str],
        items_descriptions_8k: List[str],
    ) -> MappingEvidence:
        captured_build_args.update(
            {
                "ticker": ticker,
                "cik": cik,
                "filed_at": filed_at,
                "vw": vw,
                "form_type": form_type,
                "accession_num": accession_num,
                "company_name": company_name,
            }
        )
        return expected_evidence

    monkeypatch.setattr(esc, "extract_data_from_links", fake_extract_data_from_links)
    monkeypatch.setattr(esc, "within_validity_window", fake_within_validity_window)
    monkeypatch.setattr(esc, "build_mapping_evidence", fake_build_mapping_evidence)

    result = esc.extract_entry_data(
        run_data=run_data,
        updated_ts=updated_ts,
        session=_DummySession(),
        raw_record=raw_record,
    )

    assert result is expected_evidence
    assert captured_build_args["ticker"] == "AAA"
    assert captured_build_args["cik"] == "0000000001"
    assert captured_build_args["filed_at"] == link_data["filed_at"]
    assert captured_build_args["vw"] == window
    assert captured_build_args["form_type"] == "8-K"
    assert captured_build_args["accession_num"] == "0000000001-20-000001"
    assert captured_build_args["company_name"] == "Example Corp"


def test_extract_entry_data_returns_none_when_out_of_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Validate that `extract_entry_data` returns None when the filing date
    is outside the validity window.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `extract_data_from_links` and
        `within_validity_window(...)`.

    Returns
    -------
    None
        The test passes if:
            - `extract_data_from_links(...)` returns a non-None `FilledLinkData`,
            - `within_validity_window(...)` returns False, and
            - no evidence is constructed (function returns None).

    Raises
    ------
    AssertionError
        If evidence is incorrectly built or a non-None value is returned.
    """

    window: ValidityWindow = (
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2020-02-01", tz="UTC"),
    )
    run_data: esc.RunData = {
        "ticker": "AAA",
        "candidate": "0000000001",
        "validity_window": window,
        "name_space_bindings": {"ns": "urn"},
        "logger": cast(InfraLogger, _DummyLogger()),
        "entry": object(),
        "oldest_filing_date": pd.Timestamp("2100-01-01", tz="UTC"),
    }
    updated_ts = pd.Timestamp("2020-01-10", tz="UTC")
    raw_record: RawRecord = cast(RawRecord, object())

    link_data: esc.FilledLinkData = cast(
        esc.FilledLinkData,
        {
            "cik": "0000000001",
            "form_type": "8-K",
            "filed_at": pd.Timestamp("2020-03-01", tz="UTC"),
            "accession_num": "0000000001-20-000002",
            "company_name": "Example Corp",
            "items_8k": [],
            "items_descriptions_8k": [],
        },
    )

    monkeypatch.setattr(
        esc,
        "extract_data_from_links",
        lambda rd, rr, upd, sess: link_data,
    )
    monkeypatch.setattr(
        esc,
        "within_validity_window",
        lambda dt_val, vw: False,
    )

    called_build = {"count": 0}

    def fake_build_mapping_evidence(*args: Any, **kwargs: Any) -> MappingEvidence:
        called_build["count"] += 1
        return cast(MappingEvidence, object())

    monkeypatch.setattr(esc, "build_mapping_evidence", fake_build_mapping_evidence)

    result = esc.extract_entry_data(
        run_data=run_data,
        updated_ts=updated_ts,
        session=_DummySession(),
        raw_record=raw_record,
    )

    assert result is None
    assert called_build["count"] == 0


def test_handle_entry_routes_through_extract_and_append(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Check that `handle_entry` passes the same buffer through
    `extract_entry_data(...)` and `append_collected_evidence(...)`.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `extract_entry_data` and `append_collected_evidence`
        with side-effectful fakes.

    Returns
    -------
    None
        The test passes if:
            - `extract_entry_data(...)` is called with the provided
              `run_data`, `updated_ts`, `session`, and `raw_record`, and
            - `append_collected_evidence(...)` receives the candidate,
              logger, evidence returned by the extractor, and the same
              buffer instance.

    Raises
    ------
    AssertionError
        If either call does not receive the expected arguments or the
        buffer is not threaded correctly.
    """

    logger = cast(InfraLogger, _DummyLogger())
    window: ValidityWindow = (
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2020-02-01", tz="UTC"),
    )
    run_data: esc.RunData = {
        "ticker": "AAA",
        "candidate": "0000000001",
        "validity_window": window,
        "name_space_bindings": {"ns": "urn"},
        "logger": logger,
        "entry": object(),
        "oldest_filing_date": pd.Timestamp("2100-01-01", tz="UTC"),
    }
    updated_ts = pd.Timestamp("2020-01-10", tz="UTC")
    raw_record: RawRecord = cast(RawRecord, object())
    buffer: Dict[str, MappingEvidence] = {}
    session = _DummySession()

    expected_evidence = cast(
        MappingEvidence,
        _DummyEvidence(
            evidence_id="ev-1",
            form_type="8-K",
            candidate_cik="0000000001",
            filed_at=updated_ts,
            accession_num="0000000001-20-000003",
        ),
    )

    captured_extract_args: Dict[str, Any] = {}
    captured_append_args: Dict[str, Any] = {}

    def fake_extract_entry_data(
        rd: esc.RunData, upd: pd.Timestamp, sess: requests.Session, rr: RawRecord
    ) -> MappingEvidence | None:
        captured_extract_args.update(
            {"run_data": rd, "updated_ts": upd, "session": sess, "raw_record": rr}
        )
        return expected_evidence

    def fake_append_collected_evidence(
        candidate: str,
        logger_param: InfraLogger,
        gathered_data: MappingEvidence | None,
        current_buffer: Dict[str, MappingEvidence],
    ) -> None:
        captured_append_args.update(
            {
                "candidate": candidate,
                "logger": logger_param,
                "gathered_data": gathered_data,
                "buffer": current_buffer,
            }
        )

    monkeypatch.setattr(esc, "extract_entry_data", fake_extract_entry_data)
    monkeypatch.setattr(esc, "append_collected_evidence", fake_append_collected_evidence)

    esc.handle_entry(
        run_data=run_data,
        current_collected_evidence=buffer,
        updated_ts=updated_ts,
        session=session,
        raw_record=raw_record,
    )

    assert captured_extract_args["run_data"] is run_data
    assert captured_extract_args["updated_ts"] == updated_ts
    assert captured_extract_args["session"] is session
    assert captured_extract_args["raw_record"] is raw_record

    assert captured_append_args["candidate"] == "0000000001"
    assert captured_append_args["logger"] is logger
    assert captured_append_args["gathered_data"] is expected_evidence
    assert captured_append_args["buffer"] is buffer


def test_fetch_edgar_evidence_drives_single_page_and_uses_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that `fetch_edgar_evidence` drives a single-page date-walk,
    delegates to helpers, and terminates when the page-break condition
    is met.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `build_url`, `make_request`, `init_run_data`,
        `evaluate_updated_timestamp`, `check_entry_form_type_eligibility`,
        `generate_initial_raw_record`, `handle_entry`, and
        `evaluate_page_break_conditions`.

    Returns
    -------
    None
        The test passes if:
            - `build_url(...)` is called with the candidate, window, logger,
              and initial `page_cursor=None`,
            - `handle_entry(...)` is invoked once per eligible entry, and
            - the function returns after a single page when
              `evaluate_page_break_conditions(...)` returns True.

    Raises
    ------
    AssertionError
        If any helper is not called with expected arguments or the while-loop
        does not terminate as desired.
    """

    ticker = "AAA"
    candidate = "0000000001"
    window: ValidityWindow = (
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2020-02-01", tz="UTC"),
    )
    logger = cast(InfraLogger, _DummyLogger())
    buffer: Dict[str, MappingEvidence] = {}
    session = _DummySession()

    entries: Entries = cast(Entries, ["e1", "e2"])
    initial_oldest = pd.Timestamp("2100-01-01", tz="UTC")
    dummy_run_data: esc.RunData = {
        "ticker": ticker,
        "candidate": candidate,
        "validity_window": window,
        "name_space_bindings": {"ns": "urn"},
        "logger": logger,
        "entry": None,
        "oldest_filing_date": initial_oldest,
    }

    captured_build_url: Dict[str, Any] = {}
    captured_handle_calls: List[Dict[str, Any]] = []
    page_break_calls: List[Tuple[Entries, esc.RunData]] = []
    updated_ts = pd.Timestamp("2020-01-10", tz="UTC")

    def fake_build_url(
        cand: str,
        vw: ValidityWindow,
        log: InfraLogger,
        cursor: pd.Timestamp | None,
    ) -> str:
        captured_build_url.update(
            {"candidate": cand, "window": vw, "logger": log, "cursor": cursor}
        )
        return "https://example.test/edgar"

    class DummyResponse:
        def __enter__(self) -> "DummyResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def fake_make_request(
        url: str,
        expect_json: bool,
        max_retries: int = 5,
        backoff_factor: float = 0.5,
        timeout: tuple[float, float] = (3.5, 10.0),
        sleep_time: float = 0.0,
        session: requests.Session | None = None,
    ) -> DummyResponse:
        assert url == "https://example.test/edgar"
        return DummyResponse()

    def fake_init_run_data(
        ticker_param: str,
        candidate_param: str,
        validity_window_param: ValidityWindow,
        logger_param: InfraLogger,
        response: requests.Response,
        page_cursor: pd.Timestamp | None,
    ) -> tuple[Entries, esc.RunData]:
        assert ticker_param == ticker
        assert candidate_param == candidate
        assert validity_window_param == window
        assert logger_param is logger
        assert page_cursor is None
        return entries, dummy_run_data

    def fake_evaluate_updated_timestamp(rd: esc.RunData) -> pd.Timestamp | None:
        # Always in-window, so we exercise handle_entry.
        return updated_ts

    def fake_check_entry_form_type_eligibility(entry: Any, ns: Any) -> bool:
        # Both entries are eligible.
        return True

    def fake_generate_initial_raw_record(cand: str, vw: ValidityWindow) -> RawRecord:
        # Return a dummy object; contents not inspected.
        return cast(RawRecord, object())

    def fake_handle_entry(
        rd: esc.RunData,
        current_buffer: Dict[str, MappingEvidence],
        updated_ts_param: pd.Timestamp,
        sess: requests.Session,
        rr: RawRecord,
    ) -> None:
        captured_handle_calls.append(
            {
                "run_data": rd,
                "buffer": current_buffer,
                "updated_ts": updated_ts_param,
                "session": sess,
                "raw_record": rr,
            }
        )

    def fake_evaluate_page_break_conditions(ents: Entries, rd: esc.RunData) -> bool:
        page_break_calls.append((ents, rd))
        # Force a single-iteration loop.
        return True

    monkeypatch.setattr(esc, "build_url", fake_build_url)
    monkeypatch.setattr(esc, "make_request", fake_make_request)
    monkeypatch.setattr(esc, "init_run_data", fake_init_run_data)
    monkeypatch.setattr(esc, "evaluate_updated_timestamp", fake_evaluate_updated_timestamp)
    monkeypatch.setattr(
        esc, "check_entry_form_type_eligibility", fake_check_entry_form_type_eligibility
    )
    monkeypatch.setattr(esc, "generate_initial_raw_record", fake_generate_initial_raw_record)
    monkeypatch.setattr(esc, "handle_entry", fake_handle_entry)
    monkeypatch.setattr(esc, "evaluate_page_break_conditions", fake_evaluate_page_break_conditions)

    esc.fetch_edgar_evidence(
        ticker=ticker,
        candidate=candidate,
        validity_window=window,
        logger=logger,
        current_collected_evidence=buffer,
        session=session,
    )

    assert captured_build_url["candidate"] == candidate
    assert captured_build_url["window"] == window
    assert captured_build_url["logger"] is logger
    assert captured_build_url["cursor"] is None

    # Two entries, both eligible → two handle_entry calls.
    assert len(captured_handle_calls) == 2
    for call in captured_handle_calls:
        assert call["run_data"] is dummy_run_data
        assert call["buffer"] is buffer
        assert call["updated_ts"] == updated_ts
        assert call["session"] is session

    # Page-break evaluated once for the single page we simulated.
    assert len(page_break_calls) == 1
    ents, rd = page_break_calls[0]
    assert ents == entries
    assert rd is dummy_run_data
