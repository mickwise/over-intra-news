"""
Purpose
-------
Unit tests for wayback_snapshot_scrape.

Key behaviors
-------------
- Verify that `extract_table`:
  - prefers the primary XPath when it returns at least one table, and
  - falls back to the secondary XPath when the primary returns nothing.
- Validate that `find_column_mappings`:
  - maps CIK and ticker headers to 1-based indices, and
  - returns None when either header cannot be resolved.
- Ensure that `extract_rows`:
  - returns the row whose ticker cell matches exactly (case-insensitive), and
  - does not confuse strict matches with prefix matches (e.g., 'GL' vs 'GLW').
- Confirm that `find_candidate_cik`:
  - extracts digits from the CIK cell,
  - only accepts 10-digit CIK strings,
  - and emits a debug log when a CIK is found.
- Check that `handle_candidate`:
  - creates a new WayBackCandidate on first observation, and
  - updates first_seen / last_seen and URLs on subsequent observations.
- Verify that `scrape_snapshot`:
  - short-circuits when the table or column mappings cannot be found, and
  - for valid snapshots, routes each batch item through validity checks,
    CIK resolution, and candidate handling.

Conventions
-----------
- HTML/root elements are replaced with lightweight dummy objects that expose
  a compatible `.xpath(...)` API.
- Logging is captured via a minimal logger stand-in that mimics the subset
  of the `InfraLogger` interface used by the module.
- Tests focus purely on control flow, argument wiring, and normalization logic;
  XPath expressions and lxml internals are treated as implementation details.

Downstream usage
----------------
Run with `pytest` as part of the CI suite. These tests serve as executable
documentation for how Wayback snapshot parsing interacts with batch tickers,
validity windows, and the WayBackCandidate lifetime index.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, cast

import pandas as pd
import pytest

from infra.logging.infra_logger import InfraLogger
from infra.seeds.seed_evidence.seed_evidence_types import ValidityWindow
from infra.seeds.seed_evidence.wayback import wayback_snapshot_scrape as wbs
from infra.seeds.seed_evidence.wayback.wayback_config import WayBackCandidate


class _DummyLogger:
    """
    Purpose
    -------
    Minimal stand-in for `InfraLogger` for Wayback snapshot parsing tests.

    Key behaviors
    -------------
    - Collects all `.info(...)`, `.warning(...)`, and `.debug(...)` calls.
    - Stores both the message identifier and structured context for assertions.

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
    - This class implements only the subset of the real logger interface
      used in the snapshot scraping module.
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


def test_extract_table_prefers_primary_xpath_and_falls_back() -> None:
    """
    Verify that `extract_table` prefers the primary XPath and falls back when needed.

    Parameters
    ----------
    html_root : Any
        lxml HTML root element used by the test double to simulate different
        XPath matches.

    Returns
    -------
    None
        The test passes if `extract_table` returns:
            - the first table from `XPATH_WAYBACK_TABLE` when available, and
            - a table from `XPATH_WAYBACK_TABLE_FALLBACK` only when the primary
              XPath yields no results.

    Raises
    ------
    AssertionError
        If the function does not respect the primary → fallback resolution order
        or returns an unexpected table object.

    Notes
    -----
    - This test uses simple stubs for `html_root.xpath(...)` to avoid depending
      on real HTML structure.
    """

    class DummyRoot:
        def __init__(self, primary: List[Any], fallback: List[Any]) -> None:
            self.primary = primary
            self.fallback = fallback

        def xpath(self, expr: str) -> List[Any]:
            if expr == wbs.XPATH_WAYBACK_TABLE:
                return self.primary
            if expr == wbs.XPATH_WAYBACK_TABLE_FALLBACK:
                return self.fallback
            return []

    primary_table = object()
    fallback_table = object()

    root_primary = DummyRoot(primary=[primary_table], fallback=[fallback_table])
    table_primary = wbs.extract_table(root_primary)
    assert table_primary is primary_table

    root_fallback = DummyRoot(primary=[], fallback=[fallback_table])
    table_fallback = wbs.extract_table(root_fallback)
    assert table_fallback is fallback_table

    root_none = DummyRoot(primary=[], fallback=[])
    table_none = wbs.extract_table(root_none)
    assert table_none is None


def test_find_column_mappings_resolves_cik_and_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Ensure `find_column_mappings` locates CIK and ticker header indices correctly.

    Parameters
    ----------
    table : Any
        Stubbed table element providing HTML-like header rows with <th> elements.

    Returns
    -------
    None
        The test passes if the function returns the expected (cik_col, ticker_col)
        1-based indices when both headers are present, and `None` when either
        header is missing.

    Raises
    ------
    AssertionError
        If the returned indices do not match the header positions or if a case
        that should yield `None` instead returns a mapping.

    Notes
    -----
    - Header normalization (strip + upper) and scanning through
      `HEADER_TICKER_OPTIONS` are both exercised in this test.
    """

    class DummyCell:
        def __init__(self, text: str) -> None:
            self._text = text

        def xpath(self, expr: str) -> List[str]:
            if expr == ".//text()":
                return [self._text]
            return []

    class DummyHeaderRow:
        def __init__(self, texts: List[str]) -> None:
            self._cells = [DummyCell(t) for t in texts]

        def xpath(self, expr: str) -> List[DummyCell]:
            if expr == "./th":
                return self._cells
            return []

    class DummyTable:
        def __init__(self, texts: List[str]) -> None:
            self._header_row = DummyHeaderRow(texts)

        def xpath(self, expr: str) -> List[DummyHeaderRow]:
            if expr == ".//tr[th]":
                return [self._header_row]
            return []

    # Ensure predictable ticker header options.
    monkeypatch.setattr(
        wbs,
        "HEADER_TICKER_OPTIONS",
        ["TICKER SYMBOL", "SYMBOL", "TICKER"],
    )

    table = DummyTable(["CIK", "Ticker Symbol", "Other"])
    mapping = wbs.find_column_mappings(table)
    assert mapping == (1, 2)

    # Missing CIK header → None.
    table_no_cik = DummyTable(["Ticker Symbol", "Other"])
    assert wbs.find_column_mappings(table_no_cik) is None

    # Missing known ticker header → None.
    table_no_ticker = DummyTable(["CIK", "NAME"])
    assert wbs.find_column_mappings(table_no_ticker) is None


def test_extract_rows_returns_exact_ticker_match_not_prefix() -> None:
    """
    Validate that `extract_rows` returns only rows whose ticker matches exactly.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `table.xpath` so specific pseudo-rows can be returned and
        inspected without real HTML.

    Returns
    -------
    None
        The test passes if:
            - rows whose ticker cell matches the requested ticker exactly
              (case-insensitive) are returned, and
            - rows where the ticker is only a prefix (e.g., 'GL' vs 'GLW')
              are rejected.

    Raises
    ------
    AssertionError
        If a prefix-only match is accepted or an exact match fails to be
        returned.

    Notes
    -----
    - This test ensures the final Python-side equality check is enforced on
      top of the XPath-based prefilter.
    """

    class DummyRow:
        def __init__(self, ticker_text: str) -> None:
            self.ticker_text = ticker_text

        def xpath(self, expr: str) -> List[str]:
            # Called with "td[index]//text()"
            if "//text()" in expr:
                return [self.ticker_text]
            return []

    class DummyTable:
        def __init__(self, rows: List[DummyRow]) -> None:
            self._rows = rows

        def xpath(self, expr: str) -> List[DummyRow]:
            # Ignore the full XPath; return all rows to exercise Python-side filtering.
            return self._rows

    row_gl = DummyRow("GL")
    row_glw = DummyRow("GLW")
    table = DummyTable([row_gl, row_glw])

    result = wbs.extract_rows(table, ticker_column_index=2, ticker="GL")
    assert result is row_gl

    result_none = wbs.extract_rows(table, ticker_column_index=2, ticker="XYZ")
    assert result_none is None


def test_find_candidate_cik_accepts_10_digit_and_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Check that `find_candidate_cik` extracts and normalizes a 10-digit CIK string.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to stub `extract_rows` so the test can control the row
        content returned for a given ticker.

    Returns
    -------
    None
        The test passes if:
            - a row containing a CIK cell with extra non-digit characters still
              yields a 10-digit string, and
            - non-matching or malformed rows cause the function to return None.

    Raises
    ------
    AssertionError
        If a valid 10-digit CIK is not returned for the happy path, or if an
        invalid CIK is incorrectly accepted.

    Notes
    -----
    - The test also asserts that a debug log is emitted when a valid CIK is
      found for a ticker.
    """

    class DummyRow:
        def __init__(self, cik_text: str) -> None:
            self._cik_text = cik_text

        def xpath(self, expr: str) -> List[str]:
            # Expect "td[<cik_col>]//text()" here.
            if "//text()" in expr:
                return [self._cik_text]
            return []

    logger = _DummyLogger()

    def fake_extract_rows(table: Any, ticker_col_index: int, ticker: str) -> DummyRow | None:
        return DummyRow("00-12345678")  # non-digit chars that should be stripped

    monkeypatch.setattr(wbs, "extract_rows", fake_extract_rows)

    cik = wbs.find_candidate_cik(
        ticker="ABC",
        table=object(),
        header_mapping=(1, 2),
        logger=cast(InfraLogger, logger),
    )

    assert cik == "0012345678"
    assert logger.debugs, "Expected a debug log for found CIK"
    msg, ctx = logger.debugs[0]
    assert msg == "found_cik_for_ticker_in_wayback_snapshot"
    assert ctx["ticker"] == "ABC"
    assert ctx["cik"] == "0012345678"


def test_find_candidate_cik_rejects_non_10_digit(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Ensure `find_candidate_cik` rejects CIK cells that do not yield 10 digits.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to stub `extract_rows` so the test can control the CIK
        cell content returned for a given ticker.

    Returns
    -------
    None
        The test passes if `find_candidate_cik` returns None when the CIK cell,
        after stripping non-digits, does not contain exactly 10 digits, and no
        debug log is emitted.

    Raises
    ------
    AssertionError
        If a non-10-digit CIK is incorrectly accepted, or if any debug message
        is logged for an invalid CIK.

    Notes
    -----
    - This test complements the happy-path test by covering the failure branch
      where the CIK text is structurally wrong (too short).
    """

    class DummyRow:
        def __init__(self, cik_text: str) -> None:
            self._cik_text = cik_text

        def xpath(self, expr: str) -> List[str]:
            if "//text()" in expr:
                return [self._cik_text]
            return []

    logger = _DummyLogger()

    def fake_extract_rows(table: Any, ticker_col_index: int, ticker: str) -> DummyRow | None:
        return DummyRow("12345")  # too short even after digit stripping

    monkeypatch.setattr(wbs, "extract_rows", fake_extract_rows)

    cik = wbs.find_candidate_cik(
        ticker="ABC",
        table=object(),
        header_mapping=(1, 2),
        logger=cast(InfraLogger, logger),
    )

    assert cik is None
    assert not logger.debugs, "No debug log expected when CIK is invalid"


def test_handle_candidate_creates_and_updates_lifetimes() -> None:
    """
    Exercise `handle_candidate` behavior for candidate creation and lifetime updates.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if:
            - on first observation of (ticker, validity_window, candidate_cik),
              a new WayBackCandidate is created, stored in `seen_candidates`,
              and appended to `candidate_list`,
            - later snapshots with the same candidate extend `last_seen` and
              `last_seen_url` when they are newer,
            - earlier snapshots update `first_seen` and `first_seen_url` when
              they predate the current first_seen, and
            - passing a falsy `candidate_cik` results in a no-op.

    Raises
    ------
    AssertionError
        If the candidate is not created, is duplicated, or has incorrect
        first_seen / last_seen timestamps or URLs after multiple updates.

    Notes
    -----
    - This test does not mock WayBackCandidate; it asserts directly on the
      dataclass fields stored in `seen_candidates` and `candidate_list`.
    """

    start = pd.Timestamp("2020-01-02", tz="UTC")
    end = pd.Timestamp("2020-02-01", tz="UTC")
    window: ValidityWindow = (start, end)
    ticker = "ABC"
    candidate_cik = "0000123456"

    seen_candidates: Dict[Tuple[str, ValidityWindow], Dict[str, WayBackCandidate]] = {
        (ticker, window): {}
    }
    candidate_list: List[WayBackCandidate] = []

    first_snapshot = pd.Timestamp("2020-01-10", tz="UTC")
    first_url = "https://web.archive.org/web/20200110000000id_/page"

    wbs.handle_candidate(
        candidate_cik=candidate_cik,
        ticker=ticker,
        validity_window=window,
        snapshot_date=first_snapshot,
        archive_url=first_url,
        seen_candidates=seen_candidates,
        candidate_list=candidate_list,
    )

    assert len(candidate_list) == 1
    created = candidate_list[0]
    assert created.ticker == ticker
    assert created.candidate_cik == candidate_cik
    assert created.first_seen == first_snapshot
    assert created.last_seen == first_snapshot
    assert created.first_seen_url == first_url
    assert created.last_seen_url == first_url

    later_snapshot = pd.Timestamp("2020-01-20", tz="UTC")
    later_url = "https://web.archive.org/web/20200120000000id_/page"

    wbs.handle_candidate(
        candidate_cik=candidate_cik,
        ticker=ticker,
        validity_window=window,
        snapshot_date=later_snapshot,
        archive_url=later_url,
        seen_candidates=seen_candidates,
        candidate_list=candidate_list,
    )

    assert created.last_seen == later_snapshot
    assert created.last_seen_url == later_url
    assert created.first_seen == first_snapshot

    earlier_snapshot = pd.Timestamp("2020-01-05", tz="UTC")
    earlier_url = "https://web.archive.org/web/20200105000000id_/page"

    wbs.handle_candidate(
        candidate_cik=candidate_cik,
        ticker=ticker,
        validity_window=window,
        snapshot_date=earlier_snapshot,
        archive_url=earlier_url,
        seen_candidates=seen_candidates,
        candidate_list=candidate_list,
    )

    assert created.first_seen == earlier_snapshot
    assert created.first_seen_url == earlier_url
    assert created.last_seen == later_snapshot

    # Falsy candidate_cik should be a no-op.
    before_len = len(candidate_list)
    wbs.handle_candidate(
        candidate_cik="",
        ticker=ticker,
        validity_window=window,
        snapshot_date=later_snapshot,
        archive_url=later_url,
        seen_candidates=seen_candidates,
        candidate_list=candidate_list,
    )
    assert len(candidate_list) == before_len


def test_scrape_snapshot_short_circuits_when_no_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that `scrape_snapshot` short-circuits cleanly when no table is found.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `extract_table` and `find_column_mappings` to simulate a
        snapshot where no components table is discoverable.

    Returns
    -------
    None
        The test passes if:
            - `extract_table` is called exactly once,
            - `find_column_mappings` is never called when the table is None, and
            - `candidate_list` remains empty after the function returns.

    Raises
    ------
    AssertionError
        If column mappings are requested despite a missing table, or if any
        candidates are appended for a snapshot without a valid table.

    Notes
    -----
    - This test covers the early-return branch in `scrape_snapshot`, ensuring
      downstream helpers are not invoked when the HTML does not contain the
      expected components table.
    """

    calls: Dict[str, int] = {"extract_table": 0, "find_column_mappings": 0}

    def fake_extract_table(root: Any) -> Any | None:
        calls["extract_table"] += 1
        return None

    def fake_find_column_mappings(table: Any) -> tuple[int, int] | None:
        calls["find_column_mappings"] += 1
        return (1, 2)

    monkeypatch.setattr(wbs, "extract_table", fake_extract_table)
    monkeypatch.setattr(wbs, "find_column_mappings", fake_find_column_mappings)

    logger = _DummyLogger()
    batch: List[Tuple[str, ValidityWindow]] = []
    seen_candidates: Dict[Any, Any] = {}
    candidate_list: List[WayBackCandidate] = []

    wbs.scrape_snapshot(
        batch=batch,
        html_root=object(),
        seen_candidates=seen_candidates,
        candidate_list=candidate_list,
        archive_url="https://web.archive.org/web/...",
        snapshot_date=pd.Timestamp("2020-01-02", tz="UTC"),
        logger=cast(InfraLogger, logger),
    )

    assert calls["extract_table"] == 1
    assert calls["find_column_mappings"] == 0
    assert not candidate_list


def test_scrape_snapshot_routes_through_validity_and_candidate_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that `scrape_snapshot` wires validity checks and candidate resolution correctly.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `extract_table`, `find_column_mappings`,
        `within_validity_window`, `find_candidate_cik`, and `handle_candidate`
        so that control flow can be asserted without real HTML parsing.

    Returns
    -------
    None
        The test passes if:
            - `within_validity_window` is invoked once for each (ticker, window)
              pair in the batch,
            - `find_candidate_cik` is called only for tickers whose validity
              window contains `snapshot_date`, and
            - `handle_candidate` is called exactly for tickers that both pass
              the validity check and yield a non-None candidate CIK.

    Raises
    ------
    AssertionError
        If any helper is not called the expected number of times, is called
        with incorrect arguments, or if candidates are produced for tickers
        that are out of their validity window.

    Notes
    -----
    - This test focuses on orchestration rather than parsing, ensuring that
      `scrape_snapshot` delegates correctly to lower-level helpers.
    """

    logger = _DummyLogger()

    snapshot_date = pd.Timestamp("2020-01-10", tz="UTC")
    window1: ValidityWindow = (
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2020-01-31", tz="UTC"),
    )
    window2: ValidityWindow = (
        pd.Timestamp("2020-02-01", tz="UTC"),
        pd.Timestamp("2020-02-28", tz="UTC"),
    )
    batch: List[Tuple[str, ValidityWindow]] = [("AAA", window1), ("BBB", window2)]

    seen_candidates: Dict[Tuple[str, ValidityWindow], Dict[str, WayBackCandidate]] = {
        ("AAA", window1): {},
        ("BBB", window2): {},
    }
    candidate_list: List[Any] = []

    def fake_extract_table(root: Any) -> Any:
        return "TABLE"

    def fake_find_column_mappings(table: Any) -> tuple[int, int] | None:
        assert table == "TABLE"
        return (1, 2)

    validity_calls: List[Tuple[pd.Timestamp, ValidityWindow]] = []

    def fake_within_validity_window(date: pd.Timestamp, window: ValidityWindow) -> bool:
        validity_calls.append((date, window))
        return window is window1  # only AAA is valid at snapshot_date

    candidate_calls: List[Tuple[str, Any, tuple[int, int]]] = []

    def fake_find_candidate_cik(
        ticker: str, table: Any, header_mapping: tuple[int, int], logger_param: InfraLogger
    ) -> str | None:
        candidate_calls.append((ticker, table, header_mapping))
        if ticker == "AAA":
            return "0000123456"
        return None

    handle_calls: List[Dict[str, Any]] = []

    def fake_handle_candidate(
        candidate_cik: str,
        ticker: str,
        validity_window: ValidityWindow,
        snapshot_date_param: pd.Timestamp,
        archive_url: str,
        seen_candidates_param: Any,
        candidate_list_param: List[Any],
    ) -> None:
        handle_calls.append(
            {
                "candidate_cik": candidate_cik,
                "ticker": ticker,
                "validity_window": validity_window,
                "snapshot_date": snapshot_date_param,
                "archive_url": archive_url,
            }
        )
        candidate_list_param.append(candidate_cik)

    monkeypatch.setattr(wbs, "extract_table", fake_extract_table)
    monkeypatch.setattr(wbs, "find_column_mappings", fake_find_column_mappings)
    monkeypatch.setattr(wbs, "within_validity_window", fake_within_validity_window)
    monkeypatch.setattr(wbs, "find_candidate_cik", fake_find_candidate_cik)
    monkeypatch.setattr(wbs, "handle_candidate", fake_handle_candidate)

    wbs.scrape_snapshot(
        batch=batch,
        html_root=object(),
        seen_candidates=seen_candidates,
        candidate_list=candidate_list,
        archive_url="https://web.archive.org/web/20200110000000id_/page",
        snapshot_date=snapshot_date,
        logger=cast(InfraLogger, logger),
    )

    # within_validity_window called for both batch entries.
    assert {w for _d, w in validity_calls} == {window1, window2}

    # find_candidate_cik should only be called for windows where within_validity_window == True.
    called_tickers = {t for t, _table, _hdr in candidate_calls}
    assert called_tickers == {"AAA"}

    assert len(handle_calls) == 1
    hc = handle_calls[0]
    assert hc["ticker"] == "AAA"
    assert hc["candidate_cik"] == "0000123456"
    assert hc["snapshot_date"] == snapshot_date
    assert candidate_list == ["0000123456"]
