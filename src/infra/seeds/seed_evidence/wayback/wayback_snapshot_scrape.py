"""
Purpose
-------
Provide HTML-level parsing for Wayback snapshots of the S&P 500
components page: locate the components table, resolve ticker/CIK
columns, match tickers for a given batch, and update WayBackCandidate
state accordingly.

Key behaviors
-------------
- Extract the relevant components table from a Wikipedia Wayback snapshot.
- Determine the ticker and CIK column indices from header rows.
- Match each (ticker, validity_window) in the batch whose window covers
  the snapshot date, and attempt to resolve a 10-digit CIK for it.
- Maintain first_seen / last_seen timestamps and URLs for each
  (ticker, validity_window, candidate_cik) via the SeenCandidates index.

Conventions
-----------
- `snapshot_date` is a tz-aware UTC pandas.Timestamp.
- Validity windows use half-open [start, end) semantics.
- Tickers in the batch are assumed to be normalized upstream.

Downstream usage
----------------
This module is invoked by the Wayback orchestrator via `scrape_snapshot(...)`
for each archived snapshot URL and timestamp returned by the CDX layer.
"""

from typing import Any, List

import pandas as pd

from infra.logging.infra_logger import InfraLogger
from infra.seeds.seed_evidence.edgar_search.edgar_search_conditions import within_validity_window
from infra.seeds.seed_evidence.seed_evidence_types import ValidityWindow
from infra.seeds.seed_evidence.wayback.wayback_config import (
    HEADER_TICKER_OPTIONS,
    XPATH_WAYBACK_TABLE,
    XPATH_WAYBACK_TABLE_FALLBACK,
    WayBackCandidate,
)
from infra.seeds.seed_evidence.wayback.wayback_typing import Batch, SeenCandidates


def scrape_snapshot(
    batch: Batch,
    html_root: Any,
    seen_candidates: SeenCandidates,
    candidate_list: List[WayBackCandidate],
    archive_url: str,
    snapshot_date: pd.Timestamp,
    logger: InfraLogger,
) -> None:
    """
    Extract candidate CIKs for all relevant batch tickers from one snapshot.

    Parameters
    ----------
    batch : Batch
        List of (ticker, validity_window) pairs to test against this snapshot.
    html_root : Any
        lxml HTML root element parsed from the archived page content.
    seen_candidates : SeenCandidates
        Nested dict for deduplication and candidate lifetime tracking.
    candidate_list : List[WayBackCandidate]
        Accumulator list for newly discovered WayBackCandidate instances.
    archive_url : str
        Wayback URL of the snapshot currently being processed.
    snapshot_date : pd.Timestamp
        tz-aware UTC timestamp derived from the capture time encoded in the URL.
    logger : InfraLogger
        Logger for warnings and debug events.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    - The components table and column mappings are resolved once per snapshot;
      if they cannot be found, the function returns early without modifying
      `candidate_list`.
    - For each (ticker, validity_window) pair, the snapshot is only considered
      if `within_validity_window(snapshot_date, validity_window)` is true.
    """

    table: Any | None = extract_table(html_root)
    if table is None:
        return None
    header_mapping: tuple[int, int] | None = find_column_mappings(table)
    if header_mapping is None:
        return None
    for item in batch:
        ticker = item[0]
        validity_window = item[1]
        if within_validity_window(snapshot_date, validity_window):
            candidate_cik = find_candidate_cik(ticker, table, header_mapping, logger)
            if candidate_cik:
                handle_candidate(
                    candidate_cik,
                    ticker,
                    validity_window,
                    snapshot_date,
                    archive_url,
                    seen_candidates,
                    candidate_list,
                )
    return None


def extract_table(html_root: Any) -> Any | None:
    """
    Locate the S&P 500 components table in the archived HTML snapshot.

    Parameters
    ----------
    html_root : Any
        lxml HTML root element parsed from the Wayback snapshot.

    Returns
    -------
    Any or None
        The first matching table element if found; otherwise None.

    Raises
    ------
    None

    Notes
    -----
    - First attempts a precise XPath that requires a 'CIK' header and a
      ticker-like header (e.g., 'SYMBOL' or 'TICKER SYMBOL').
    - Falls back to a looser XPath anchored on the
      'S&P 500 COMPONENT STOCKS' section heading if the primary XPath fails.
    """

    tables: List[Any] = html_root.xpath(XPATH_WAYBACK_TABLE)
    table: Any = tables[0] if tables else None
    if table is None:
        tables = html_root.xpath(XPATH_WAYBACK_TABLE_FALLBACK)
        table = tables[0] if tables else None
    return table


def find_column_mappings(table: Any) -> tuple[int, int] | None:
    """
    Determine the column indices of the CIK and ticker headers in the table.

    Parameters
    ----------
    table : Any
        lxml element representing the components table.

    Returns
    -------
    tuple[int, int] or None
        (cik_col_index, ticker_col_index) as 1-based indices suitable for
        use in XPath predicates, or None if either header cannot be found.

    Raises
    ------
    None

    Notes
    -----
    - Header text is uppercased and stripped before comparison.
    - The ticker column is resolved by scanning HEADER_TICKER_OPTIONS
      in order (e.g., 'TICKER SYMBOL', 'SYMBOL', 'TICKER').
    """

    header_rows: List[Any] = table.xpath(".//tr[th]")
    if not header_rows:
        return None
    table_headers: List[Any] = header_rows[0].xpath("./th")
    table_header_text: List[str] = [
        "".join(th.xpath(".//text()")).strip().upper() for th in table_headers
    ]
    idx_map: dict[str, int] = {name: i for i, name in enumerate(table_header_text)}
    cik_col: int | None = idx_map.get("CIK")
    ticker_col: int | None = None
    for option in HEADER_TICKER_OPTIONS:
        if option in idx_map:
            ticker_col = idx_map.get(option)
            break
    if cik_col is None or ticker_col is None:
        return None
    return cik_col + 1, ticker_col + 1  # Convert to 1-based index for XPath


def find_candidate_cik(
    ticker: str, table: Any, header_mapping: tuple[int, int], logger: InfraLogger
) -> str | None:
    """
    Resolve a candidate CIK for a given ticker from the components table.

    Parameters
    ----------
    ticker : str
        Normalized ticker symbol for which to look up a CIK.
    table : Any
        lxml element representing the components table.
    header_mapping : tuple[int, int]
        (cik_col_index, ticker_col_index) as 1-based indices in the table.
    logger : InfraLogger
        Logger for DEBUG events when a CIK is successfully resolved.

    Returns
    -------
    str or None
        A 10-digit CIK string if a matching row is found and the CIK cell
        contains a 10-digit value; otherwise None.

    Raises
    ------
    None

    Notes
    -----
    - Row selection is delegated to `extract_rows(...)`.
    - All non-digit characters are stripped from the CIK cell text before
      validating the length.
    """

    row: Any | None = extract_rows(table, header_mapping[1], ticker)
    if row is None:
        return None
    cik_text: str = "".join(row.xpath(f"td[{header_mapping[0]}]//text()")).strip()
    cik_digits: str = "".join(ch for ch in cik_text if ch.isdigit())
    if cik_digits and len(cik_digits) == 10:
        logger.debug(
            "found_cik_for_ticker_in_wayback_snapshot",
            context={"ticker": ticker, "cik": cik_digits},
        )
        return cik_digits
    return None


def extract_rows(table: Any, ticker_column_index: int, ticker: str) -> Any | None:
    """
    Find the table row corresponding to a given ticker symbol.

    Parameters
    ----------
    table : Any
        lxml element representing the components table.
    ticker_column_index : int
        1-based index of the ticker column within the table.
    ticker : str
        Normalized ticker symbol to match against the row's ticker cell.

    Returns
    -------
    Any or None
        The first row element whose ticker cell corresponds to `ticker`
        under the module's normalization policy, or None if no such row
        can be found.

    Raises
    ------
    None

    Notes
    -----
    - An initial XPath filter narrows rows based on case-insensitive
      starts-with checks on the ticker cell text or its anchor text,
      and on the anchor href.
    - A final Python-side equality check enforces an exact
      case-insensitive match between the cell text and the batch ticker
      to avoid prefix collisions (e.g., 'GL' vs 'GLW').
    """

    rows: List[Any] = table.xpath(
        f".//tr[ td and ( \
            starts-with(translate(normalize-space(string(td[{ticker_column_index}])), \
                'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ'), '{ticker}') \
            or starts-with(translate(normalize-space(string(td[{ticker_column_index}]//a)), \
                'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ'), '{ticker}') \
            or contains(translate(string(td[{ticker_column_index}]//a/@href), \
                'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ'), '/{ticker}') \
            )]"
    )
    if not rows:
        return None
    for row in rows:
        ticker_text: str = "".join(row.xpath(f"td[{ticker_column_index}]//text()")).strip()
        if ticker_text.upper() == ticker.upper():
            return row
    return None


def handle_candidate(
    candidate_cik: str,
    ticker: str,
    validity_window: ValidityWindow,
    snapshot_date: pd.Timestamp,
    archive_url: str,
    seen_candidates: SeenCandidates,
    candidate_list: List[WayBackCandidate],
) -> None:
    """
    Update or insert a WayBackCandidate based on a newly observed snapshot.

    Parameters
    ----------
    candidate_cik : str
        Candidate 10-digit CIK string resolved for the ticker at this snapshot,
        or a falsy value if no CIK could be found.
    ticker : str
        Ticker symbol associated with this candidate.
    validity_window : ValidityWindow
        Half-open [start, end) window in which the ticker is considered valid.
    snapshot_date : pd.Timestamp
        tz-aware UTC timestamp for the Wayback snapshot.
    archive_url : str
        Wayback URL where the candidate was observed.
    seen_candidates : SeenCandidates
        Nested dict keyed by (ticker, validity_window)[candidate_cik] used
        for deduplication and lifetime tracking.
    candidate_list : List[WayBackCandidate]
        Accumulator list of created WayBackCandidate instances.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If (ticker, validity_window) is not present in `seen_candidates`,
        which indicates an internal contract violation in batch setup.

    Notes
    -----
    - On first observation of (ticker, validity_window, candidate_cik),
      a new WayBackCandidate is created, added to `seen_candidates`, and
      appended to `candidate_list`.
    - On subsequent observations, only the first_seen / last_seen timestamps
      and URLs are updated to reflect the extended lifetime of the candidate.
    """

    if candidate_cik:
        if candidate_cik not in seen_candidates[(ticker, validity_window)].keys():
            candidate = WayBackCandidate(
                ticker=ticker,
                validity_window=validity_window,
                candidate_cik=candidate_cik,
                first_seen=snapshot_date,
                last_seen=snapshot_date,
                first_seen_url=archive_url,
                last_seen_url=archive_url,
            )
            seen_candidates[(ticker, validity_window)][candidate_cik] = candidate
            candidate_list.append(candidate)
        else:
            existing_candidate = seen_candidates[(ticker, validity_window)][candidate_cik]
            timestamp_snapshot = snapshot_date
            if timestamp_snapshot > existing_candidate.last_seen:
                existing_candidate.last_seen = timestamp_snapshot
                existing_candidate.last_seen_url = archive_url
            if timestamp_snapshot < existing_candidate.first_seen:
                existing_candidate.first_seen = timestamp_snapshot
                existing_candidate.first_seen_url = archive_url
    return None
