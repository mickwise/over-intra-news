"""
Purpose
-------
Utility helpers for the EDGAR Atom harvesting pipeline: URL construction, run-time
context packaging, XML element access with/without namespaces, window checks, and
paging control. Centralizes small, reusable behaviors used by the search module.

Key behaviors
-------------
- Builds company Atom feed URLs with a consistent query parameter set.
- Packages per-entry context into a `RunData` TypedDict for downstream functions.
- Reads `<updated>` timestamps and decides whether to continue, skip, or stop paging.
- Finds XML nodes regardless of namespace presence (XPath fallback).
- Detects final/empty pages to terminate paging loops cleanly.
- Persists minimal Atom entry metadata into the `RawRecord`.

Conventions
-----------
- Timezone: all timestamps are in UTC.
- Window semantics: left-closed, right-open (`start <= ts < end`).
- Pagination: `REQUEST_COUNT` controls page size; pages with `0` or `< REQUEST_COUNT`
  entries signal termination.
- Logging: events include `stage="edgar_search"` in `context`.

Downstream usage
----------------
Imported by the EDGAR search module to:
- Build feed URLs (`build_company_feed_url`).
- Evaluate entry timestamps (`evaluate_updated_timestamp`).
- Package entry-scoped context (`create_run_data`).
- Check window inclusion (`within_validity_window`).
- Navigate Atom XML (`find_element`, `find_all_elements`).
- Decide when to stop paging (`evaluate_page_break_conditions`).
- Attach Atom metadata to the `RawRecord` (`set_atom_entry`).
"""

from typing import Any, List, TypeAlias, TypedDict
from urllib.parse import urlencode

import pandas as pd

from infra.logging.infra_logger import InfraLogger
from infra.seeds.seed_firm_characteristics.records.raw_record import RawRecord
from infra.seeds.seed_firm_characteristics.records.table_records import (
    MappingEvidence,
    ValidityWindow,
)

REQUEST_COUNT: int = 100
REQUEST_DELAY_SECONDS: float = 0.1  # To respect SEC rate limits
ELIGIBLE_FORM_TYPES: frozenset[str] = frozenset(
    {
        # Annual reports
        "10-K",
        "10-KT",
        "10-K/A",
        # Quarterly reports
        "10-Q",
        "10-QT",
        "10-Q/A",
        # Current reports
        "8-K",
        "8-K/A",
        # Registration of securities
        "8-A12B",
        "8-A12B/A",
        "8-A12G",
        "8-A12G/A",
        # Registration statements
        "S-1",
        "S-3",
        "S-4",
        "S-1/A",
        "S-3/A",
        "S-4/A",
        # Foreign filings
        "20-F",
        "40-F",
        "20-F/A",
        "40-F/A",
        # Foreign current reports
        "6-K",
        "6-K/A",
    }
)

# ticker -> window_str -> evidence_id -> MappingEvidence
CollectedEvidence: TypeAlias = dict[str, dict[str, dict[str, MappingEvidence]]]
NameSpaceBindings: TypeAlias = dict[str, Any]


class RunData(TypedDict):
    """
    Purpose
    -------
    Typed container for per-entry state shared across extraction steps.

    Key behaviors
    -------------
    - Provides a single, typed mapping that carries the ticker, validity window,
      namespace bindings, logger, and the Atom `entry` element.

    Fields
    ------
    ticker : str
        Canonicalized ticker for the current scan.
    validity_window : ValidityWindow
        Tuple of (start_utc, end_utc) with half-open semantics.
    name_space_bindings : dict[str, Any] | None
        Namespace map (e.g., {'ns': 'http://www.w3.org/2005/Atom'}) or None when absent.
    logger : InfraLogger
        Structured logger used by helpers for observability.
    entry : Any
        The Atom `<entry>` element (lxml node) under inspection.

    Notes
    -----
    - This is a `TypedDict`, not a class; access via `run_data["field"]`.
    """

    ticker: str
    validity_window: ValidityWindow
    name_space_bindings: NameSpaceBindings | None
    logger: InfraLogger
    entry: Any


def build_company_feed_url(
    ticker: str, inclusive_window_end: str, start_offset: int, logger: InfraLogger
) -> str:
    """
    Construct the SEC EDGAR company Atom feed URL for a page and log the batch start.

    Parameters
    ----------
    ticker : str
        CIK or ticker value to query (`CIK` parameter).
    inclusive_window_end : str
        Upper bound (inclusive) for EDGAR's `dateb` query parameter, formatted YYYYMMDD.
    start_offset : int
        Zero-based pagination offset for the Atom feed.
    logger : InfraLogger
        Logger used to emit a `batch_fetch_start` DEBUG event.

    Returns
    -------
    str
        Fully composed URL for the requested page.

    Notes
    -----
    - Emits DEBUG `batch_fetch_start` with `ticker` and `offset` in context.
    """

    params = {
        "action": "getcompany",
        "CIK": ticker,
        "owner": "exclude",
        "dateb": inclusive_window_end,
        "count": REQUEST_COUNT,
        "start": start_offset,
        "output": "atom",
    }
    url = "https://www.sec.gov/cgi-bin/browse-edgar?" + urlencode(params)
    logger.debug(
        "batch_fetch_start",
        context={"stage": "edgar_search", "ticker": ticker, "offset": start_offset},
    )
    return url


def evaluate_updated_timestamp(
    run_data: RunData,
) -> tuple[pd.Timestamp | None, bool]:
    """
    Read an Atom entry's `<updated>` timestamp and decide whether to process, skip, or stop.

    Parameters
    ----------
    run_data : RunData
        Entry context including the lxml node, namespace bindings, logger, and window.

    Returns
    -------
    tuple[pandas.Timestamp | None, bool]
        (ts, missing_updated_flag)
        - If `<updated>` exists and ts >= window.start: (ts, False) → proceed.
        - If `<updated>` exists and ts < window.start: (None, False) → signal caller to stop paging.
        - If `<updated>` is missing: (None, True) → skip this entry and continue paging.

    Notes
    -----
    - `<updated>` is parsed with UTC; comparison uses half-open window semantics.
    - On "older than window", emits INFO `loop_break` with `updated_ts` and `window_start`.
    - Caller is responsible for acting on the (None, False) stop signal.
    """

    entry_updated: Any = find_element(run_data["entry"], "updated", run_data["name_space_bindings"])
    if entry_updated is not None:
        updated_ts: pd.Timestamp = pd.to_datetime(entry_updated.text, utc=True)
        if updated_ts < run_data["validity_window"][0]:
            run_data["logger"].info(
                "loop_break",
                msg="older than window",
                context={
                    "ticker": run_data["ticker"],
                    "updated_ts": updated_ts.isoformat(),
                    "window_start": run_data["validity_window"][0].isoformat(),
                },
            )
            return None, False
        else:
            return updated_ts, False
    return None, True


def create_run_data(
    ticker: str,
    validity_window: ValidityWindow,
    name_space_bindings: NameSpaceBindings | None,
    logger: InfraLogger,
    entry: Any,
) -> RunData:
    """
    Assemble a `RunData` mapping for a single Atom entry.

    Parameters
    ----------
    ticker : str
        Canonicalized ticker for context.
    validity_window : ValidityWindow
        Left-closed, right-open (start_utc, end_utc) tuple.
    name_space_bindings : dict[str, Any] | None
        Namespace map for XPath or None if the feed is non-namespaced.
    logger : InfraLogger
        Logger to pass through to downstream steps.
    entry : Any
        The Atom `<entry>` lxml element.

    Returns
    -------
    RunData
        Typed mapping containing all fields required by downstream helpers.

    Notes
    -----
    - Intended to reduce positional parameter churn and keep signatures stable.
    """

    return {
        "ticker": ticker,
        "validity_window": validity_window,
        "name_space_bindings": name_space_bindings,
        "logger": logger,
        "entry": entry,
    }


def within_validity_window(date: pd.Timestamp, validity_window: ValidityWindow) -> bool:
    """
    Check whether a timestamp falls within the half-open validity window.

    Parameters
    ----------
    date : pandas.Timestamp
        UTC timestamp to test.
    validity_window : ValidityWindow
        (start_utc, end_utc) tuple.

    Returns
    -------
    bool
        True if `start_utc <= date < end_utc`, else False.
    """

    return validity_window[0] <= date < validity_window[1]


def find_element(parent: Any, tag: str, name_space_bindings: dict[str, Any] | None) -> Any | None:
    """
    Find the first child node with the given local name, with namespace fallback.

    Parameters
    ----------
    parent : Any
        lxml element to search under.
    tag : str
        Local tag name to match (e.g., 'updated', 'title').
    name_space_bindings : dict[str, Any] | None
        Namespace map (e.g., {'ns': 'http://www.w3.org/2005/Atom'}) or None.

    Returns
    -------
    Any | None
        The first matching lxml node or None if not found.

    Notes
    -----
    - If a default namespace is present, uses `.find('ns:tag', namespaces=...)`.
    - Otherwise falls back to XPath `//*[local-name()='tag']` under the parent.
    """

    if name_space_bindings:
        return parent.find(f"ns:{tag}", namespaces=name_space_bindings)
    else:
        local_name_node: Any = parent.xpath(f".//*[local-name()='{tag}']")
        return local_name_node[0] if local_name_node else None


def find_all_elements(
    parent: Any, tag: str, name_space_bindings: dict[str, Any] | None
) -> List[Any]:
    """
    Find all child nodes with the given local name, with namespace fallback.

    Parameters
    ----------
    parent : Any
        lxml element to search under.
    tag : str
        Local tag name to match.
    name_space_bindings : dict[str, Any] | None
        Namespace map or None when absent.

    Returns
    -------
    list[Any]
        List of matching lxml nodes (possibly empty).

    Notes
    -----
    - Mirrors `find_element` logic but returns all matches instead of the first.
    """

    if name_space_bindings:
        return parent.findall(f"ns:{tag}", namespaces=name_space_bindings)
    else:
        return parent.xpath(f".//*[local-name()='{tag}']")


def evaluate_page_break_conditions(
    entries: List[Any], logger: InfraLogger, ticker: str, start_offset: int
) -> bool:
    """
    Determine whether the current Atom page is terminal (empty or final short page).

    Parameters
    ----------
    entries : list[Any]
        List of Atom `<entry>` nodes on the current page.
    logger : InfraLogger
        Logger used to emit `loop_break` reasons.
    ticker : str
        Ticker for logging context.
    start_offset : int
        Current page offset for logging context.

    Returns
    -------
    bool
        True if the caller should stop paging; False to continue.

    Notes
    -----
    - Emits INFO `loop_break` with reason 'no entries' when `len(entries) == 0`.
    - Emits INFO `loop_break` with reason 'short page' when `len(entries) < REQUEST_COUNT`.
    """

    n: int = len(entries)
    if n == 0:
        logger.info(
            "loop_break",
            msg="no entries",
            context={"stage": "edgar_search", "ticker": ticker, "offset": start_offset},
        )
        return True
    if n < REQUEST_COUNT:
        logger.info(
            "loop_break",
            msg="short page",
            context={"stage": "edgar_search", "ticker": ticker, "count": n},
        )
        return True
    return False


def set_atom_entry(
    entry: Any,
    link_href: str,
    updated_ts: pd.Timestamp,
    raw_record: RawRecord,
    name_space_bindings: NameSpaceBindings | None,
) -> None:
    """
    Persist minimal Atom entry metadata into the `RawRecord`.

    Parameters
    ----------
    entry : Any
        Atom `<entry>` lxml element.
    link_href : str
        URL from the entry's 'alternate' link (filing index page).
    updated_ts : pandas.Timestamp
        Entry's `<updated>` timestamp (UTC).
    raw_record : RawRecord
        Mutable record to enrich; `raw_record['atom_entry']` will be set/overwritten.
    name_space_bindings : dict[str, Any] | None
        Namespace map or None for the XPath fallback.

    Returns
    -------
    None

    Notes
    -----
    - Captures: 'alternate_link', 'updated' (ISO-8601), 'entry_id', and 'title'.
    """

    entry_id = find_element(entry, "id", name_space_bindings)
    entry_title = find_element(entry, "title", name_space_bindings)
    raw_record["atom_entry"] = {
        "alternate_link": link_href,
        "updated": updated_ts.isoformat(),
        "entry_id": entry_id.text if entry_id is not None else None,
        "title": entry_title.text if entry_title is not None else None,
    }


def extract_evidence_by_id(
    ticker: str, window_key: str, evidence_id: str, collected_evidence: CollectedEvidence
) -> MappingEvidence | None:
    """
    Look up a `MappingEvidence` by (ticker, window_key, evidence_id).

    Purpose
    -------
    - Convenience accessor for downstream loaders and split logic to retrieve
      a specific evidence row from the nested `CollectedEvidence` structure.

    Parameters
    ----------
    ticker : str
        Uppercased ticker key.
    window_key : str
        Canonical window string, e.g., 'YYYY-MM-DD to YYYY-MM-DD'.
    evidence_id : str
        Deterministic UUIDv5 of the evidence row.
    collected_evidence : CollectedEvidence
        Nested mapping: ticker → window_key → evidence_id → MappingEvidence.

    Returns
    -------
    MappingEvidence | None
        The evidence if present; otherwise None.

    Notes
    -----
    - Safe to call even when intermediate keys are missing.
    """

    ticker_evidences: dict[str, dict[str, MappingEvidence]] | None = collected_evidence.get(ticker)
    if ticker_evidences is None:
        return None
    window_evidences: dict[str, MappingEvidence] | None = ticker_evidences.get(window_key)
    if window_evidences is None:
        return None
    evidence: MappingEvidence | None = window_evidences.get(evidence_id)
    return evidence
