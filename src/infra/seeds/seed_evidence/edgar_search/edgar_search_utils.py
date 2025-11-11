"""
Purpose
-------
Provide utility helpers for the EDGAR Atom harvesting pipeline: company-feed URL
construction, run-time context packaging, XML navigation with or without namespaces,
and minimal Atom entry metadata persistence. This module centralizes the small,
reusable behaviors used by the EDGAR search core and its conditions module.

Key behaviors
-------------
- Build company Atom feed URLs with a consistent query parameter set.
- Package per-page and per-entry context into a `RunData` TypedDict.
- Locate XML nodes with namespace-aware helpers that gracefully fall back to
  local-name XPath queries.
- Parse EDGAR Atom responses into `<entry>` elements and namespace bindings.
- Persist minimal Atom entry metadata into a `RawRecord` for provenance.

Conventions
-----------
- All timestamps are treated as UTC.
- Window semantics are half-open: `start <= ts < end`.
- Pagination is controlled via `PAGE_SIZE` and the `dateb` parameter; the `start`
  parameter is always 0.
- Logging attaches a `"stage"` key to `context` to identify the emitting helper.

Downstream usage
----------------
- Imported by the EDGAR search core to:
  - build feed URLs (`build_company_feed_url`, `build_url`),
  - assemble `RunData` instances (`create_run_data`),
  - traverse Atom XML (`find_element`, `find_all_elements`),
  - and attach Atom-level metadata to provenance records (`set_atom_entry`).
- Also imported by `edgar_search_conditions` for namespace-aware XML lookups.
"""

from typing import Any, List, TypedDict
from urllib.parse import urlencode

import pandas as pd
import requests
from lxml import etree

from infra.logging.infra_logger import InfraLogger
from infra.seeds.seed_evidence.edgar_search.edgar_config import PAGE_SIZE
from infra.seeds.seed_evidence.records.raw_record import RawRecord
from infra.seeds.seed_evidence.seed_evidence_types import Entries, NameSpaceBindings, ValidityWindow


class RunData(TypedDict):
    """
    Purpose
    -------
    Container for per-page and per-entry context passed among EDGAR search helpers.

    Key behaviors
    -------------
    - Carries the canonical ticker and the candidate identifier used in EDGAR queries.
    - Holds the validity window, XML namespace bindings, and the current `<entry>`.
    - Tracks the oldest filing date seen so far on the current feed walk.

    Parameters
    ----------
    ticker : str
        Canonical project ticker associated with this collection (used for attribution).
    validity_window : ValidityWindow
        Half-open `(start_utc, end_utc)` window used for gating.
    candidate : str
        Identifier used as the EDGAR `CIK` parameter (ticker or 10-digit CIK).
    name_space_bindings : NameSpaceBindings | None
        Namespace map for namespaced XPath queries, or `None` when no namespace is used.
    logger : InfraLogger
        Structured logger propagated through the pipeline.
    entry : Any
        The current Atom `<entry>` element; updated as the paging loop advances.
    oldest_filing_date : pandas.Timestamp
        Oldest filing or `<updated>` timestamp encountered so far, used by
        page-break logic.

    Attributes
    ----------
    ticker : str
        Canonical ticker for this run.
    validity_window : ValidityWindow
        Window against which timestamps are compared.
    candidate : str
        Ticker or CIK actually used in the EDGAR request.
    name_space_bindings : NameSpaceBindings | None
        Namespace bindings used by `find_element` and `find_all_elements`.
    logger : InfraLogger
        Logger for all helper functions participating in this run.
    entry : Any
        The current Atom `<entry>` being processed.
    oldest_filing_date : pandas.Timestamp
        Updated as entries and filings are processed; used to move the `dateb` cursor.

    Notes
    -----
    - This is a `TypedDict`; it carries structured data only and defines no methods.
    """

    ticker: str
    validity_window: ValidityWindow
    candidate: str
    name_space_bindings: NameSpaceBindings | None
    logger: InfraLogger
    entry: Any
    oldest_filing_date: pd.Timestamp


def build_url(
    ticker: str,
    validity_window: ValidityWindow,
    logger: InfraLogger,
    end_date: pd.Timestamp | None = None,
) -> str:
    """
    Build the EDGAR company Atom feed URL for a given window and cutoff date.

    Parameters
    ----------
    ticker : str
        Ticker or 10-digit CIK for EDGAR's `CIK` parameter.
    validity_window : ValidityWindow
        Half-open `(start_utc, end_utc)` window; when `end_date` is None, the
        effective `dateb` uses `(end_utc - 1 day)`.
    logger : InfraLogger
        Logger for debug breadcrumbs when a non-window end date is supplied.
    end_date : pandas.Timestamp | None, optional
        Effective inclusive `dateb` cutoff. When `None`, uses `(window_end - 1 day)`.

    Returns
    -------
    str
        Fully composed EDGAR Atom URL for the requested page.

    Raises
    ------
    None

    Notes
    -----
    - EDGAR's `dateb` parameter is inclusive.
    - Pagination is performed by lowering `dateb` across iterations; `start` is
      always 0.
    - When `end_date` is provided explicitly, a debug event is logged with
      `stage="url_construction"`.
    """

    if end_date is not None:
        logger.debug(
            "using_non_window_end_end_date_for_url_construction",
            context={
                "stage": "url_construction",
                "ticker": ticker,
                "end_date": end_date.strftime("%Y%m%d") if end_date else None,
            },
        )
        return build_company_feed_url(
            ticker,
            end_date.strftime("%Y%m%d"),
            logger,
        )
    else:
        inclusive_window_end: str = (validity_window[1] - pd.Timedelta(days=1)).strftime("%Y%m%d")
        return build_company_feed_url(ticker, inclusive_window_end, logger)


def build_company_feed_url(candidate: str, inclusive_window_end: str, logger: InfraLogger) -> str:
    """
    Construct the EDGAR company Atom feed URL for a single page and log the batch start.

    Parameters
    ----------
    candidate : str
        10-digit CIK to pass as EDGAR's `CIK` parameter.
    inclusive_window_end : str
        Inclusive upper bound for EDGAR's `dateb` in `YYYYMMDD` format.
    logger : InfraLogger
        Logger used to emit a `batch_fetch_start` DEBUG event.

    Returns
    -------
    str
        Fully composed EDGAR URL for the requested page.

    Raises
    ------
    None

    Notes
    -----
    - Pins `owner=exclude` and `output=atom` for consistency.
    - Uses `PAGE_SIZE` as the `count` parameter.
    - Logs a `batch_fetch_start` debug event with:
    - `stage="build_company_feed_url"`,
    - the `ticker`,
    - and `offset=0`.
    """

    params = {
        "action": "getcompany",
        "CIK": candidate,
        "owner": "exclude",
        "dateb": inclusive_window_end,
        "count": PAGE_SIZE,
        "start": 0,
        "output": "atom",
    }
    url = "https://www.sec.gov/cgi-bin/browse-edgar?" + urlencode(params)
    logger.debug(
        "batch_fetch_start",
        context={"stage": "build_company_feed_url", "candidate": candidate, "offset": 0},
    )
    return url


def create_run_data(
    ticker: str,
    validity_window: ValidityWindow,
    candidate: str,
    name_space_bindings: NameSpaceBindings | None,
    logger: InfraLogger,
    entry: Any,
    oldest_filing_date: pd.Timestamp | None,
) -> RunData:
    """
    Assemble a `RunData` mapping for a single Atom page or entry.

    Parameters
    ----------
    ticker : str
        Canonicalized ticker for context and evidence records.
    validity_window : ValidityWindow
        Left-closed, right-open `(start_utc, end_utc)` window.
    candidate : str
        Identifier actually used in the EDGAR query (ticker or CIK).
    name_space_bindings : NameSpaceBindings | None
        Namespace map for XPath, or `None` for local-name fallbacks.
    logger : InfraLogger
        Logger to pass through to downstream steps.
    entry : Any
        Atom `<entry>` element for the current loop step, or `None` before assignment.
    oldest_filing_date : pandas.Timestamp | None
        Oldest filing timestamp seen so far; when `None`, this is seeded with
        `validity_window[1]` (the window end).

    Returns
    -------
    RunData
        Typed mapping containing all context required by downstream helpers.

    Raises
    ------
    None

    Notes
    -----
    - If `oldest_filing_date` is `None`, the field is initialized to the window end
      so that the first page can still advance the date-walk cursor even if no
      qualifying filings are parsed.
    """

    return {
        "ticker": ticker,
        "validity_window": validity_window,
        "candidate": candidate,
        "name_space_bindings": name_space_bindings,
        "logger": logger,
        "entry": entry,
        "oldest_filing_date": (
            oldest_filing_date if oldest_filing_date is not None else validity_window[1]
        ),
    }


def extract_entries_and_namespace(
    response: requests.Response, logger: InfraLogger
) -> tuple[Entries, NameSpaceBindings | None]:
    """
    Parse an EDGAR Atom response into `<entry>` elements and namespace bindings.

    Parameters
    ----------
    response : requests.Response
        HTTP response from a company Atom feed request.
    logger : InfraLogger
        Logger for warnings when the root element is not an Atom `<feed>`.

    Returns
    -------
    tuple[Entries, NameSpaceBindings | None]
        A tuple `(entries, ns_bindings)` where:
        - `entries` is a list of Atom `<entry>` elements (possibly empty).
        - `ns_bindings` is a namespace map suitable for use with `find`/`findall`,
        or `None` when no default namespace is present.

    Raises
    ------
    None

    Notes
    -----
    - Uses `recover=True` on the XML parser to tolerate minor XML errors.
    - When the root element is not `<feed>`, logs a warning and returns an empty
      list of entries.
    - If the root element is `<html>`, this is treated as a non-Atom response and
      also yields an empty entry list.
    - When a default namespace exists, entries are located via `findall("ns:entry")`
      with a namespace map; otherwise, a local-name XPath is used.
    """

    xml_root: Any = etree.fromstring(
        response.content, parser=etree.XMLParser(ns_clean=True, recover=True)
    )
    root_local: str = etree.QName(xml_root).localname.lower()

    if root_local != "feed":
        logger.warning(
            "unexpected_atom_root",
            msg=f"expected 'feed' root, got '{root_local}'",
            context={"stage": "extract_entries_and_namespace"},
        )
        if root_local == "html":
            return [], None
        return [], None
    name_space: Any = xml_root.nsmap.get(None, None)
    if name_space is not None:
        entries = xml_root.findall("ns:entry", namespaces={"ns": name_space})
        return entries, {"ns": name_space}
    else:
        entries = xml_root.xpath("//*[local-name()='entry']")
        return entries, None


def find_element(
    parent: Any, tag: str, name_space_bindings: NameSpaceBindings | None
) -> Any | None:
    """
    Find the first descendant element with a given local tag name, with namespace fallback.

    Parameters
    ----------
    parent : Any
        lxml element under which to search.
    tag : str
        Local tag name to match (e.g., `"updated"`, `"title"`, `"id"`).
    name_space_bindings : NameSpaceBindings | None
        Namespace map for namespaced lookups; `None` triggers a local-name XPath.

    Returns
    -------
    Any | None
        The first matching node, or `None` if no match is found.

    Raises
    ------
    None

    Notes
    -----
    - When `name_space_bindings` is provided, uses `parent.find("ns:<tag>", namespaces=...)`.
    - When `name_space_bindings` is `None`, uses a descendant local-name XPath
      (`.//*[local-name()='<tag>']`) and returns the first result if any.
    """

    if name_space_bindings:
        return parent.find(f"ns:{tag}", namespaces=name_space_bindings)
    else:
        local_name_node: Any = parent.xpath(f".//*[local-name()='{tag}']")
        return local_name_node[0] if local_name_node else None


def find_all_elements(
    parent: Any, tag: str, name_space_bindings: NameSpaceBindings | None
) -> List[Any]:
    """
    Find all descendant elements with a given local tag name, with namespace fallback.

    Parameters
    ----------
    parent : Any
        lxml element under which to search.
    tag : str
        Local tag name to match.
    name_space_bindings : NameSpaceBindings | None
        Namespace map for namespaced lookups; `None` triggers a local-name XPath.

    Returns
    -------
    list[Any]
        A (possibly empty) list of matching nodes.

    Raises
    ------
    None

    Notes
    -----
    - Mirrors `find_element(...)` but returns all matches rather than the first one.
    """

    if name_space_bindings:
        return parent.findall(f"ns:{tag}", namespaces=name_space_bindings)
    else:
        return parent.xpath(f".//*[local-name()='{tag}']")


def set_atom_entry(
    entry: Any,
    link_href: str,
    updated_ts: pd.Timestamp,
    raw_record: RawRecord,
    name_space_bindings: NameSpaceBindings | None,
) -> None:
    """
    Persist minimal Atom entry metadata into a `RawRecord`.

    Parameters
    ----------
    entry : Any
        Atom `<entry>` element.
    link_href : str
        The entry's `rel="alternate"` link pointing to the filing index page.
    updated_ts : pandas.Timestamp
        The entry's `<updated>` timestamp (UTC).
    raw_record : RawRecord
        Mutable per-entry record to enrich; the `"atom_entry"` field is set or
        overwritten.
    name_space_bindings : NameSpaceBindings | None
        Namespace map (or `None`) used to resolve `id` and `title` within `entry`.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    - Captures the following in `raw_record["atom_entry"]`:
        - `alternate_link` – the filing index URL,
        - `updated` – the `<updated>` timestamp in ISO-8601 form,
        - `entry_id` – the Atom `<id>` text, if present,
        - `title` – the Atom `<title>` text, if present.
    - Resolves `id` and `title` via `find_element(...)` to handle both namespaced
      and non-namespaced feeds.
    """

    entry_id = find_element(entry, "id", name_space_bindings)
    entry_title = find_element(entry, "title", name_space_bindings)
    raw_record["atom_entry"] = {
        "alternate_link": link_href,
        "updated": updated_ts.isoformat(),
        "entry_id": entry_id.text if entry_id is not None else None,
        "title": entry_title.text if entry_title is not None else None,
    }
