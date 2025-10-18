"""
Purpose
-------
Harvest SEC EDGAR Atom feed entries for a set of tickers and date windows, extract
minimal filing metadata from the linked "complete submission" TXT, and accumulate:
(1) MappingEvidence records per (ticker × window) and (2) a set of candidate company names.

Key behaviors
-------------
- Iterates each ticker’s validity windows and pages the EDGAR Atom feed with a fixed page size.
- Applies a left-closed, right-open validity window filter to both the Atom entry timestamp
  ("updated") and the filing's "FILED AS OF DATE".
- For each qualifying entry, follows the "alternate" link, locates a TXT filing, parses
  a small header subset, and builds MappingEvidence.
- Emits structured logs (DEBUG/INFO/WARNING/ERROR) via the provided InfraLogger.
- Respects SEC rate limiting between page fetches.

Conventions
-----------
- Timezone: all timestamps normalized/compared in UTC.
- Window semantics: validity_window[0] <= ts < validity_window[1].
- Pagination: page size is REQUEST_COUNT; loop terminates when the final page is detected.
- Rate limiting: sleeps REQUEST_DELAY_SECONDS (+ small jitter) between page fetches
  (happens within make_request).

Downstream usage
----------------
- Call `collect_evidence(...)` with:
  - `ticker_validity_windows: dict[str, ValidityWindows]`
  - `collected_evidence: dict[str, dict[str, dict[str, MappingEvidence]]]` keyed by
    "YYYY-MM-DD to YYYY-MM-DD" window strings and evidence IDs.
  - `potential_names: dict[str, dict[str, dict[str, NameRecord]]]`
  - `logger: InfraLogger`
- This module mutates `collected_evidence` and `potential_names` in place; it does not return data.
- Network behavior and retries are delegated to `make_request(...)` (context manager).
"""

from typing import Any, List, TypeAlias

import pandas as pd
import requests
from lxml import etree

from infra.logging.infra_logger import InfraLogger
from infra.seeds.seed_firm_characteristics.edgar_search.edgar_filing_parse import (
    FilledLinkData,
    handle_alternate_link,
)
from infra.seeds.seed_firm_characteristics.edgar_search.edgar_search_utils import (
    ELIGIBLE_FORM_TYPES,
    REQUEST_COUNT,
    REQUEST_DELAY_SECONDS,
    CollectedEvidence,
    NameSpaceBindings,
    RunData,
    build_company_feed_url,
    create_run_data,
    evaluate_page_break_conditions,
    evaluate_updated_timestamp,
    find_all_elements,
    find_element,
    set_atom_entry,
    within_validity_window,
)
from infra.seeds.seed_firm_characteristics.records.raw_record import (
    SCHEMA_VERSION,
    RawQuery,
    RawRecord,
)
from infra.seeds.seed_firm_characteristics.records.table_records import (
    MappingEvidence,
    NameRecord,
    ValidityWindow,
    ValidityWindows,
    build_mapping_evidence,
    validity_window_to_str,
)
from infra.utils.requests_utils import make_request

Entries: TypeAlias = List[Any]

# ticker -> window_str -> evidence_id -> NameRecord
PotentialNames: TypeAlias = dict[str, dict[str, dict[str, NameRecord]]]

SOURCE: str = "edgar:atom→filing-index→submission.txt"
PRODUCER: str = "fetch_edgar_evidence"


def collect_evidence(
    ticker_validity_windows: dict[str, ValidityWindows],
    collected_evidence: CollectedEvidence,
    potential_names: PotentialNames,
    logger: InfraLogger,
) -> None:
    """
    Drive EDGAR harvesting for all tickers and their validity windows.

    Parameters
    ----------
    ticker_validity_windows : dict[str, ValidityWindows]
        Mapping from ticker to a list of (window_start, window_end) timestamps (UTC).
    collected_evidence : dict[str, dict[str, dict[str, MappingEvidence]]]
        Output accumulator; for each ticker and window-string key, appends MappingEvidence.
    potential_names : dict[str, dict[str, dict[str, NameRecord]]]
        Output accumulator of unique company names per ticker.
    logger : InfraLogger
        Structured logger; events carry stage='edgar_search' context.

    Returns
    -------
    None

    Notes
    -----
    - Creates a run wide requests.Session for connection reuse.
    - Emits an INFO `collect_start` with the number of tickers and DEBUG `collect_window`
      per (ticker × window) iteration.
    - Delegates page fetching and entry handling to `fetch_edgar_evidence(...)`.
    """

    logger.info(
        "collect_start", context={"stage": "edgar_search", "tickers": len(ticker_validity_windows)}
    )
    with requests.Session() as session:
        for ticker in ticker_validity_windows.keys():
            windows = ticker_validity_windows[ticker]
            for window in windows:
                current_window_str = validity_window_to_str(window)
                logger.info(
                    "collect_window",
                    context={
                        "stage": "edgar_search",
                        "ticker": ticker,
                        "window_start": window[0].isoformat(),
                        "window_end": window[1].isoformat(),
                    },
                )
                fetch_edgar_evidence(
                    ticker,
                    window,
                    logger,
                    collected_evidence[ticker][current_window_str],
                    potential_names[ticker][current_window_str],
                    session,
                )
                logger.info(
                    "finished_collection_for_window",
                    context={
                        "stage": "edgar_search",
                        "ticker": ticker,
                        "amount_of_evidence_collected": len(
                            collected_evidence[ticker][current_window_str]
                        ),
                        "amount_of_collected_names": len(
                            potential_names[ticker][current_window_str]
                        ),
                    },
                )


def fetch_edgar_evidence(
    ticker: str,
    validity_window: ValidityWindow,
    logger: InfraLogger,
    current_collected_evidence: dict[str, MappingEvidence],
    current_potential_names: dict[str, NameRecord],
    session: requests.Session,
) -> None:
    """
    Fetch and scan the EDGAR Atom feed pages for a single (ticker, validity_window).

    Parameters
    ----------
    ticker : str
        Canonicalized ticker to query in the EDGAR company feed.
    validity_window : ValidityWindow
        (start_utc, end_utc) window; start inclusive, end exclusive.
    logger : InfraLogger
        Structured logger used throughout this scan.
    current_collected_evidence : dict[str, MappingEvidence]
        Per-window evidence list to append to as records are found.
    current_potential_names : dict[str, NameRecord]]]
        Per-window list to append NameRecords to.
    session : requests.Session
        requests.Session to use for HTTP requests.

    Returns
    -------
    None

    Notes
    -----
    - Builds the EDGAR company Atom feed URL per page using `build_company_feed_url(...)`.
    - For each page:
        - Parses entries and namespace bindings.
        - For each entry, assembles RunData and determines `updated_ts` via
            `evaluate_updated_timestamp(...)`. Entries older than the window end the scan early.
        - Hands qualifying entries to `handle_entry(...)`.
        - Page loop stops when `evaluate_page_break_conditions(...)` signals final page.
        - Respects `REQUEST_DELAY_SECONDS` between pages.
    """

    start_offset: int = 0
    inclusive_window_end: str = (validity_window[1] - pd.Timedelta(days=1)).strftime("%Y%m%d")
    while True:
        url: str = build_company_feed_url(ticker, inclusive_window_end, start_offset, logger)

        # Fetch the page
        with make_request(
            url, expect_json=False, sleep_time=REQUEST_DELAY_SECONDS, session=session
        ) as response:
            entries, name_space_bindings = extract_entries_and_namespace(response, logger)
            for entry in entries:
                if not check_entry_form_type_eligibility(entry, name_space_bindings):
                    continue
                run_data: RunData = create_run_data(
                    ticker, validity_window, name_space_bindings, logger, entry
                )
                updated_ts, missing_updated = evaluate_updated_timestamp(run_data)
                if updated_ts is None:
                    if not missing_updated:
                        return None
                    continue
                handle_entry(
                    run_data,
                    current_collected_evidence,
                    current_potential_names,
                    updated_ts,
                    session,
                )
            if evaluate_page_break_conditions(entries, logger, ticker, start_offset):
                return None
            start_offset += REQUEST_COUNT


def check_entry_form_type_eligibility(
    entry: Any, name_space_bindings: NameSpaceBindings | None
) -> bool:
    """
    Validate whether an Atom entry’s form type is in ELIGIBLE_FORM_TYPES using the entry’s category
    or title fields.

    Parameters
    ----------
    entry : Any
        Parsed Atom entry element (e.g., an lxml Element) for a single SEC filing.
    name_space_bindings : NameSpaceBindings | None
        Optional XML namespace bindings used by `find_element` when locating nodes.

    Returns
    -------
    bool
        True if the entry’s form type (from `category@term` or the first token of `title`)
        is present in ELIGIBLE_FORM_TYPES; otherwise False. If neither field exists, returns False.

    Raises
    ------
    None
        This function does not raise exceptions under normal circumstances.

    Notes
    -----
    - Resolution order:
        1) `category` element: read attribute `term`, then normalize via `.upper().strip()`.
        2) If absent, `title` element: take the first whitespace-separated token, then normalize via
           `.upper().strip()`.
    - Comparison is case-insensitive due to uppercasing.
    - `ELIGIBLE_FORM_TYPES` is a global set of allowed forms (e.g., {"10-K", "10-Q", "8-K", ...}).
    - `find_element` abstracts XML lookup and may rely on `name_space_bindings`.
    """

    form_type_category: Any = find_element(entry, "category", name_space_bindings)
    if form_type_category is not None:
        form_type: str = form_type_category.get("term", "").upper().strip()
        if form_type not in ELIGIBLE_FORM_TYPES:
            return False
        return True
    form_type_title: Any = find_element(entry, "title", name_space_bindings)
    if form_type_title is not None:
        form_type = form_type_title.text.split(" ")[0].strip().upper()
        if form_type not in ELIGIBLE_FORM_TYPES:
            return False
        return True
    return False


def handle_entry(
    run_data: RunData,
    current_collected_evidence: dict[str, MappingEvidence],
    current_potential_names: dict[str, NameRecord],
    updated_ts: pd.Timestamp,
    session: requests.Session,
) -> None:
    """
    Process a single Atom entry: filter by window, follow links, and enqueue evidence.

    Parameters
    ----------
    run_data : RunData
        Typed dict containing ticker, validity_window, logger, entry, and namespace bindings.
    current_collected_evidence : dict[str, MappingEvidence]
        Target list to append MappingEvidence to.
    current_potential_names : dict[str, NameRecord]]]
        List to append NameRecords to.
    updated_ts : pandas.Timestamp
        Entry's "updated" timestamp (UTC) already validated as within the window start.
    session : requests.Session
        requests.Session to use for HTTP requests.

    Returns
    -------
    None

    Notes
    -----
    - Delegates core extraction to `extract_entry_data(...)`.
    """

    gathered_data: tuple[MappingEvidence, NameRecord | None] | None = extract_entry_data(
        run_data, updated_ts, session
    )
    return append_collected_evidence(
        run_data["ticker"],
        run_data["logger"],
        gathered_data,
        current_collected_evidence,
        current_potential_names,
    )


def extract_entries_and_namespace(
    response: requests.Response, logger: InfraLogger
) -> tuple[Entries, NameSpaceBindings | None]:
    """
    Parse the EDGAR Atom XML response into entry elements and a namespace map.

    Parameters
    ----------
    response : requests.Response
        HTTP response containing Atom XML.
    logger : InfraLogger
        Structured logger for error/debug events.

    Returns
    -------
    tuple[list[Any], NameSpaceBindings | None]
        A list of entry elements and a minimal namespace bindings dict or None.

    Notes
    -----
    - If the XML root is not an Atom <feed>, returns ([], None).
    - Supports both namespaced and no-namespace XML via XPath fallbacks.
    """

    xml_root: Any = etree.fromstring(
        response.content, parser=etree.XMLParser(ns_clean=True, recover=True)
    )
    root_local: str = etree.QName(xml_root).localname.lower()
    if root_local != "feed":
        logger.warning(
            "unexpected_xml_root",
            context={"stage": "edgar_search", "root_local_name": root_local},
        )
        return [], None
    name_space: Any = xml_root.nsmap.get(None, None)
    if name_space is not None:
        entries = xml_root.findall("ns:entry", namespaces={"ns": name_space})
        return entries, {"ns": name_space}
    else:
        entries = xml_root.xpath("//*[local-name()='entry']")
        return entries, None


def extract_entry_data(
    run_data: RunData, updated_ts: pd.Timestamp, session: requests.Session
) -> tuple[MappingEvidence, NameRecord | None] | None:
    """
    From one Atom entry, follow the filing link, parse the TXT header, and build evidence.

    Parameters
    ----------
    run_data : RunData
        Contains ticker, validity window, logger, entry, and namespace bindings.
    updated_ts : pandas.Timestamp
        Entry's "updated" timestamp (UTC) used for initial window gating.
    session : requests.Session
        requests.Session to use for HTTP requests.

    Returns
    -------
    tuple[MappingEvidence, NameRecord] | None
        On success, a (MappingEvidence, NameRecord) pair; otherwise None.

    Notes
    -----
    - Applies window semantics twice:
        1) Entry-level (`updated_ts`) before link traversal.
        2) Filing-level (`filed_at` parsed from TXT) before building evidence.
    - Constructs a `RawRecord` capturing query inputs, Atom entry metadata,
      filing index page, and raw HTTP of the TXT.
    - Constructs NameRecord only if a non-empty company name is parsed.
    - Uses `handle_alternate_link(...)` to resolve the TXT URL and parse headers.
    """

    validity_window: ValidityWindow = run_data["validity_window"]
    ticker: str = run_data["ticker"]
    if within_validity_window(updated_ts, validity_window):
        raw_record: RawRecord = generate_initial_raw_record(ticker, validity_window)
        link_data: FilledLinkData | None = extract_data_from_links(
            run_data, raw_record, updated_ts, session
        )
        if link_data and within_validity_window(link_data["filed_at"], validity_window):
            evidence: MappingEvidence = build_mapping_evidence(
                ticker,
                link_data["cik"],
                link_data["filed_at"],
                validity_window,
                SOURCE,
                raw_record,
                link_data["form_type"],
                link_data["accession_num"],
            )
            if link_data["company_name"]:
                name_record: NameRecord = create_name_record(link_data["company_name"], evidence)
                return evidence, name_record
            return evidence, None
    return None


def generate_initial_raw_record(ticker: str, validity_window: ValidityWindow) -> RawRecord:
    """
    Create a minimal RawRecord scaffold capturing query parameters and provenance.

    Parameters
    ----------
    ticker : str
        Ticker used in the EDGAR company feed request.
    validity_window : ValidityWindow
        (start_utc, end_utc) used to build the Atom query.

    Returns
    -------
    RawRecord
        Dict containing schema_version, source_chain, producer, and raw_query fields.

    Notes
    -----
    - Additional fields (`atom_entry`, `filing_page`, `raw_http`) are added downstream.
    """

    raw_query: RawQuery = {
        "kind": "edgar_atom",
        "ticker": ticker,
        "window_start": validity_window[0].strftime("%Y%m%d"),
        "window_end": validity_window[1].strftime("%Y%m%d"),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "source_chain": SOURCE,
        "producer": PRODUCER,
        "raw_query": raw_query,
    }


def extract_data_from_links(
    run_data: RunData, raw_record: RawRecord, updated_ts: pd.Timestamp, session: requests.Session
) -> FilledLinkData | None:
    """
    Locate the Atom entry's filing link, set Atom metadata on the RawRecord, and parse TXT.

    Parameters
    ----------
    run_data : RunData
        Entry, namespace bindings, ticker context, and logger.
    raw_record : RawRecord
        Mutable record to enrich with Atom and HTTP provenance.
    updated_ts : pandas.Timestamp
        Entry "updated" timestamp to persist into RawRecord metadata.
    session : requests.Session
        requests.Session to use for HTTP requests.

    Returns
    -------
    LinkData | None
        Parsed header fields from the TXT (accession, cik, form_type, filed_at, company_name),
        or None if no suitable link/TXT is found.

    Notes
    -----
    - Scans entry <link> elements for rel="alternate"; if present, persists Atom metadata
      via `set_atom_entry(...)` and calls `handle_alternate_link(...)`.
    """

    entry: Any = run_data["entry"]
    name_space_bindings: NameSpaceBindings | None = run_data["name_space_bindings"]
    entry_links: List[Any] = find_all_elements(entry, "link", name_space_bindings)
    for link in entry_links:
        link_rel: str = link.get("rel", None)
        if link_rel == "alternate":
            link_href: str = link.get("href", None)
            if link_href is not None:
                set_atom_entry(entry, link_href, updated_ts, raw_record, name_space_bindings)
                return handle_alternate_link(link_href, raw_record, run_data["logger"], session)
    return None


def append_collected_evidence(
    ticker: str,
    logger: InfraLogger,
    gathered_data: tuple[MappingEvidence, NameRecord | None] | None,
    current_collected_evidence: dict[str, MappingEvidence],
    current_potential_names: dict[str, NameRecord],
) -> None:
    """
    Append a parsed evidence/name pair to the current accumulators with debug logs.

    Purpose
    -------
    - If `gathered_data` is present, persist the `MappingEvidence` under its `evidence_id`
      and, when available, persist the companion `NameRecord`.

    Parameters
    ----------
    ticker : str
        Ticker used for logging context.
    logger : InfraLogger
        Structured logger for DEBUG events.
    gathered_data : tuple[MappingEvidence, NameRecord | None] | None
        Output from `extract_entry_data(...)`; None when the entry was filtered out.
    current_collected_evidence : dict[str, MappingEvidence]
        Per-window evidence mapping keyed by `evidence_id`.
    current_potential_names : dict[str, NameRecord]
        Per-window name mapping keyed by `evidence_id`.

    Returns
    -------
    None

    Notes
    -----
    - Emits 'evidence_recorded' and (when applicable) 'potential_name_added' DEBUG logs.
    - No-op when `gathered_data` is None.
    """

    if gathered_data is not None:
        entry_evidence, potential_name = gathered_data
        logger.debug(
            "evidence_recorded",
            context={
                "stage": "edgar_search",
                "ticker": ticker,
                "form_type": entry_evidence.form_type,
                "cik": entry_evidence.candidate_cik,
                "filed_at": entry_evidence.filed_at.isoformat(),
                "accession": entry_evidence.accession_num,
            },
        )
        current_collected_evidence[entry_evidence.evidence_id] = entry_evidence
        if potential_name:
            current_potential_names[entry_evidence.evidence_id] = potential_name
            logger.debug(
                "potential_name_added",
                context={
                    "stage": "edgar_search",
                    "ticker": ticker,
                    "company_name": potential_name.name,
                },
            )
    return None


def create_name_record(potential_name: str, entry_evidence: MappingEvidence) -> NameRecord:
    """
    Create a NameRecord from a candidate company name and its associated evidence.

    Parameters
    ----------
    potential_name : str
        Candidate company name extracted from the filing.
    entry_evidence : MappingEvidence
        Evidence record associated with the candidate name.

    Returns
    -------
    NameRecord
        Structured record containing the name,
        associated CIK, validity window, source, and evidence ID.
    """
    return NameRecord(
        cik=entry_evidence.candidate_cik,
        validity_window=entry_evidence.validity_window,
        name=potential_name,
        source=SOURCE,
        evidence_id=entry_evidence.evidence_id,
    )
