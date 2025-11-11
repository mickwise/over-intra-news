"""
Purpose
-------
Harvest SEC EDGAR Atom company-feed entries for a candidate CIK within a
validity window, follow filing links, parse minimal header fields, and build
`MappingEvidence` records for persistence.

Key behaviors
-------------
- Implements a date-walk paginator over the EDGAR Atom feed by stepping the `dateb`
  parameter backward in time.
- Gates entries on a half-open validity window using the Atom `<updated>` timestamp
  and the filing “FILED AS OF DATE”.
- Filters entries to eligible form types before following their `rel="alternate"` links.
- For qualifying filings, constructs `MappingEvidence` enriched with a `RawRecord`
  provenance payload.
- Uses `InfraLogger` for structured logging and `REQUEST_DELAY_SECONDS` to pace HTTP
  requests.

Conventions
-----------
- All timestamps are treated as UTC.
- Window semantics are half-open: `start <= ts < end`.
- Pagination never uses deep offsets: `start` is always 0 and only `dateb` is moved.

Downstream usage
----------------
- Called by orchestrators with:
  - `ticker` as the canonical project ticker (for attribution), and
  - `candidate` as the identifier passed to EDGAR (`CIK` parameter; ticker or 10-digit CIK).
- The caller provides a mutable `current_collected_evidence` mapping and a shared
  `requests.Session`; this module appends `MappingEvidence` into that accumulator.
"""

from typing import Any, List

import pandas as pd
import requests

from infra.logging.infra_logger import InfraLogger
from infra.seeds.seed_evidence.edgar_search.edgar_config import REQUEST_DELAY_SECONDS, SOURCE
from infra.seeds.seed_evidence.edgar_search.edgar_filing_parse import (
    FilledLinkData,
    handle_alternate_link,
)
from infra.seeds.seed_evidence.edgar_search.edgar_search_conditions import (
    check_entry_form_type_eligibility,
    evaluate_page_break_conditions,
    evaluate_updated_timestamp,
    within_validity_window,
)
from infra.seeds.seed_evidence.edgar_search.edgar_search_utils import (
    RunData,
    build_url,
    create_run_data,
    extract_entries_and_namespace,
    find_all_elements,
    set_atom_entry,
)
from infra.seeds.seed_evidence.records.evidence_record import (
    MappingEvidence,
    build_mapping_evidence,
)
from infra.seeds.seed_evidence.records.raw_record import RawRecord, generate_initial_raw_record
from infra.seeds.seed_evidence.seed_evidence_types import Entries, NameSpaceBindings, ValidityWindow
from infra.utils.requests_utils import make_request


def fetch_edgar_evidence(
    ticker: str,
    candidate: str,
    validity_window: ValidityWindow,
    logger: InfraLogger,
    current_collected_evidence: dict[str, MappingEvidence],
    session: requests.Session,
) -> None:
    """
    Scan the EDGAR Atom company feed for a candidate identifier and collect
    `MappingEvidence` within a validity window using a date-walk strategy.

    Parameters
    ----------
    ticker : str
        Canonical ticker used for attribution in the constructed evidence.
    candidate : str
        Identifier passed to EDGAR as the `CIK` parameter (ticker or 10-digit CIK).
    validity_window : ValidityWindow
        Half-open `(start_utc, end_utc)` window applied to both Atom `<updated>` and
        filing “FILED AS OF DATE”.
    logger : InfraLogger
        Structured logger used for debug/progress logging.
    current_collected_evidence : dict[str, MappingEvidence]
        Mutable accumulator updated in place, keyed by `evidence_id`.
    session : requests.Session
        Shared HTTP session for connection reuse.

    Returns
    -------
    None
        Returns when the Atom feed has been fully walked for the given window or when
        an out-of-window `<updated>` timestamp is encountered.

    Raises
    ------
    Exception
        Propagates network, XML parsing, or filing parsing errors from downstream
        helpers and HTTP utilities.

    Notes
    -----
    - Date-walk paginator:
        1) Initialize `page_cursor` as `None` and build the first URL using
        `(window_end - 1 day)` as `dateb`.
        2) Parse the page into entries and `RunData`.
        3) For each entry:
            - assign it to `run_data["entry"]`,
            - evaluate `<updated>` via `evaluate_updated_timestamp(...)`,
            - filter by form type,
            - follow `rel="alternate"` links and optionally build `MappingEvidence`.
        4) Track the oldest filing/`updated` timestamp seen on the page in
           `run_data["oldest_filing_date"]`.
        5) After processing a page, decide whether to stop via
           `evaluate_page_break_conditions(...)`.
        6) If continuing, set `page_cursor = oldest_filing_date - 1 day` and repeat.
    - `current_collected_evidence` is the only persistence channel; no value is
       returned from the function.
    """

    page_cursor: pd.Timestamp | None = None
    while True:
        url: str = build_url(
            candidate,
            validity_window,
            logger,
            page_cursor,
        )
        with make_request(
            url, expect_json=False, sleep_time=REQUEST_DELAY_SECONDS, session=session
        ) as response:
            entries, run_data = init_run_data(
                ticker, candidate, validity_window, logger, response, page_cursor
            )
            for entry in entries:
                run_data["entry"] = entry
                updated_ts = evaluate_updated_timestamp(run_data)
                if updated_ts is None:
                    return None
                if not check_entry_form_type_eligibility(entry, run_data["name_space_bindings"]):
                    continue
                run_data["oldest_filing_date"] = min(run_data["oldest_filing_date"], updated_ts)
                raw_record: RawRecord = generate_initial_raw_record(candidate, validity_window)
                handle_entry(run_data, current_collected_evidence, updated_ts, session, raw_record)
            if evaluate_page_break_conditions(entries, run_data):
                return None
            page_cursor = run_data["oldest_filing_date"] - pd.Timedelta(days=1)


def init_run_data(
    ticker: str,
    candidate: str,
    validity_window: ValidityWindow,
    logger: InfraLogger,
    response: requests.Response,
    page_cursor: pd.Timestamp | None,
) -> tuple[Entries, RunData]:
    """
    Initialize `RunData` for a single Atom page and extract its `<entry>` elements.

    Parameters
    ----------
    ticker : str
        Canonical ticker for attribution in evidence records.
    candidate : str
        Identifier used in the EDGAR query (`CIK` parameter).
    validity_window : ValidityWindow
        Half-open `(start_utc, end_utc)` window used downstream for gating.
    logger : InfraLogger
        Structured logger for this run.
    response : requests.Response
        HTTP response returned from the EDGAR Atom endpoint.
    page_cursor : pandas.Timestamp | None
        Effective `dateb` cutoff for this page; when `None`, the window end is used
        to seed `oldest_filing_date`.

    Returns
    -------
    tuple[Entries, RunData]
        A pair of:
        - `entries`: the list of Atom `<entry>` elements for this page.
        - `run_data`: context bundle initialized for this page, ready to be updated
        per entry.

    Raises
    ------
    Exception
        Propagates XML parsing errors if the response body cannot be parsed into an
        XML tree.
    """

    entries, name_space_bindings = extract_entries_and_namespace(response, logger)
    run_data: RunData = create_run_data(
        ticker, validity_window, candidate, name_space_bindings, logger, None, page_cursor
    )
    return entries, run_data


def handle_entry(
    run_data: RunData,
    current_collected_evidence: dict[str, MappingEvidence],
    updated_ts: pd.Timestamp,
    session: requests.Session,
    raw_record: RawRecord,
) -> None:
    """
    Process a single Atom `<entry>`: follow its filing link and accumulate evidence.

    Parameters
    ----------
    run_data : RunData
        Per-entry context including ticker, candidate, validity window, namespaces,
        logger, and the current `<entry>`.
    current_collected_evidence : dict[str, MappingEvidence]
        In-place accumulator keyed by `evidence_id`.
    updated_ts : pandas.Timestamp
        The entry's `<updated>` timestamp (UTC) already validated against the window.
    session : requests.Session
        Shared HTTP session for following filing links.
    raw_record : RawRecord
        Mutable provenance record to enrich with Atom and filing metadata.

    Returns
    -------
    None

    Raises
    ------
    Exception
        Propagates errors from link traversal, filing parsing, or evidence construction.

    Notes
    -----
    - Delegates to `extract_entry_data(...)` to build a `MappingEvidence` instance,
      then to `append_collected_evidence(...)` to log and store it.
    """

    gathered_data: MappingEvidence | None = extract_entry_data(
        run_data, updated_ts, session, raw_record
    )
    return append_collected_evidence(
        run_data["candidate"],
        run_data["logger"],
        gathered_data,
        current_collected_evidence,
    )


def extract_entry_data(
    run_data: RunData, updated_ts: pd.Timestamp, session: requests.Session, raw_record: RawRecord
) -> MappingEvidence | None:
    """
    Follow an entry's filing link, parse minimal headers, and build `MappingEvidence`.

    Parameters
    ----------
    run_data : RunData
        Context bundle including ticker, candidate, validity window, namespaces,
        logger, and the current `<entry>`.
    updated_ts : pandas.Timestamp
        The entry's `<updated>` timestamp used for provenance and window gating.
    session : requests.Session
        Shared HTTP session for network I/O.
    raw_record : RawRecord
        Root provenance record.

    Returns
    -------
    MappingEvidence | None
        A fully constructed evidence object when the filing date falls within the
        validity window; `None` when no eligible filing is found or the filing date
        is outside the window.

    Raises
    ------
    Exception
        Propagates network and parsing errors from filing index/TXT handling.

    Notes
    -----
    - Applies the validity window twice:
        1) Before link traversal via `evaluate_updated_timestamp(...)` in the caller.
        2) After parsing the filing header via `within_validity_window(...)` on
           `link_data["filed_at"]`.
    - Uses `build_mapping_evidence(...)` to construct the final record with
      ticker, CIK, filing date, form type, accession number, and company name.
    """

    validity_window: ValidityWindow = run_data["validity_window"]
    ticker: str = run_data["ticker"]
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
            link_data["company_name"],
            link_data["items_8k"],
            link_data["items_descriptions_8k"],
        )
        return evidence
    return None


def extract_data_from_links(
    run_data: RunData, raw_record: RawRecord, updated_ts: pd.Timestamp, session: requests.Session
) -> FilledLinkData | None:
    """
    Locate the `rel="alternate"` link on an entry, persist Atom metadata, and parse
    the filing headers.

    Parameters
    ----------
    run_data : RunData
        Entry context containing the current `<entry>`, namespace bindings, ticker,
        candidate, validity window, and logger.
    raw_record : RawRecord
        Entry-scoped provenance structure; `set_atom_entry(...)` writes Atom fields
        into it.
    updated_ts : pandas.Timestamp
        The entry's `<updated>` timestamp (UTC) recorded into provenance.
    session : requests.Session
        Shared HTTP session used to fetch the filing index/TXT.

    Returns
    -------
    FilledLinkData | None
        Parsed `(accession, cik, form_type, filed_at, company_name)` from the filing
        index/TXT, or `None` when no suitable `rel="alternate"` link is present or
        parsing fails to produce a valid header set.

    Raises
    ------
    Exception
        Propagates network and parsing errors from `handle_alternate_link(...)`.

    Notes
    -----
    - Uses `find_all_elements(...)` to scan for `<link>` elements in a namespace-aware
      way.
    - When an `alternate` link is found, this function:
        1) calls `set_atom_entry(...)` to persist Atom-level metadata, and
        2) delegates to `handle_alternate_link(...)` to parse the filing headers.
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
                return handle_alternate_link(link_href, raw_record, run_data, session)
    return None


def append_collected_evidence(
    candidate: str,
    logger: InfraLogger,
    gathered_data: MappingEvidence | None,
    current_collected_evidence: dict[str, MappingEvidence],
) -> None:
    """
    Append a `MappingEvidence` to the per-window accumulator with debug logging.

    Parameters
    ----------
    candidate : str
        Identifier used for the EDGAR page (ticker or candidate CIK), logged for
        traceability.
    logger : InfraLogger
        Structured logger.
    gathered_data : MappingEvidence | None
        Evidence produced by `extract_entry_data(...)`; may be `None` if the filing
        was filtered out or fell outside the validity window.
    current_collected_evidence : dict[str, MappingEvidence]
        Accumulator keyed by `evidence_id`; updated in place.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    - When `gathered_data` is not `None`, emits an `evidence_recorded` DEBUG event
      including form type, candidate CIK, filed-at timestamp, and accession number.
    - Uses the `evidence_id` field on `MappingEvidence` as the key in
      `current_collected_evidence`.
    """

    if gathered_data is not None:
        entry_evidence = gathered_data
        logger.debug(
            "evidence_recorded",
            context={
                "stage": "append_collected_evidence",
                "candidate_or_ticker": candidate,
                "form_type": entry_evidence.form_type,
                "cik": entry_evidence.candidate_cik,
                "filed_at": entry_evidence.filed_at.isoformat(),
                "accession": entry_evidence.accession_num,
            },
        )
        current_collected_evidence[entry_evidence.evidence_id] = entry_evidence
    return None
