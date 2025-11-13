"""
Purpose
-------
Drive EDGAR harvesting across all (ticker × validity_window × candidate_cik) triples
discovered from the `wayback_candidates` table and persist the resulting evidence.
Keeps a single `requests.Session` per run and delegates per-window persistence to
the evidence loader.

Key behaviors
-------------
- Load (ticker, validity_window, candidate_cik) triples from `wayback_candidates`
  into a `CandidateDict`.
- For each (ticker, validity_window) key, optionally override the candidate CIK
  using `SHORT_WINDOW_TICKERS_MAP` when a curated mapping exists.
- Skip any (ticker, window_key) pairs found in `NO_ELIGIBLE_FORMS_PAIRS`, and any
  (ticker, window_key, candidate_cik) triples found in the caller-supplied
  `exclusion_set`.
- For each remaining (ticker, validity_window, candidate_cik) combination, invoke
  the EDGAR harvester (`fetch_edgar_evidence(...)`) and accumulate `MappingEvidence`
  into an in-memory buffer.
- Persist collected evidence via `persist_collected_data(...)`, which writes both
  evidence rows and a run-registry completion record.

Conventions
-----------
- All timestamps are UTC and represented as tz-aware `pandas.Timestamp` objects.
- Validity windows use half-open semantics: `start <= ts < end`.
- Keys in the `CandidateDict` are `(ticker, ValidityWindow)` pairs, where
  `ValidityWindow` is `(start_utc: Timestamp, end_utc: Timestamp)`.
- Keys in `NO_ELIGIBLE_FORMS_PAIRS` are `(ticker, window_key)` pairs where
  `window_key` is the canonical `'YYYY-MM-DD to YYYY-MM-DD'` string produced by
  `validity_window_to_str(...)`.
- Keys in the `exclusion_set` are `(ticker, window_key, candidate_cik)` triples
  using the same window_key convention.
- At most one `requests.Session` is created per `collect_evidence(...)` call.

Downstream usage
----------------
Call `collect_evidence(exclusion_set, run_id, logger)` from the seed-evidence
orchestrator after:
- seeding `wayback_candidates` from Wayback/Wikipedia, and
- computing the exclusion set of completed (ticker × validity_window × candidate_cik)
  triples from `edgar_run_registry`.

This module does not perform any Wayback fallback itself; it assumes that
Wayback-derived candidate CIKs already exist in `wayback_candidates`.
"""

from typing import List
import pandas as pd
import requests

from infra.logging.infra_logger import InfraLogger
from infra.seeds.seed_evidence.edgar_search.edgar_config import (
    NO_ELIGIBLE_FORMS_PAIRS,
    SHORT_WINDOW_TICKERS_MAP,
)
from infra.seeds.seed_evidence.edgar_search.edgar_search_core import fetch_edgar_evidence
from infra.seeds.seed_evidence.loading.load_evidence_table import persist_collected_data
from infra.seeds.seed_evidence.records.evidence_record import MappingEvidence
from infra.seeds.seed_evidence.seed_evidence_types import (
    CandidateDict,
    ValidityWindow,
    validity_window_to_str,
)
from infra.utils.db_utils import connect_to_db


def collect_evidence(
    exclusion_set: set[tuple[str, str, str]], run_id: str, logger: InfraLogger
) -> None:
    """
    Run EDGAR harvesting for all (ticker × validity_window × candidate_cik) triples,
    honoring the exclusion set and curated skip lists.

    Parameters
    ----------
    exclusion_set : set[tuple[str, str, str]]
        Completed triples `(ticker, window_key, candidate_cik)` where `window_key` is
        the canonical `'YYYY-MM-DD to YYYY-MM-DD'` string; any triple in this set is
        skipped.
    run_id : str
        Identifier for this run (typically a UUID4); attached to persisted evidence
        and run-registry metadata.
    logger : InfraLogger
        Structured logger used throughout the flow; all events include
        `stage="edgar_search"` in their context.

    Returns
    -------
    None
        Iterates over all candidate triples, invokes EDGAR harvesting per window, and
        persists evidence as a side effect.

    Raises
    ------
    Exception
        Any unexpected exception from network, parsing, or persistence layers is
        propagated to the caller; this function does not swallow errors.

    Notes
    -----
    - Constructs a `CandidateDict` via `extract_wayback_candidates(...)` and logs the
    number of (ticker, window) keys as a `collect_start` event.
    - Uses a single `requests.Session` for all EDGAR requests in this run to maximize
    connection reuse.
    - For tickers present in `SHORT_WINDOW_TICKERS_MAP`, replaces the list of discovered
    candidate CIKs with a single curated CIK from the map before harvesting.
    - Skips work when `(ticker, window_key)` is in `NO_ELIGIBLE_FORMS_PAIRS` or
    `(ticker, window_key, candidate_cik)` is in the caller-provided `exclusion_set`.
    - Delegates per-window harvesting and persistence to `handle_evidence_collection(...)`.
    """

    candidate_dict: CandidateDict = extract_wayback_candidates()
    logger.info("collect_start", context={"stage": "edgar_search", "pairs": len(candidate_dict)})
    with requests.Session() as session:
        for (ticker, window), candidates in candidate_dict.items():
            for candidate in candidates:
                window_key: str = validity_window_to_str(window)
                if (ticker, window_key) in NO_ELIGIBLE_FORMS_PAIRS:
                    continue
                if (ticker, window_key, candidate) in exclusion_set:
                    logger.debug(
                        "skipping_ticker_due_to_exclusion",
                        context={
                            "stage": "edgar_search",
                            "ticker": ticker,
                            "window_start": window[0].isoformat(),
                            "window_end": window[1].isoformat(),
                            "candidate": candidate,
                        },
                    )
                    continue
                handle_evidence_collection(ticker, window, candidate, logger, run_id, session)


def extract_wayback_candidates() -> CandidateDict:
    """
    Load Wayback-derived candidate CIKs from the `wayback_candidates` table.

    Parameters
    ----------
    None

    Returns
    -------
    CandidateDict
        A mapping from `(ticker, validity_window)` to a list of candidate CIK strings.
        The `validity_window` is reconstructed as a pair of tz-aware UTC
        `pandas.Timestamp` objects from the Postgres DATERANGE bounds.

    Raises
    ------
    psycopg2.Error
        If the query against `wayback_candidates` fails or the connection cannot be
        established.
    BaseException
        Propagated from `connect_to_db(...)` on unexpected driver- or environment-
        level failures.

    Notes
    -----
    - Assumes `validity_window` is stored as a Postgres DATERANGE with finite
      `[start, end)` bounds; `row[1].lower` and `row[1].upper` are converted to
      UTC `Timestamp` objects.
    - Returns an empty `CandidateDict` when the table has no rows.
    """

    query: str = """
    SELECT ticker,
            validity_window,
            candidate_cik
    FROM wayback_cik_candidates;
    """
    with connect_to_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows: List[tuple] = cursor.fetchall()
            candidates: CandidateDict = {}
            for row in rows:
                ticker: str = row[0]
                validity_window: ValidityWindow = (
                    pd.Timestamp(row[1].lower, tz="UTC"),
                    pd.Timestamp(row[1].upper, tz="UTC"),
                )
                candidate_cik: str = row[2]
                key: tuple[str, ValidityWindow] = (ticker, validity_window)
                if key not in candidates:
                    candidates[key] = []
                candidates[key].append(candidate_cik)
            for key, candidate_list in SHORT_WINDOW_TICKERS_MAP.items():
                candidates[key] = candidate_list
            return candidates


def handle_evidence_collection(
    ticker: str,
    window: ValidityWindow,
    candidate: str,
    logger: InfraLogger,
    run_id: str,
    session: requests.Session,
) -> None:
    """
    Execute the EDGAR harvesting and persistence flow for a single
    (ticker × validity_window × candidate_cik) combination.

    Parameters
    ----------
    ticker : str
        Canonical ticker for which evidence is attributed (e.g., `"AAPL"`).
    window : ValidityWindow
        Half-open `(start_utc, end_utc)` window used for gating entries and filings.
    candidate : str
        Identifier passed to EDGAR as the `CIK` parameter; typically a 10-digit CIK
        discovered via Wayback, but may also be a ticker in curated cases.
    logger : InfraLogger
        Structured logger used for INFO/DEBUG/WARNING events.
    run_id : str
        Run identifier propagated to `persist_collected_data(...)` for auditability.
    session : requests.Session
        Shared HTTP session for issuing EDGAR requests within this run.

    Returns
    -------
    None
        Accumulates `MappingEvidence` in a local buffer and delegates persistence to
        `handle_collected_evidence(...)`.

    Raises
    ------
    Exception
        Propagates any errors from `fetch_edgar_evidence(...)` or
        `handle_collected_evidence(...)` (including database failures).

    Notes
    -----
    - Captures `start_time` at the beginning of collection for this window; this
      timestamp is persisted alongside evidence in the run registry.
    - Initializes an empty `mapping_evidence_buffer: dict[evidence_id, MappingEvidence]`
      and passes it into `fetch_edgar_evidence(...)` to be populated in place.
    - After harvesting completes, calls `handle_collected_evidence(...)` to either
      persist the buffer or log that no evidence was collected.
    """

    start_time: pd.Timestamp = pd.Timestamp.now(tz="UTC")
    mapping_evidence_buffer: dict[str, MappingEvidence] = {}
    logger.info(
        "collect_window",
        context={
            "stage": "edgar_search",
            "ticker": ticker,
            "window_start": window[0].isoformat(),
            "window_end": window[1].isoformat(),
            "candidate": candidate,
        },
    )
    fetch_edgar_evidence(
        ticker,
        candidate,
        window,
        logger,
        mapping_evidence_buffer,
        session,
    )
    handle_collected_evidence(
        ticker,
        window,
        candidate,
        run_id,
        start_time,
        logger,
        mapping_evidence_buffer,
    )


def handle_collected_evidence(
    ticker: str,
    window: ValidityWindow,
    candidate: str,
    run_id: str,
    start_time: pd.Timestamp,
    logger: InfraLogger,
    mapping_evidence_buffer: dict[str, MappingEvidence],
) -> None:
    """
    Persist accumulated evidence for a single (ticker × validity_window × candidate_cik)
    or log a warning when no evidence exists.

    Parameters
    ----------
    ticker : str
        Canonical ticker associated with the evidence.
    window : ValidityWindow
        Half-open `(start_utc, end_utc)` window used during harvesting; persisted as
        a Postgres DATERANGE with `[start, end)` bounds.
    candidate : str
        EDGAR query identifier (ticker or 10-digit CIK) used when collecting this
        window; logged for traceability.
    run_id : str
        Run identifier written alongside the run-registry completion row.
    start_time : pandas.Timestamp
        UTC timestamp captured at the start of harvesting for this window; persisted
        for audit and duration measurement.
    logger : InfraLogger
        Structured logger for INFO/DEBUG/WARNING events.
    mapping_evidence_buffer : dict[str, MappingEvidence]
        In-memory accumulator of evidence for this window, keyed by `evidence_id`.

    Returns
    -------
    None

    Raises
    ------
    Exception
        Any exception thrown by `persist_collected_data(...)` (e.g., database insert
        or commit failures) is propagated; the caller relies on transaction rollback
        from the connection context manager.

    Notes
    -----
    - When `mapping_evidence_buffer` is non-empty:
        - Logs `finished_collection_for_window` with the evidence count.
        - Calls `persist_collected_data(...)` to:
            * insert rows into `ticker_cik_evidence`, and
            * insert a completion row into `edgar_run_registry`.
        - Logs `persisted_evidence_for_window` with window bounds and final count.
    - When the buffer is empty:
        - Logs `no_evidence_persisted_for_window` at WARNING level; no database
          writes are attempted for this (ticker, window, candidate) triple.
    """

    if mapping_evidence_buffer:
        logger.info(
            "finished_collection_for_window",
            context={
                "stage": "edgar_search",
                "ticker": ticker,
                "candidate": candidate,
                "amount_of_evidence_collected": len(mapping_evidence_buffer),
            },
        )
        persist_collected_data(
            run_id,
            ticker,
            window,
            candidate,
            start_time,
            logger,
            list(mapping_evidence_buffer.values()),
        )
        logger.info(
            "persisted_evidence_for_window",
            context={
                "stage": "edgar_search",
                "ticker": ticker,
                "candidate": candidate,
                "window_start": window[0].isoformat(),
                "window_end": window[1].isoformat(),
                "evidence_count": len(mapping_evidence_buffer),
            },
        )
    else:
        logger.warning(
            "no_evidence_persisted_for_window",
            context={
                "stage": "edgar_search",
                "ticker": ticker,
                "candidate": candidate,
                "window_start": window[0].isoformat(),
                "window_end": window[1].isoformat(),
            },
        )
