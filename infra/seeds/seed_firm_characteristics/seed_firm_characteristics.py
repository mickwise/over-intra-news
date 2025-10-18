"""
Module: seed_firm_characteristics (orchestrator)

Purpose
-------
Coordinate the end-to-end seeding of firm characteristics for a configured run
date. This module wires together S&P 500 membership extraction, construction of
per-ticker validity windows, EDGAR evidence harvesting, score aggregation, and
final table loads.

High-level flow
---------------
1) Read `END_DATE` from the environment and initialize a structured logger.
2) Fetch daily S&P 500 membership snapshots for the run window.
3) Derive per-ticker [start, end) validity windows (UTC day granularity).
4) Harvest EDGAR filings within each (ticker × window) and collect:
   - MappingEvidence (for ticker→CIK).
   - Candidate company names.
5) Aggregate evidence into posteriors and best-candidate selections.
6) Delegate persistence (including any mid-window split logic) to `load_tables`.

Inputs & environment
--------------------
- `END_DATE` (required): ISO date string. Parsed to a tz-aware UTC `pd.Timestamp`.
- S&P membership CSV is accessed by `extract_snp_membership_windows(...)`.

Outputs
-------
- Populated curated tables via `load_tables(...)`. This module itself returns `None`.

Time & window semantics
-----------------------
- All timestamps are UTC.
- Validity windows are half-open `[start, end)` at day precision.
- Open windows at the tail are closed at `END_DATE + 1 day` to preserve right-open semantics.

Logging
-------
- Emits a small set of INFO logs marking major milestones:
  start, validity-window construction, scoring start/end.
- Downstream components (EDGAR harvester, loaders) emit granular logs as needed.

Idempotency & retries
---------------------
- Safe to re-run for the same `END_DATE` when downstream loaders are idempotent
  (e.g., deterministic keys / upserts).
- Exceptions are not swallowed; they propagate to the caller (job runner) for handling.

Assumptions
-----------
- `extract_snp_membership_windows` returns `date` as tz-aware UTC and `tickers` as
  comma-separated strings.
- `load_tables` performs the mid-window split for CIK episodes and builds dependent tables
  from the curated ticker↔CIK mapping.
"""

import os
from typing import TypeAlias

import pandas as pd
from dotenv import load_dotenv

from infra.logging.infra_logger import InfraLogger, initialize_logger
from infra.seeds.seed_firm_characteristics.edgar_search.edgar_search_core import (
    CollectedEvidence,
    PotentialNames,
    collect_evidence,
)
from infra.seeds.seed_firm_characteristics.loading.load_tables import load_tables
from infra.seeds.seed_firm_characteristics.records.table_records import (
    ValidityWindows,
    validity_window_to_str,
)
from infra.seeds.seed_firm_characteristics.scoring.confidence_scores import ConfidenceScores
from infra.seeds.seed_snp_memberships import extract_snp_membership_windows
from infra.utils.db_utils import str_to_timestamp

ValidTickers: TypeAlias = set[str]


def collect_firm_characteristics() -> None:
    """
    Orchestrate end-to-end seeding of firm characteristics.

    Purpose
    -------
    - Parse run configuration (END_DATE), initialize logging, and drive the pipeline:
      1) fetch S&P 500 membership snapshots for the configured window,
      2) derive per-ticker validity windows ([start, end) semantics),
      3) harvest EDGAR evidence within each window,
      4) aggregate evidence into confidence scores/posteriors, and
      5) load curated tables (including downstream split handling) via `load_tables`.

    Side effects
    ------------
    - Emits a small number of INFO logs (start, windows constructed, scoring start/done).
    - Writes to persistence layers inside `load_tables(...)`.

    Environment
    -----------
    - END_DATE: ISO date string; parsed to a UTC `pd.Timestamp`.

    Raises
    ------
    - Propagates exceptions from subroutines (fetching, scoring, or loading) for the caller
      to handle at the job boundary.
    """
    load_dotenv()
    end_date_str: str | None = os.getenv("END_DATE")
    if end_date_str is None:
        raise ValueError("END_DATE environment variable is required but not set.")
    end_date: pd.Timestamp = str_to_timestamp(end_date_str)
    logger: InfraLogger = initialize_logger(
        component_name="seed_firm_characteristics",
        level="INFO",
        run_meta={"end_date": end_date.date().isoformat()},
    )
    logger.info("collect_firm_characteristics")
    snp_membership_windows: pd.DataFrame = extract_snp_membership_windows(logger)
    ticker_validity_windows: dict[str, ValidityWindows] = extract_ticker_validity_windows(
        snp_membership_windows, end_date
    )
    logger.info(
        "ticker_validity_windows_constructed",
        context={"ticker_count": len(ticker_validity_windows.keys())},
    )
    collected_evidence, confidence_scores, potential_names = initialize_mapping_data_structures(
        ticker_validity_windows
    )
    collect_evidence(ticker_validity_windows, collected_evidence, potential_names, logger)
    logger.info("scoring_start")
    calculate_scores(collected_evidence, confidence_scores)
    logger.info("scoring_done")
    load_tables(
        ticker_validity_windows, collected_evidence, confidence_scores, potential_names, logger
    )


def extract_ticker_validity_windows(
    snp_membership_windows: pd.DataFrame, end_date: pd.Timestamp
) -> dict[str, ValidityWindows]:
    """
    Convert daily S&P membership snapshots into per-ticker validity windows.

    Purpose
    -------
    - Normalize and explode daily `tickers`, then group by day to build a clean
      time series of membership sets. From that series, derive half-open windows
      [start, end) per ticker covering its inclusion periods.

    Parameters
    ----------
    snp_membership_windows : pandas.DataFrame
        Columns: `date` (tz-aware UTC day) and `tickers` (comma-separated string).
        Assumes rows are already filtered to the overall run window.
    end_date : pandas.Timestamp
        UTC upper bound for the run; used to close any open windows.

    Returns
    -------
    dict[str, ValidityWindows]
        Mapping: ticker → list of (start_utc, end_utc) pairs with [start, end) semantics.

    Notes
    -----
    - Empty or malformed tickers are dropped after normalization.
    - Open windows at the tail are closed at `end_date + 1 day` to preserve right-open semantics.
    """
    snp_membership_windows["tickers"] = snp_membership_windows["tickers"].str.split(",")
    snp_membership_windows = snp_membership_windows.explode("tickers", ignore_index=True)
    snp_membership_windows["tickers"] = normalize_tickers(snp_membership_windows["tickers"])
    snp_membership_windows["tickers"] = snp_membership_windows["tickers"].replace("", pd.NA)
    snp_membership_windows.dropna(subset=["tickers"], inplace=True)
    snp_membership_windows = (
        snp_membership_windows.groupby("date", sort=False)["tickers"].agg(list).reset_index()
    )
    valid_tickers: ValidTickers = extract_valid_tickers(snp_membership_windows["tickers"])
    return construct_validity_windows(valid_tickers, snp_membership_windows, end_date)


def construct_validity_windows(
    valid_tickers: ValidTickers, snp_membership_windows: pd.DataFrame, end_date: pd.Timestamp
) -> dict[str, ValidityWindows]:
    """
    Build half-open membership episodes per ticker from daily set diffs.

    Purpose
    -------
    - Iterate the daily membership sets and detect joins/leaves:
        * On join: start a new window at `date` with a temporary open end.
        * On leave: close the last window at `date`.
      After the final day, close any still-open windows at `end_date + 1 day`.

    Parameters
    ----------
    valid_tickers : set[str]
        Canonical set of tickers observed in the cleaned daily snapshots.
    snp_membership_windows : pandas.DataFrame
        Columns: `date` (tz-aware UTC day) and `tickers` (list[str]) grouped by date.
    end_date : pandas.Timestamp
        UTC upper bound for the run; used to close open windows.

    Returns
    -------
    dict[str, ValidityWindows]
        Mapping: ticker → list of (start_utc, end_utc) with [start, end) semantics.

    Notes
    -----
    - Window ends equal the first day the ticker is *not* in the index.
    - Closing at `end_date + 1 day` preserves right-open intervals up to the run boundary.
    """
    ticker_windows: dict[str, ValidityWindows] = {ticker: [] for ticker in valid_tickers}
    proxy_timestamp: pd.Timestamp = pd.Timestamp("0001-01-01", tz="UTC")

    # Iterate through membership windows to build validity periods
    previous_snp_members: set = set()
    for date, tickers in snp_membership_windows.itertuples(index=False):
        current_snp_members: set[str] = set(tickers)
        added_tickers: set[str] = current_snp_members - previous_snp_members
        removed_tickers: set[str] = previous_snp_members - current_snp_members
        for ticker in added_tickers:
            ticker_windows[ticker].append((date, proxy_timestamp))
        for ticker in removed_tickers:
            if ticker_windows[ticker]:
                start, _ = ticker_windows[ticker][-1]
                ticker_windows[ticker][-1] = (start, date)
        previous_snp_members = current_snp_members

    # Close any open validity windows at the end date and collect
    for ticker in previous_snp_members:
        if ticker_windows[ticker]:
            start, _ = ticker_windows[ticker][-1]
            ticker_windows[ticker][-1] = (start, end_date + pd.Timedelta(days=1))

    return ticker_windows


def normalize_tickers(tickers: pd.Series) -> pd.Series:
    """
    Normalize raw ticker strings to a canonical exchange format.

    Transformations
    ---------------
    - Uppercase
    - Trim whitespace
    - Remove internal whitespace
    - Map '.' and '/' to '-' (exchange-style class separators)

    Parameters
    ----------
    tickers : pandas.Series
        Series of raw ticker strings.

    Returns
    -------
    pandas.Series
        Cleaned tickers; invalid entries may become empty strings and are dropped upstream.
    """
    tickers = tickers.str.upper().str.strip()
    tickers = tickers.str.replace(r"\s+", "", regex=True)
    tickers = tickers.str.replace(r"[./]", "-", regex=True)
    return tickers


def extract_valid_tickers(tickers: pd.Series) -> ValidTickers:
    """
    Collect the unique set of tickers present across all days.

    Parameters
    ----------
    tickers : pandas.Series
        Series of lists of normalized tickers per day.

    Returns
    -------
    set[str]
        Union of all tickers observed in the series.
    """
    valid_tickers: set = set()
    for ticker_list in tickers:
        valid_tickers.update(ticker_list)
    return valid_tickers


def initialize_mapping_data_structures(
    ticker_validity_windows: dict[str, ValidityWindows],
) -> tuple[CollectedEvidence, ConfidenceScores, PotentialNames]:
    """
    Initialize empty, per-window accumulators for evidence, scores, and candidate names.

    Purpose
    -------
    - For each (ticker × validity_window), create:
    1) an evidence bucket keyed by deterministic `evidence_id`,
    2) a confidence-score accumulator (via `ConfidenceScores`), and
    3) a potential-names bucket keyed by `evidence_id`.

    Parameters
    ----------
    ticker_validity_windows : dict[str, ValidityWindows]
        Mapping of ticker → list of half-open [start, end) windows (tz-aware UTC).

    Returns
    -------
    tuple[CollectedEvidence, ConfidenceScores, PotentialNames]
        - collected_evidence: ticker → window_key → evidence_id → MappingEvidence (initially empty).
        - confidence_scores: mutable accumulator pre-seeded with all (ticker, window_key) slots.
        - potential_names: ticker → window_key → evidence_id → NameRecord (initially empty).

    Notes
    -----
    - The `window_key` format is the canonical 'YYYY-MM-DD to YYYY-MM-DD'.
    - All three structures are aligned on the same (ticker, window_key) grid.
    """

    collected_evidence: CollectedEvidence = {
        ticker: {validity_window_to_str(window): {} for window in windows}
        for ticker, windows in ticker_validity_windows.items()
    }
    confidence_scores: ConfidenceScores = ConfidenceScores(ticker_validity_windows)
    potential_names: PotentialNames = {
        ticker: {validity_window_to_str(window): {} for window in windows}
        for ticker, windows in ticker_validity_windows.items()
    }
    return collected_evidence, confidence_scores, potential_names


def calculate_scores(
    collected_evidence: CollectedEvidence, confidence_scores: ConfidenceScores
) -> None:
    """
    Aggregate MappingEvidence into per-window confidence scores/posteriors.

    Purpose
    -------
    - Iterate all per-window evidence lists and feed them (deduped by evidence_id)
      into `confidence_scores.batch_evidence_to_scores(...)`.

    Parameters
    ----------
    collected_evidence : CollectedEvidence
        Ticker → window_key → evidence_id -> MappingEvidence.
    confidence_scores : ConfidenceScores
        Mutable accumulator used to store scores/posteriors.

    Returns
    -------
    ConfidenceScores
        The same accumulator instance for convenience chaining.

    Notes
    -----
    - Idempotent with respect to repeated calls only if upstream evidence lists
      themselves are not appended to multiple times.
    """
    for ticker, windows_dict in collected_evidence.items():
        if ticker != "CRL":
            continue
        for _, evidences_dict in windows_dict.items():
            evidences = list(evidences_dict.values())
            confidence_scores.batch_evidence_to_scores(evidences)
    return None


collect_firm_characteristics()
