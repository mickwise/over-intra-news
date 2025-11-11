"""
Purpose
-------
Orchestrate EDGAR evidence collection for a configured run date. This module
derives per-ticker validity windows from S&P 500 membership snapshots, seeds
Wayback-derived CIK candidates, builds a skip set from edgar_run_registry, and
dispatches the EDGAR harvester to collect filing-based evidence.

Key behaviors
-------------
- Read END_DATE from the environment and initialize structured logging.
- Fetch daily S&P 500 membership snapshots for the run horizon.
- Derive per-ticker half-open [start, end) validity windows (UTC, day precision).
- Seed Wayback CIK candidates by querying the Internet Archive for the S&P 500
  components page and loading results into wayback_candidates.
- Query edgar_run_registry to build an exclusion set of completed
  (ticker × validity_window × candidate_cik) triples and pass it downstream.
- Invoke the EDGAR evidence collector across the remaining work.

Conventions
-----------
- All timestamps are UTC.
- Validity windows use half-open [start, end) semantics; open tails are closed at
  END_DATE + 1 day.
- Tickers are normalized to uppercase; "." and "/" are mapped to "-" to match the
  EDGAR-canonical class separator.
- Wayback seeding operates over the same validity windows used for EDGAR search,
  and candidates are deduplicated by (ticker, validity_window, CIK).

Downstream usage
----------------
Call run_edgar_harvest(logger_level) from a job runner after exporting END_DATE.
This module returns None. It seeds Wayback CIK candidates, performs EDGAR evidence
collection, and relies on downstream loaders to persist evidence and write
completion rows to edgar_run_registry atomically after successful loads.
"""

import datetime as dt
import os
import uuid
from typing import List, TypeAlias

import numpy as np
import pandas as pd
import psycopg2
import requests
from dotenv import load_dotenv

from infra.logging.infra_logger import InfraLogger, initialize_logger
from infra.seeds.seed_evidence.edgar_search.edgar_search_orchestrator import collect_evidence
from infra.seeds.seed_evidence.loading.load_wayback_candidates_table import load_wayback_candidates
from infra.seeds.seed_evidence.seed_evidence_types import ValidityWindows, validity_window_to_str
from infra.seeds.seed_evidence.wayback.wayback_orchestrator import batch_extract_candidates_wayback
from infra.seeds.seed_evidence.wayback.wayback_typing import Batch, WayBackCandidate
from infra.seeds.seed_snp_memberships import extract_snp_membership_windows
from infra.utils.db_utils import connect_to_db, str_to_timestamp

ValidTickers: TypeAlias = set[str]


def run_edgar_harvest(
    logger_level: str | None, seed_wayback: bool = False, scrape_edgar: bool = False
) -> None:
    """
    Run the evidence orchestrator: build per-ticker windows, seed Wayback candidates,
    compute the exclusion set, and collect EDGAR evidence for remaining work.

    Parameters
    ----------
    logger_level: str | None
        Optional logging level override; defaults to "INFO" if not provided.
        Level must be in {"DEBUG", "INFO", "WARNING", "ERROR"}.
    seed_wayback : bool
        If True, seed Wayback-derived CIK candidates before EDGAR harvesting.
    scrape_edgar : bool
        If True, perform EDGAR evidence collection after Wayback seeding.

    Returns
    -------
    None
        Coordinates window construction, constructs the exclusion set from
        edgar_run_registry, seeds Wayback CIK candidates, and delegates to
        the EDGAR harvester.

    Raises
    ------
    ValueError
        If END_DATE is not set or cannot be parsed.
    psycopg2.Error
        Propagated from database connectivity or query errors when reading or
        writing run-related metadata.
    requests.HTTPError
        Propagated from downstream HTTP calls (Wayback or EDGAR) that fail with
        a non-retryable status code.
    BaseException
        Propagated from lower-level helpers (Wayback seeding, EDGAR search, DB
        utilities) on transport failures, retry exhaustion, or unexpected errors.

    Notes
    -----
    - Generates a run_id per invocation and attaches it to logs.
    - Emits milestone INFO logs (start, windows constructed, exclusion size).
    - Invokes `seed_wayback_table(...)` to populate Wayback-derived CIK candidates
      before calling the EDGAR harvester.
    - This function stops after EDGAR evidence collection; scoring/curated loads
      are out of scope.
    - Edgar harvesting depends on wayback candidates being populated for the
      relevant (ticker, validity_window) pairs.
    """

    load_dotenv()
    end_date_str: str | None = os.getenv("END_DATE")
    if end_date_str is None:
        raise ValueError("END_DATE environment variable is required but not set.")
    end_date: pd.Timestamp = str_to_timestamp(end_date_str)
    run_id, exclusion_set = set_up_run()
    if logger_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        logger_level = "INFO"
    logger: InfraLogger = initialize_logger(
        component_name="seed_evidence",
        level=logger_level,
        run_meta={
            "run_id": run_id,
            "excluded_triples_count": len(exclusion_set),
        },
    )
    logger.info("construct_validity_windows_start")
    snp_membership_windows: pd.DataFrame = extract_snp_membership_windows(logger)
    ticker_validity_windows: dict[str, ValidityWindows] = extract_ticker_validity_windows(
        snp_membership_windows, end_date, logger
    )
    logger.info("ticker_validity_windows_constructed")
    if seed_wayback:
        seed_wayback_table(ticker_validity_windows, logger)
    if scrape_edgar:
        collect_evidence(exclusion_set, run_id, logger)


def set_up_run() -> tuple[str, set[tuple[str, str, str]]]:
    """
    Initialize a run and compute the exclusion set of completed and manually
    overridden (ticker × validity_window × candidate_cik) triples.

    Parameters
    ----------
    None

    Returns
    -------
    tuple[str, set[tuple[str, str, str]]]
        (run_id, exclusion_set) where:
        - run_id is a UUID4 string identifying this run.
        - exclusion_set is a set of (ticker, window_key, candidate_cik) triples
          that should be skipped during EDGAR harvesting. window_key is the
          canonical 'YYYY-MM-DD to YYYY-MM-DD' representation of a half-open
          [start, end) validity window produced by `validity_window_to_str(...)`.

    Raises
    ------
    psycopg2.Error
        If querying `edgar_run_registry` or `ticker_cik_manual_adjudication`
        fails, or if a database connection cannot be established.
    BaseException
        Propagated from `connect_to_db(...)` for unexpected driver- or
        environment-level failures.

    Notes
    -----
    - `edgar_run_registry` is assumed to enforce a finite, non-empty
      DATERANGE `validity_window` via its `valid_date_range` constraint
      (no infinite bounds, `lower < upper`, `[start, end)` semantics).
      `ticker_cik_manual_adjudication` enforces the same invariants.
    - The exclusion set is constructed by concatenating:
        * all (ticker, validity_window, candidate_cik) rows from
          `edgar_run_registry`, representing triples that have already been
          harvested and persisted; and
        * all (ticker, validity_window, associated_cik) rows from
          `ticker_cik_manual_adjudication` where `action = 'manual_override'`
          and `associated_cik` matches a 10-digit CIK. These manual overrides
          are interpreted as “do not harvest this candidate CIK for this
          (ticker, validity_window)”.
    - Each DATERANGE is converted to a `window_key` string via
      `validity_window_to_str((start_date, end_date))` before inserting
      (ticker, window_key, candidate_cik) into `exclusion_set`.
    - If a row unexpectedly exposes a `validity_window` with a NULL lower or
      upper bound after fetching from the driver, that row is skipped and
      should be investigated as a potential data-quality issue.
    """

    run_id: str = str(uuid.uuid4())
    exclusion_set: set[tuple[str, str, str]] = set()
    run_registry_query: str = """
    SELECT ticker, validity_window, candidate_cik
    FROM edgar_run_registry
    """
    manual_adjudication_query = """
        SELECT ticker, validity_window, associated_cik
        FROM ticker_cik_manual_adjudication
        WHERE action = 'manual_override'
        AND associated_cik ~ '^[0-9]{10}$'
    """
    with connect_to_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(run_registry_query)
            registry_exclusion_set: List[tuple[str, psycopg2.extras.DateRange, str]] = (
                cursor.fetchall()
            )
            cursor.execute(manual_adjudication_query)
            manual_adjudication_exclusion_set: List[tuple[str, psycopg2.extras.DateRange, str]] = (
                cursor.fetchall()
            )
    raw_exclusion_set: List[tuple[str, psycopg2.extras.DateRange, str]] = (
        registry_exclusion_set + manual_adjudication_exclusion_set
    )
    for ticker_window_candidate in raw_exclusion_set:
        if ticker_window_candidate[1]:
            raw_start_date: dt.date | None = ticker_window_candidate[1].lower
            raw_end_date: dt.date | None = ticker_window_candidate[1].upper
            if raw_start_date and raw_end_date:
                start_date: pd.Timestamp = pd.Timestamp(raw_start_date)
                end_date: pd.Timestamp = pd.Timestamp(raw_end_date)
                window_key: str = validity_window_to_str((start_date, end_date))
                exclusion_set.add(
                    (ticker_window_candidate[0], window_key, ticker_window_candidate[2])
                )

    return run_id, exclusion_set


def extract_ticker_validity_windows(
    snp_membership_windows: pd.DataFrame, end_date: pd.Timestamp, logger: InfraLogger
) -> dict[str, ValidityWindows]:
    """
    Convert daily S&P membership snapshots into per-ticker validity windows.

    Purpose
    -------
    - Normalize and explode daily `tickers`, apply a date-scoped alias rewrite
    (e.g., AABA → YHOO for the relevant period), then group by day to build
    a clean time series of membership sets. From that series, derive half-open
    windows [start, end) per ticker covering its inclusion periods.

    Parameters
    ----------
    snp_membership_windows : pandas.DataFrame
        Columns: `date` (tz-aware UTC day) and `tickers` (comma-separated string).
        Assumes rows are already filtered to the overall run window.
    end_date : pandas.Timestamp
        UTC upper bound for the run; used to close any open windows.
    logger : InfraLogger
        Structured logger for lightweight instrumentation (alias rewrite counts).

    Returns
    -------
    dict[str, ValidityWindows]
        Mapping: ticker → list of (start_utc, end_utc) pairs with [start, end) semantics.

    Notes
    -----
    - Empty or malformed tickers are dropped after normalization.
    - Alias rewriting uses `rewrite_aliased_tickers(...)` against a curated CSV
      of (alias, canonical, [start, end)) and is applied *before* per-day grouping.
    - Open windows at the tail are closed at `end_date + 1 day` to preserve right-open semantics.
    """

    snp_membership_windows["tickers"] = snp_membership_windows["tickers"].str.split(",")
    snp_membership_windows = snp_membership_windows.explode("tickers", ignore_index=True)
    snp_membership_windows["tickers"] = normalize_tickers(snp_membership_windows["tickers"])
    snp_membership_windows["tickers"] = snp_membership_windows["tickers"].replace("", pd.NA)
    snp_membership_windows.dropna(subset=["tickers"], inplace=True)
    snp_membership_windows["tickers"] = rewrite_aliased_tickers(
        snp_membership_windows[["tickers", "date"]].copy(), logger
    )
    snp_membership_windows = (
        snp_membership_windows.groupby("date", sort=False)["tickers"].agg(list).reset_index()
    )
    valid_tickers: ValidTickers = extract_valid_tickers(snp_membership_windows["tickers"])
    logger.info(
        "valid_tickers_extracted",
        context={"valid_ticker_count": len(valid_tickers)},
    )
    return construct_validity_windows(valid_tickers, snp_membership_windows, end_date)


def rewrite_aliased_tickers(tickers_and_dates: pd.DataFrame, logger: InfraLogger) -> pd.Series:
    """
    Rewrite tickers using a date-bounded alias map and return a Series aligned to the input index.

    Parameters
    ----------
    tickers_and_dates : pandas.DataFrame
        Left frame containing at least:
        - 'tickers' : str — raw ticker symbol.
        - 'date'    : pandas.Timestamp (tz-aware, UTC) — snapshot date to test alias validity.
        The function preserves row order/index from this frame.
    logger : InfraLogger
        Structured logger used for a DEBUG summary of applied rewrites.

    Returns
    -------
    pandas.Series
        A Series named 'tickers' with the same index and length as `tickers_and_dates`,
        where each element is either the canonical replacement (when an alias applies for
        that row's date) or the original ticker when no alias is valid.

    Raises
    ------
    FileNotFoundError
        If 'local_data/symbol_aliases.csv' cannot be found.
    ValueError
        If 'start_date'/'end_date' columns in the alias CSV cannot be parsed to timestamps.

    Notes
    -----
    - Alias source: reads 'local_data/symbol_aliases.csv' with required columns:
      'alias' (str), 'canonical' (str), 'start_date' (YYYY-MM-DD or ISO),
      'end_date' (YYYY-MM-DD or ISO).
    - Interval policy: an alias applies when start_date <= date < end_date (half-open on the right).
      Dates are parsed as UTC and compared against a tz-aware UTC 'date'.
    - Non-matches: rows without any valid alias retain the original 'tickers' value.
    - Alignment: the returned Series preserves the original index of `tickers_and_dates`
      (stable assignment back to a column is safe).
    """

    alias_df: pd.DataFrame = pd.read_csv("local_data/symbol_aliases.csv", comment="#")
    alias_df["start_date"] = pd.to_datetime(alias_df["start_date"], utc=True)
    alias_df["end_date"] = pd.to_datetime(alias_df["end_date"], utc=True)
    tickers_and_dates["row_id"] = tickers_and_dates.index.to_series()
    merged: pd.DataFrame = pd.merge(
        tickers_and_dates, right=alias_df, how="left", left_on="tickers", right_on="alias"
    )
    valid_alias_mask: pd.Series = (merged["date"] >= merged["start_date"]) & (
        merged["date"] < merged["end_date"]
    )
    merged["ticker_rewritten"] = np.where(valid_alias_mask, merged["canonical"], merged["tickers"])
    merged["rank"] = valid_alias_mask.astype("int8")
    dedup: pd.DataFrame = (
        merged.sort_values(["row_id", "rank"], ascending=[True, False])
        .drop_duplicates(subset="row_id", keep="first")
        .sort_values("row_id")
    )
    rewritten_rows = int(dedup["rank"].sum())
    if rewritten_rows:
        distinct_aliases = int(merged.loc[valid_alias_mask, "alias"].nunique())
        logger.debug(
            "alias_rewrites_applied",
            context={"rewritten_rows": rewritten_rows, "distinct_aliases": distinct_aliases},
        )
    return dedup["ticker_rewritten"].rename("tickers")


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
    Normalize raw ticker strings to the canonical EDGAR style.

    Transformations
    ---------------
    - Uppercase
    - Trim whitespace
    - Remove internal whitespace
    - Map "." and "/" to "-" (EDGAR-canonical class separator)

    Parameters
    ----------
    tickers : pandas.Series
        Series of raw ticker strings.

    Returns
    -------
    pandas.Series
        Cleaned tickers; invalid entries may become empty strings and are dropped upstream.

    Notes
    -----
    - This normalization is kept consistent with DB constraints and with the SEC EDGAR
      company feed inputs.
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


def seed_wayback_table(
    ticker_validity_windows: dict[str, ValidityWindows], logger: InfraLogger
) -> None:
    """
    Seed Wayback-derived CIK candidates for all ticker validity windows.

    Parameters
    ----------
    ticker_validity_windows : dict[str, ValidityWindows]
        Mapping from ticker symbol to a list of half-open [start_utc, end_utc)
        validity windows over which it appears in the S&P 500 index.
    logger : InfraLogger
        Structured logger used for high-level telemetry of the Wayback seeding run.

    Returns
    -------
    None
        Dispatches the (ticker, validity_window) batch to the Wayback orchestrator
        and persists discovered candidates into the wayback_candidates table.

    Raises
    ------
    requests.HTTPError
        Propagated from `batch_extract_candidates_wayback(...)` if CDX or snapshot
        requests fail with a non-retryable status code.
    psycopg2.Error
        Propagated from `load_wayback_candidates(...)` if inserts into the
        wayback_candidates table fail.
    BaseException
        Propagated from lower-level helpers on transport, retry, or DB failures.

    Notes
    -----
    - Flattens the per-ticker window map into a Batch of (ticker, validity_window)
      pairs before dispatching to the Wayback orchestrator.
    - Intended to be called once per run by `run_edgar_harvest()` after validity
      windows are constructed from S&P membership snapshots.
    """

    ticker_window_tuples: Batch = [
        (ticker, window)
        for ticker, windows in ticker_validity_windows.items()
        for window in windows
    ]
    with requests.Session() as session:
        candidate_list: List[WayBackCandidate] = batch_extract_candidates_wayback(
            ticker_window_tuples, logger, session
        )
    load_wayback_candidates(candidate_list)
