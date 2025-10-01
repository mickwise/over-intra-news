"""
Purpose
-------
Load a commit-pinned daily snapshot of S&P 500 constituents from GitHub,
constrain it to a configured date window, and return the filtered snapshots
for downstream spell construction.

Key behaviors
-------------
- Fetches a CSV from a specific Git commit (provenance and reproducibility).
- Normalizes the `date` column to timezone-aware UTC Timestamps.
- Filters rows to the inclusive `[START_DATE, END_DATE]` window.

Conventions
-----------
- `START_DATE` and `END_DATE` are ISO strings in environment variables and are parsed to UTC.
- Input CSV is expected to have columns: `date` (YYYY-MM-DD), `tickers` (comma-separated tickers).
- Data source is the open-source GitHub repository `fja05680/sp500`,
  pinned to commit `0803e40971b4e470fd3b3bef107b3c6bae579cfc` for reproducibility.

Downstream usage
----------------
Call `extract_snp_membership_windows()` to obtain daily constituents
within the configured window, then convert snapshots to per-ticker membership
spells in a subsequent step.
"""

import os

import pandas as pd
from dotenv import load_dotenv

from infra.utils.db_utils import str_to_timestamp

FJA05680_SHA: str = "0803e40971b4e470fd3b3bef107b3c6bae579cfc"
SNP_HISTORICAL_URL: str = (
    "https://raw.githubusercontent.com/fja05680/sp500/0803e40971b"
    "4e470fd3b3bef107b3c6bae579cfc/S%26P%20500%20Historical%20Components%20%"
    "26%20Changes(07-12-2025).csv"
)


def extract_snp_membership_windows() -> pd.DataFrame:
    """
    Return daily S&P 500 constituent snapshots filtered to the configured date window.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.DataFrame
        A DataFrame with at least `date` (UTC Timestamp) and `tickers` (comma-separated str),
        filtered to `START_DATE <= date <= END_DATE`. Index is reset and monotonic.

    Raises
    ------
    KeyError
        If `START_DATE` or `END_DATE` is not found in the environment.
    ValueError
        If the date strings cannot be parsed to timestamps.

    Notes
    -----
    - `START_DATE` and `END_DATE` are loaded via `dotenv` and parsed with `str_to_timestamp`.
    - This function delegates the actual CSV fetch and filtering
    to `extract_historical_constituents`.
    """
    load_dotenv()
    start_date: pd.Timestamp = str_to_timestamp(os.environ["START_DATE"])
    end_date: pd.Timestamp = str_to_timestamp(os.environ["END_DATE"])
    return extract_historical_constituents(start_date, end_date)


def extract_historical_constituents(
    start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Load, normalize, and window-filter daily S&P 500 constituent snapshots from a commit-pinned CSV.

    Parameters
    ----------
    start_date : pandas.Timestamp
        Inclusive lower bound (UTC) for filtering the `date` column.
    end_date : pandas.Timestamp
        Inclusive upper bound (UTC) for filtering the `date` column.

    Returns
    -------
    pandas.DataFrame
        The subset of rows whose `date` falls within `[start_date, end_date]`, with
        `date` normalized to UTC Timestamps and the index reset.

    Raises
    ------
    FileNotFoundError
        If the remote CSV cannot be fetched.
    ValueError
        If the CSV lacks a `date` column or the dates cannot be parsed with the expected format.

    Notes
    -----
    - The CSV is fetched from a raw GitHub URL pinned to a specific commit for reproducibility.
    - `date` is parsed using the explicit format `%Y-%m-%d` and
    set to UTC to avoid tz-naive comparisons.
    - This function does not modify or validate the `tickers` payload beyond filtering by date.
    """
    raw_historical_constituents: pd.DataFrame = pd.read_csv(
        SNP_HISTORICAL_URL,
    )
    raw_historical_constituents["date"] = pd.to_datetime(
        raw_historical_constituents["date"], format="%Y-%m-%d", utc=True
    )
    return raw_historical_constituents[
        (raw_historical_constituents["date"] >= start_date)
        & (raw_historical_constituents["date"] <= end_date)
    ].reset_index(drop=True)
