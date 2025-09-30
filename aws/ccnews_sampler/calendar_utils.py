"""
Purpose
-------
Fetch a month slice of the NYSE trading calendar from Postgres and provide basic
date utilities for building time windows.

Key behaviors
-------------
- Computes [start, end) month bounds for querying.
- Issues a single SQL query to load the month’s calendar with ordered rows.
- Returns a DataFrame indexed by `trading_day` (datetime-like).
- Ensures session times are UTC (tz-aware) for downstream comparisons.

Conventions
-----------
- `trading_day` is the New York civil date (midnight-based Timestamp index).
- `session_open_utc` and `session_close_utc` are UTC instants.
- Query uses `[start, end)` bounds and `ORDER BY trading_day`.

Downstream usage
----------------
- Call `extract_nyse_cal(year, month)` to obtain the month’s calendar, then pass
  it to quota computation and WARC scanning modules.
- Returned calendar has a `DatetimeIndex` at runtime; static type checkers
  may still require `cast(pd.DatetimeIndex, calendar.index)` downstream.

"""

import datetime as dt

import pandas as pd
from psycopg2.extensions import connection

from ingest.db_utils import connect_to_db


def month_bounds(year: int, month: int) -> tuple[dt.date, dt.date]:
    """
    Compute inclusive start and exclusive end dates for the given month.

    Parameters
    ----------
    year : int
        Four-digit year (e.g., 2024).
    month : int
        Month number 1–12.

    Returns
    -------
    tuple[date, date]
        (start_inclusive, end_exclusive) for use in SQL range predicates.

    Notes
    -----
    - End is the first day of the following month, handling December rollover.
    """
    start = dt.date(year, month, 1)
    end = dt.date(year + (month == 12), (month % 12) + 1, 1)
    return start, end


def build_calendar_query() -> str:
    """
    Return the SQL used to fetch the trading calendar slice.

    Returns
    -------
    str
        Parameterized SQL with placeholders for [start, end).

    Notes
    -----
    - Expects two parameters: start_date, end_date.
    - Orders by `trading_day` for deterministic indexing.
    """
    return """
        SELECT
            trading_day,
            session_open_utc,
            session_close_utc,
            is_trading_day
        FROM trading_calendar
        WHERE trading_day >= %s
          AND trading_day <  %s
        ORDER BY trading_day;
    """


def extract_nyse_cal(year: str, month: str) -> pd.DataFrame:
    """
    Load the NYSE calendar for a given year-month window from Postgres.

    Parameters
    ----------
    year : str
        Year token (e.g., "2024"); will be cast to int.
    month : str
        Month token (e.g., "09"); will be cast to int.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by `trading_day` (datetime-like), with columns:
        - session_open_utc (tz-aware UTC)
        - session_close_utc (tz-aware UTC)
        - is_trading_day (bool)
        - is_half_day (bool)

    Raises
    ------
    Exception
        Propagates database/IO errors from `read_sql`.

    Notes
    -----
    - Uses `parse_dates=["trading_day"]` so the index is datetime-like.
    - The index is enforced as a `DatetimeIndex` named "trading_day".
      At runtime this avoids further wrapping; with MyPy, explicit `cast`
      may still be needed to satisfy type inference.

    """
    conn: connection = connect_to_db()
    try:
        query: str = build_calendar_query()
        time_bounds = month_bounds(int(year), int(month))
        calendar: pd.DataFrame = pd.read_sql(
            query, conn, index_col="trading_day", params=time_bounds, parse_dates=["trading_day"]
        )
        calendar.index = pd.DatetimeIndex(calendar.index, name="trading_day")
    finally:
        conn.close()
    calendar.sort_index(inplace=True)
    calendar["session_open_utc"] = pd.to_datetime(calendar["session_open_utc"], utc=True)
    calendar["session_close_utc"] = pd.to_datetime(calendar["session_close_utc"], utc=True)
    return calendar
