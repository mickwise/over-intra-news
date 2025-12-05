"""
Purpose
-------
Fetch a padded slice of the NYSE trading calendar from Postgres and derive
per-trading-day overnight fractions for use in CC-NEWS sampling. After
computing fractions for the padded window, the module returns only the
requested month’s trading days, and also exposes a helper for finding the
next trading day from a UTC timestamp.

Key behaviors
-------------
- Computes month `[start, end)` bounds with configurable left/right padding and
  special-case handling at the sampling horizon edges.
- Issues a single SQL query to load the padded calendar window (including
  non-trading days for context but dropping them before return).
- Converts session open/close times to tz-aware UTC timestamps.
- Computes each trading day’s overnight fraction from the previous trading
  close to the current trading open, automatically including intervening
  weekend/holiday time in the next trading day’s overnight period.
- Drops padded and non-trading rows so only the core month’s trading days are
  returned.
- Maintains a shared XNYS exchange calendar object and provides
  `extract_next_trading_day(...)` to map a UTC timestamp to the next NYSE
  trading day key.

Conventions
-----------
- Calendar index is the New York trading date (`DatetimeIndex` at midnight).
- `session_open_utc` and `session_close_utc` are tz-aware UTC `Timestamp`s.
- `is_trading_day` is retained for diagnostics but all returned rows from
  `extract_nyse_cal` have `True` values.
- `extract_next_trading_day` expects a tz-aware UTC `Timestamp` and returns
  a New York civil date string in `DATE_FMT`, or `None` once
  `LAST_SAMPLING_DAY` is reached.
- All arithmetic is vectorized; no per-row Python loops are used.

Downstream usage
----------------
- Call `extract_nyse_cal(year, month)` to obtain a month slice indexed by
  trading day with `session_*` columns, `is_trading_day`, and
  `overnight_fraction` populated for each trading day. Feed this DataFrame
  into quota and sampling components.

- Use `extract_next_trading_day(utc_ts)` when routing after-close items to
  the correct next trading day (or into spillover once the current month’s
  calendar slice ends).
"""

import datetime as dt
from typing import Any, cast

import exchange_calendars as xcals
import pandas as pd

from aws.ccnews_sampler.ccnews_sampler_config import (
    DATE_FMT,
    DATE_TZ,
    LAST_SAMPLING_DAY,
    LAST_TRADING_DAY_PRE_SAMPLE,
    PAD_DAYS,
)
from infra.utils.db_utils import connect_to_db

XNYS_CAL = xcals.get_calendar("XNYS")


def month_bounds(year: int, month: int) -> tuple[dt.date, dt.date]:
    """
    Compute a month’s `[start, end)` date bounds with configurable padding and edge cases.

    Parameters
    ----------
    year : int
        Four-digit year (e.g., 2024).
    month : int
        Month number in the range 1–12.

    Returns
    -------
    tuple[datetime.date, datetime.date]
        A `(start_inclusive, end_exclusive)` pair suitable for SQL `[start, end)` range
        predicates, including left/right padding where allowed.

    Raises
    ------
    ValueError
        If `year` or `month` cannot be used to construct a valid `datetime.date`.

    Notes
    -----
    - Left padding by `PAD_DAYS` is applied for all months except August 2016
      (the left edge of the data horizon).
    - Right padding by `PAD_DAYS` is applied for all months except July 2025
      and August 2025 (the right edge of the data horizon).
    - Padding is expressed in whole civil days; callers are responsible for any
      additional clipping against global data limits.
    """

    start = dt.date(year, month, 1)
    end = dt.date(year + (month == 12), (month % 12) + 1, 1)
    if year != 2016 or month != 8:
        start -= PAD_DAYS

    if year != 2025 or month not in {7, 8}:
        end += PAD_DAYS
    return start, end


def build_calendar_query() -> str:
    """
    Return the SQL query used to fetch a padded trading calendar window.

    Parameters
    ----------
    None

    Returns
    -------
    str
        Parameterized SQL selecting `trading_day`, `session_open_utc`,
        `session_close_utc`, and `is_trading_day` for a `[start, end)` date range,
        ordered by `trading_day`.

    Raises
    ------
    None

    Notes
    -----
    - The query expects two bound parameters: `start_date` and `end_date`, both
      `DATE`-like values suitable for the `trading_day` column.
    - Session-time columns are returned in UTC; non-trading days have
      NULL session times.
    - Callers are responsible for binding parameters and executing the query
      against Postgres.
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
    Load a padded NYSE calendar window from Postgres, compute overnight fractions,
    and return only the core request month.

    Parameters
    ----------
    year : str
        Four-digit calendar year as a string (e.g., "2024"); converted to `int`
        internally.
    month : str
        Two-digit month string (e.g., "09"); converted to `int` internally.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by `trading_day` (New York civil dates) for the requested
        month, with at least the following columns:
        - `session_open_utc` : tz-aware UTC `Timestamp`
        - `session_close_utc` : tz-aware UTC `Timestamp`
        - `is_trading_day` : bool
        - `overnight_fraction` : float in [0, 1] defined for every civil day.

    Raises
    ------
    Exception
        Any database or IO error raised while acquiring the connection or executing
        the SQL query is propagated to the caller.

    Notes
    -----
    - Performs exactly one SQL query over a padded `[start, end)` window based on
      `month_bounds(...)`.
    - Ensures session columns are converted to tz-aware UTC timestamps before
      overnight computation.
    - Delegates overnight computation and pad dropping to
      `calculate_overnight_fraction(...)`; the returned frame is a view of the same
      object after in-place mutation and row filtering.
    """

    with connect_to_db() as conn:
        query: str = build_calendar_query()
        year_int: int = int(year)
        month_int: int = int(month)
        time_bounds: tuple[dt.date, dt.date] = month_bounds(year_int, month_int)

        calendar: pd.DataFrame = pd.read_sql(
            query,
            cast(Any, conn),
            index_col="trading_day",
            params=time_bounds,
            parse_dates=["trading_day"],
        )
        calendar.index = pd.DatetimeIndex(calendar.index, name="trading_day")
    calendar.sort_index(inplace=True)
    calendar["session_open_utc"] = pd.to_datetime(calendar["session_open_utc"], utc=True)
    calendar["session_close_utc"] = pd.to_datetime(calendar["session_close_utc"], utc=True)
    return calculate_overnight_fraction(calendar, year_int, month_int)


def calculate_overnight_fraction(calendar: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """
    Compute overnight fractions for a padded calendar window and return only the
    core month’s trading days.

    Parameters
    ----------
    calendar : pandas.DataFrame
        Padded calendar window indexed by civil dates, containing
        `session_open_utc`, `session_close_utc`, and `is_trading_day` columns.
    year : int
        Four-digit year of the requested month.
    month : int
        Month number (1–12) of the requested month.

    Returns
    -------
    pandas.DataFrame
        The same `calendar` object, mutated in place and restricted to trading-day
        rows within the requested `(year, month)`, with an `overnight_fraction`
        column populated for each trading day.

    Notes
    -----
    - Uses `calculate_overnight_fraction_trading_days(...)` to compute fractions
      only for trading days and assigns them by label alignment.
    - Weekend and holiday time is implicitly included in the next trading day’s
      `overnight_fraction`.
    - Rows outside the core `(year, month)` are dropped after filtering.
    """

    overnight_frac: pd.Series = calculate_overnight_fraction_trading_days(calendar, year, month)
    calendar.loc[overnight_frac.index, "overnight_fraction"] = overnight_frac
    dt_index = cast(pd.DatetimeIndex, calendar.index)
    core_calendar_mask = (
        (dt_index.year == year) & (dt_index.month == month) & (calendar["is_trading_day"])
    )
    indices_to_drop: pd.Index = calendar.index.difference(calendar[core_calendar_mask].index)
    calendar.drop(indices_to_drop, axis=0, inplace=True)
    return calendar


def calculate_overnight_fraction_trading_days(
    calendar: pd.DataFrame, year: int, month: int
) -> pd.Series:
    """
    Compute overnight fractions for trading days within a padded calendar window.

    Parameters
    ----------
    calendar : pandas.DataFrame
        Padded calendar window containing at least `is_trading_day`,
        `session_open_utc`, and `session_close_utc` columns.
    year : int
        Four-digit year of the requested month (used for horizon edge handling).
    month : int
        Month number (1–12) of the requested month (used for horizon edge handling).

    Returns
    -------
    pandas.Series
        Series of `overnight_fraction` values indexed by trading days only
        (subset of `calendar.index`) within the padded window.

    Raises
    ------
    KeyError
        If the required columns are missing from `calendar`.
    ValueError
        If the special-case horizon injection (e.g., Aug 2016) cannot be satisfied
        due to missing historical schedule data.

    Notes
    -----
    - Overnight minutes are measured from each trading day’s previous trading
      close to its own session open, so weekend and holiday periods are naturally
      included in the next trading day’s overnight duration.
    - The returned Series covers trading days only; non-trading dates are omitted.
    - The Series is designed for direct assignment into
      `calendar.loc[series.index, "overnight_fraction"]`.
    """

    trading_days: pd.DataFrame = calendar.loc[calendar["is_trading_day"]].copy()
    trading_days["intraday_minutes"] = cast(
        pd.Series,
        (
            trading_days["session_close_utc"] - trading_days["session_open_utc"]
        ).dt.total_seconds()  # type: ignore
        / 60,
    ).clip(0)
    trading_days["previous_close_utc"] = trading_days["session_close_utc"].shift(1)

    # Handle the special case of Aug 1, 2016, which is the first sampling day.
    if year == 2016 and month == 8:
        last_trading_close: pd.Timestamp = XNYS_CAL.schedule.loc[LAST_TRADING_DAY_PRE_SAMPLE][
            "close"
        ]
        trading_days.loc["2016-08-01", "previous_close_utc"] = last_trading_close

    trading_days["overnight_minutes"] = (
        (
            trading_days["session_open_utc"] - trading_days["previous_close_utc"]
        ).dt.total_seconds()  # type: ignore
        / 60
    ).clip(0)
    trading_days["overnight_fraction"] = trading_days["overnight_minutes"] / (
        trading_days["intraday_minutes"] + trading_days["overnight_minutes"]
    )
    return trading_days["overnight_fraction"]


def extract_next_trading_day(
    current_ts: pd.Timestamp,
) -> str | None:
    """
    Compute the next NYSE trading-day key after a given UTC timestamp.

    Parameters
    ----------
    current_ts : pandas.Timestamp
        Tz-aware UTC timestamp (typically the CC-NEWS link time) from which
        to compute the next NYSE trading day.

    Returns
    -------
    str or None
        The next trading-day key ('YYYY-MM-DD') in `DATE_TZ` if the New York
        civil date of `current_ts` is strictly before `LAST_SAMPLING_DAY`;
        otherwise `None` once the sampling horizon has been reached or passed.

    Raises
    ------
    None

    Notes
    -----
    - Uses a shared XNYS exchange calendar instance to find the next open
      after `current_ts`, then converts that open instant into a New York
      civil date string using `DATE_TZ` and `DATE_FMT`.
    - Callers are expected to handle the `None` case by dropping the item
      or logging it as beyond the sampling horizon.
    """

    ny_date = current_ts.tz_convert(DATE_TZ).date()
    if ny_date >= LAST_SAMPLING_DAY:
        return None
    next_open = XNYS_CAL.next_open(current_ts)
    next_date_key = next_open.tz_convert(DATE_TZ).strftime(DATE_FMT)
    return next_date_key
