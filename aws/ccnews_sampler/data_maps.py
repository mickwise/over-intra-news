"""
Purpose
-------
Materialize fast lookup maps from the monthly NYSE calendar for WARC sampling.
Converts per-day caps and session boundaries into constant-time (in expectation) dictionaries.

Key behaviors
-------------
- Builds a set of valid yyyy-mm-dd date keys for quick membership checks.
- Builds a cap dictionary: date -> (intraday_cap, overnight_cap).
- Builds a session dictionary: date -> (session_open_seconds, session_close_seconds), using UTC
  epoch seconds and `None` for non-trading days.

Conventions
-----------
- Keys are New York civil dates formatted as '%Y-%m-%d'.
- Session times are represented as UTC epoch seconds (ints) or None.
- Input calendar is indexed by trading_day and contains *_utc columns.

Downstream usage
----------------
Use `build_data_maps(nyse_cal)` once per run, then read from the returned
`DataMaps` object inside the WARC scan loop for E[O(1)] lookup.
"""

from dataclasses import dataclass
from typing import Iterable, TypeAlias, cast

import numpy as np
import pandas as pd

SessionCaps: TypeAlias = tuple[int, int]
SessionTimes: TypeAlias = tuple[int | None, int | None]

DATE_FMT = "%Y-%m-%d"


@dataclass
class DataMaps:
    """
    Precomputed lookup tables for one month of sampling.

    Purpose
    -------
    Provide O(1)-expected-time access to per-day caps and session boundaries
    during WARC scanning, avoiding repeated DataFrame indexing.

    Key behaviors
    -------------
    - Maps each NY date string (YYYY-MM-DD) to (intraday_cap, overnight_cap).
    - Maps each NY date string to (session_open_seconds, session_close_seconds),
      using seconds since Unix epoch; missing sessions use None.
    - Exposes a set of valid dates for quick membership checks.

    Conventions
    -----------
    - Keys are New York civil dates formatted as "%Y-%m-%d".
    - Session times are epoch seconds (UTC-based) or None on non-trading days.

    Downstream usage
    ----------------
    - Use `cap_dict[date_key]` to fetch integer caps.
    - Use `session_dict[date_key]` to fetch session boundary seconds.
    - Use `valid_date_set` to validate parsed dates before any lookup.

    Attributes
    ----------
    cap_dict : dict[str, tuple[int, int]]
        Intraday and overnight sampling caps per date.
    session_dict : dict[str, tuple[int | None, int | None]]
        Session open/close epoch seconds per date (None for non-trading days).
    valid_date_set : set[str]
        All date keys present in the calendar slice (for fast membership tests).
    """

    cap_dict: dict[str, SessionCaps]
    session_dict: dict[str, SessionTimes]
    valid_date_set: set[str]


def build_data_maps(nyse_cal: pd.DataFrame) -> DataMaps:
    """
    Construct constant-time lookup structures from a monthly NYSE calendar.

    Parameters
    ----------
    nyse_cal : pd.DataFrame
        Calendar indexed by trading_day with columns including
        'intraday_cap', 'overnight_cap', 'session_open_utc', 'session_close_utc'.

    Returns
    -------
    DataMaps
        Container with:
        - cap_dict: dict[str, tuple[int, int]]
        - session_dict: dict[str, tuple[int | None, int | None]]
        - valid_date_set: set[str]

    Raises
    ------
    KeyError
        If required columns are missing from `nyse_cal`.

    Notes
    -----
    - Date keys are formatted as '%Y-%m-%d' (New York civil dates).
    - Session epoch seconds are UTC-based; non-trading days map to (None, None).
    """
    dt_index: pd.DatetimeIndex = cast(pd.DatetimeIndex, nyse_cal.index)
    str_dates: pd.Series = pd.Series(dt_index.strftime(DATE_FMT))
    valid_date_set: set[str] = set(str_dates)
    cap_dict: dict[str, SessionCaps] = build_cap_dict(
        str_dates, nyse_cal[["intraday_cap", "overnight_cap"]]
    )
    session_dict: dict[str, SessionTimes] = build_session_dict(str_dates, nyse_cal)
    return DataMaps(cap_dict=cap_dict, session_dict=session_dict, valid_date_set=valid_date_set)


def build_cap_dict(str_dates: pd.Series, caps: pd.DataFrame) -> dict[str, SessionCaps]:
    """
    Zip date strings with computed daily caps into a lookup dictionary.

    Parameters
    ----------
    str_dates : pd.Series
        Series of '%Y-%m-%d' strings aligned with `caps`.
    caps : pd.DataFrame
        Two-column frame with ['intraday_cap', 'overnight_cap'].

    Returns
    -------
    dict[str, tuple[int, int]]
        Mapping: 'YYYY-MM-DD' -> (intraday_cap, overnight_cap).

    Raises
    ------
    None

    Notes
    -----
    - Caps are cast to built-in `int` to ensure JSON-serializable values.
    """
    caps_np: np.ndarray = caps.to_numpy()
    cap_zip: Iterable[tuple[str, np.ndarray]] = zip(str_dates, caps_np)
    return {date: (int(intra), int(overnight)) for date, (intra, overnight) in cap_zip}


def build_session_dict(str_dates: pd.Series, nyse_cal: pd.DataFrame) -> dict[str, SessionTimes]:
    """
    Convert session open/close timestamps into UTC epoch seconds per date.

    Parameters
    ----------
    nyse_cal : pd.DataFrame
        Calendar with 'session_open_utc' and 'session_close_utc' as tz-aware
        timestamps (UTC) and index of trading days.

    Returns
    -------
    dict[str, tuple[int | None, int | None]]
        Mapping: 'YYYY-MM-DD' -> (open_epoch_s, close_epoch_s) where each
        element is an int or None for non-trading days.

    Raises
    ------
    KeyError
        If required session columns are missing.

    Notes
    -----
    - Missing timestamps (NaT) are mapped to None.
    - Epoch seconds are computed in UTC to avoid DST ambiguity.
    """
    non_trading_days_mask: pd.Series = ~nyse_cal["is_trading_day"]
    non_trading_days_set = set(str_dates[non_trading_days_mask])
    session_open_seconds: pd.Series = to_seconds(nyse_cal["session_open_utc"])
    session_close_seconds: pd.Series = to_seconds(nyse_cal["session_close_utc"])
    session_times: Iterable[tuple[str, tuple[np.int64, np.int64]]] = zip(
        str_dates, zip(session_open_seconds, session_close_seconds)
    )
    return {
        date: (int(open_sec), int(close_sec)) if date not in non_trading_days_set else (None, None)
        for date, (open_sec, close_sec) in session_times
    }


def to_seconds(ts: pd.Series) -> pd.Series:
    """
    Convert a tz-aware datetime Series to UTC epoch seconds (vectorized).

    Parameters
    ----------
    ts : pd.Series
        Pandas Series of timezone-aware timestamps (dtype like
        'datetime64[ns, tz]'). May contain NaT.

    Returns
    -------
    pd.Series
        Int64 Series of seconds since Unix epoch (UTC).

    Raises
    ------
    TypeError
        If `ts` is not datetime-like or is tz-naive (use tz_localize first).

    Notes
    -----
    - Implementation is fully vectorized: tz_convert → view('int64') → // 1e9.
    - This helper does not alter the input and avoids Python-level loops.
    - NaT values propagate as <NA> in the returned Int64 Series.
    """
    return ts.dt.tz_convert("UTC").view("int64") // 1_000_000_000


def to_seconds_int(ts: pd.Timestamp) -> int:
    """
    Convert a single tz-aware pandas Timestamp to UTC epoch seconds.

    Parameters
    ----------
    ts : pd.Timestamp
        Pandas Timestamp object that is timezone-aware. Must not be tz-naive.

    Returns
    -------
    int
        Seconds since Unix epoch (UTC).

    Raises
    ------
    TypeError
        If `ts` is not datetime-like or is tz-naive.

    Notes
    -----
    - This is the scalar equivalent of `to_seconds` for working with a single
      Timestamp instead of a Series.
    - Uses the `.tz_convert("UTC")` accessor and integer nanoseconds (`.value`)
      divided by 1e9 to yield whole seconds.
    """
    return int(ts.tz_convert("UTC")) // 1_000_000_000
