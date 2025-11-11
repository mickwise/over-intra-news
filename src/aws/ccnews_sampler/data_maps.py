"""
Purpose
-------
Build fast, dictionary-based lookup structures from a monthly NYSE trading
calendar for use in WARC sampling. Converts per-day caps and session boundaries
into simple maps keyed by New York trading-date strings.

Key behaviors
-------------
- Normalizes the calendar index into '%Y-%m-%d' date keys.
- Builds a cap dictionary mapping each trading day to (intraday_cap, overnight_cap).
- Builds a session dictionary mapping each trading day to
  (session_open_epoch_s, session_close_epoch_s) using UTC epoch seconds.
- Carries through the per-day `overnight_fraction` values for reuse downstream.

Conventions
-----------
- Keys are New York trading dates formatted as '%Y-%m-%d'.
- Session times are represented as UTC epoch seconds (ints).
- Input calendar is indexed by `trading_day` and includes *_utc session columns.

Downstream usage
----------------
Call `build_data_maps(nyse_cal)` once per run, and pass the resulting `DataMaps`
instance into the sampling pipeline. Downstream code should rely on dictionary
lookups (not DataFrame access) inside tight loops for per-line WARC processing.
"""

from dataclasses import dataclass
from typing import Iterable, cast

import numpy as np
import pandas as pd

from aws.ccnews_sampler.ccnews_sampler_config import DATE_FMT
from aws.ccnews_sampler.ccnews_sampler_types import SessionCaps, SessionTimes


@dataclass
class DataMaps:
    """
    Purpose
    -------
    Bundle precomputed per-day lookup tables derived from the NYSE calendar for a
    single month of sampling.

    Key behaviors
    -------------
    - Exposes mapping from NY date string to (intraday_cap, overnight_cap).
    - Exposes mapping from NY date string to (session_open_epoch_s, session_close_epoch_s),
      using UTC epoch seconds with `None` for non-trading days.
    - Preserves per-day `overnight_fraction` values for reuse in downstream logic.
    - Provides a fast membership set for validating calendar-aligned dates.

    Parameters
    ----------
    cap_dict : dict[str, SessionCaps]
        Maps 'YYYY-MM-DD' date keys to (intraday_cap, overnight_cap) integer pairs.
    session_dict : dict[str, SessionTimes]
        Maps 'YYYY-MM-DD' date keys to (session_open_epoch_s, session_close_epoch_s),
        with values in UTC epoch seconds or (None, None) on non-trading days.
    overnight_fraction_dict : dict[str, float]
        Maps 'YYYY-MM-DD' date keys to their overnight_fraction ∈ [0.0, 1.0].
    valid_date_set : set[str]
        Set of all valid 'YYYY-MM-DD' keys represented in the calendar slice.

    Attributes
    ----------
    cap_dict : dict[str, SessionCaps]
        Integer intraday/overnight caps per New York civil date.
    session_dict : dict[str, SessionTimes]
        Session open/close times in UTC epoch seconds or (None, None) for non-trading days.
    overnight_fraction_dict : dict[str, float]
        Per-day overnight fraction carried through from the calendar.
    valid_date_set : set[str]
        Membership set of all supported date keys for fast validation.

    Notes
    -----
    - Intended to be constructed once per month and reused across tight sampling loops.
    - All lookups are pure dictionary access; the underlying DataFrame is not needed
      once `DataMaps` is built.
    """

    cap_dict: dict[str, SessionCaps]
    session_dict: dict[str, SessionTimes]
    overnight_fraction_dict: dict[str, float]
    valid_date_set: set[str]


def build_data_maps(nyse_cal: pd.DataFrame) -> DataMaps:
    """
    Construct a `DataMaps` instance from a monthly NYSE calendar DataFrame.

    Parameters
    ----------
    nyse_cal : pandas.DataFrame
        Calendar indexed by `trading_day` with at least the columns:
        - 'intraday_cap' : int
        - 'overnight_cap' : int
        - 'session_open_utc' : datetime64[ns, UTC]
        - 'session_close_utc' : datetime64[ns, UTC]
        - 'is_trading_day' : bool
        - 'overnight_fraction' : float in [0.0, 1.0]

    Returns
    -------
    DataMaps
        Container with:
        - cap_dict : per-day (intraday_cap, overnight_cap)
        - session_dict : per-day (session_open_epoch_s, session_close_epoch_s)
        - overnight_fraction_dict : per-day overnight_fraction
        - valid_date_set : set of supported 'YYYY-MM-DD' keys

    Raises
    ------
    KeyError
        If one or more required columns are missing from `nyse_cal`.

    Notes
    -----
    - New York civil date keys are derived from the `trading_day` index using
      the shared DATE_FMT configuration.
    - This function is expected to be called once per run; downstream code
      should work exclusively with the resulting dictionaries for performance.
    """

    dt_index: pd.DatetimeIndex = cast(pd.DatetimeIndex, nyse_cal.index)
    str_dates: pd.Series = pd.Series(dt_index.strftime(DATE_FMT))
    valid_date_set: set[str] = set(str_dates)
    cap_dict: dict[str, SessionCaps] = build_cap_dict(
        str_dates, nyse_cal[["intraday_cap", "overnight_cap"]]
    )
    session_dict: dict[str, SessionTimes] = build_session_dict(str_dates, nyse_cal)
    overnight_series: pd.Series = nyse_cal["overnight_fraction"]
    overnight_fraction_dict: dict[str, float] = {
        date: float(cast(float, overnight_series.iloc[i])) for i, date in enumerate(str_dates)
    }
    return DataMaps(
        cap_dict=cap_dict,
        session_dict=session_dict,
        valid_date_set=valid_date_set,
        overnight_fraction_dict=overnight_fraction_dict,
    )


def build_cap_dict(str_dates: pd.Series, caps: pd.DataFrame) -> dict[str, SessionCaps]:
    """
    Build a dictionary of per-day intraday and overnight caps keyed by date string.

    Parameters
    ----------
    str_dates : pandas.Series
        Series of '%Y-%m-%d' date strings aligned row-wise with `caps`.
    caps : pandas.DataFrame
        Two-column frame with:
        - 'intraday_cap' : int-like
        - 'overnight_cap' : int-like

    Returns
    -------
    dict[str, SessionCaps]
        Mapping 'YYYY-MM-DD' -> (intraday_cap, overnight_cap) as built-in ints.

    Raises
    ------
    None

    Notes
    -----
    - Caps are explicitly cast to Python `int` to avoid numpy scalar
      leakage outside this module.
    - The function assumes `str_dates` and `caps` are positionally aligned.
    """

    caps_np: np.ndarray = caps.to_numpy()
    cap_zip: Iterable[tuple[str, np.ndarray]] = zip(str_dates, caps_np)
    return {date: (int(intra), int(overnight)) for date, (intra, overnight) in cap_zip}


def build_session_dict(str_dates: pd.Series, nyse_cal: pd.DataFrame) -> dict[str, SessionTimes]:
    """
    Convert per-day session open/close timestamps into UTC epoch-second dictionaries.

    Parameters
    ----------
    str_dates : pandas.Series
        Series of '%Y-%m-%d' date strings, positionally aligned with the rows
        of `nyse_cal`. Each element represents the New York civil date for the
        corresponding calendar row.
    nyse_cal : pandas.DataFrame
        Calendar DataFrame with at least the following columns:
        - 'session_open_utc' : datetime64[ns, UTC] or NaT
            Session open timestamp for the trading day in UTC.
        - 'session_close_utc' : datetime64[ns, UTC] or NaT
            Session close timestamp for the trading day in UTC.
        - 'is_trading_day' : bool
            Flag indicating whether the row corresponds to a trading day
            (True) or a non-trading day / holiday (False).

    Returns
    -------
    dict[str, SessionTimes]
        Dictionary mapping 'YYYY-MM-DD' date keys to per-day session times:
            - For trading days: (open_epoch_s, close_epoch_s) as Python ints
              representing seconds since Unix epoch in UTC.
            - For non-trading days: (None, None), regardless of any values
              present in the session timestamp columns.

    Raises
    ------
    KeyError
        If any of the required columns ('session_open_utc',
        'session_close_utc', 'is_trading_day') are missing from `nyse_cal`.
    TypeError
        If the session timestamp columns are not timezone-aware datetimes
        convertible to UTC.

    Notes
    -----
    - This helper assumes that `str_dates`, `session_open_utc`,
      `session_close_utc`, and `is_trading_day` are all aligned by position
      (same length, same row order).
    - Non-trading days are detected solely via `is_trading_day == False` and
      encoded as (None, None), even if non-null timestamps are present.
    """

    session_open_seconds: pd.Series = to_seconds(nyse_cal["session_open_utc"])
    session_close_seconds: pd.Series = to_seconds(nyse_cal["session_close_utc"])
    is_trading: np.ndarray = nyse_cal["is_trading_day"].to_numpy()

    session_iter: Iterable[tuple[str, int, int, bool]] = zip(
        str_dates,
        session_open_seconds,
        session_close_seconds,
        is_trading,
    )

    return {
        date: (int(open_sec), int(close_sec)) if trading else (None, None)
        for date, open_sec, close_sec, trading in session_iter
    }


def to_seconds(ts: pd.Series) -> pd.Series:
    """
    Convert a Series of tz-aware timestamps to UTC epoch seconds (vectorized).

    Parameters
    ----------
    ts : pandas.Series
        Series of timezone-aware pandas Timestamps (dtype like
        'datetime64[ns, tz]') to be converted to seconds since Unix epoch.
        May contain NaT values.

    Returns
    -------
    pandas.Series
        Int64 Series of epoch seconds (UTC) aligned with `ts`.

    Raises
    ------
    TypeError
        If `ts` is not datetime-like or contains tz-naive timestamps.

    Notes
    -----
    - Conversion is fully vectorized: the timestamps are converted to UTC,
      viewed as int64 nanoseconds, and divided by 1e9 to obtain whole seconds.
    - NaT values propagate through as large negative integers; callers that
      require explicit missing handling should mask or filter prior to use.
    """

    return ts.dt.tz_convert("UTC").astype("int64") // 1_000_000_000


def to_seconds_int(ts: pd.Timestamp) -> int:
    """
    Convert a single tz-aware pandas Timestamp to UTC epoch seconds.

    Parameters
    ----------
    ts : pandas.Timestamp
        Timezone-aware Timestamp to convert to seconds since Unix epoch (UTC).

    Returns
    -------
    int
        Epoch seconds corresponding to `ts` when expressed in UTC.

    Raises
    ------
    TypeError
        If `ts` is not datetime-like or is tz-naive.

    Notes
    -----
    - This is the scalar analogue of `to_seconds` for use in non-vectorized
      paths (e.g., per-item logic in the sampling loop).
    - The conversion uses `ts.tz_convert("UTC").value // 1_000_000_000` to
      align with the vectorized implementation.
    """

    return ts.tz_convert("UTC").value // 1_000_000_000
