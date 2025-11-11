"""
Purpose
-------
Unit tests for `aws.ccnews_sampler.data_maps`.

Key behaviors
-------------
- Verify that `build_cap_dict` converts per-day integer caps into a dictionary
  of Python ints keyed by New York civil date strings.
- Verify that `build_session_dict` correctly maps trading days to UTC epoch
  seconds and maps non-trading days to `(None, None)`.
- Verify that `build_data_maps` orchestrates the construction of all maps
  consistently and that all key sets align.
- Verify that `to_seconds` and `to_seconds_int` produce consistent epoch
  second values for tz-aware timestamps.

Conventions
-----------
- All tests operate on small in-memory DataFrames; no real database or
  calendar sources are touched.
- Date keys are formatted using the shared DATE_FMT configuration so tests
  remain aligned with the production sampler.
- Numeric comparisons that should be exact use direct equality; fractional
  checks use `pytest.approx` if needed.

Downstream usage
----------------
- Run via `pytest` as part of the CI suite.
- Serves as executable documentation of how NYSE calendar slices are
  converted into dictionary-based lookup structures for WARC sampling.
"""

import pandas as pd
import pytest

from aws.ccnews_sampler import data_maps
from aws.ccnews_sampler.ccnews_sampler_config import DATE_FMT


def test_build_cap_dict_builds_int_caps() -> None:
    """
    Verify that `build_cap_dict` produces a date-keyed mapping with Python ints.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if:
            - Each key is a 'YYYY-MM-DD' string.
            - Each value is a `(intraday_cap, overnight_cap)` pair of Python ints.
            - Caps reflect the original DataFrame values.

    Raises
    ------
    AssertionError
        If any cap is not an `int` or if the resulting mapping does not match
        the expected values.

    Notes
    -----
    - This test focuses on positional alignment between `str_dates` and the
      caps DataFrame as well as type normalization away from numpy scalars.
    """

    str_dates = pd.Series(["2024-01-01", "2024-01-02"])
    caps = pd.DataFrame(
        {"intraday_cap": [3, 7], "overnight_cap": [2, 5]},
    )

    result = data_maps.build_cap_dict(str_dates, caps)

    expected = {
        "2024-01-01": (3, 2),
        "2024-01-02": (7, 5),
    }
    assert result == expected

    for day, (intra, over) in result.items():
        assert isinstance(day, str)
        assert isinstance(intra, int)
        assert isinstance(over, int)


def test_build_session_dict_encodes_trading_days_only() -> None:
    """
    Verify that `build_session_dict` encodes trading days as epoch seconds
    when all rows are trading days (non-trading days dropped upstream).

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if:
            - Each trading day maps to `(open_epoch_s, close_epoch_s)` tuples
              matching `to_seconds_int` applied to the timestamps.
            - All keys correspond to trading dates only.

    Raises
    ------
    AssertionError
        If any mapping does not match expected epoch seconds or if extra
        unexpected keys appear.

    Notes
    -----
    - Upstream calendar utilities drop non-trading days, so this test
      validates the trading-day branch only.
    """

    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    session_open = pd.to_datetime(["2024-01-02 14:30:00", "2024-01-03 14:30:00"], utc=True)
    session_close = pd.to_datetime(["2024-01-02 21:00:00", "2024-01-03 21:00:00"], utc=True)
    nyse_cal = pd.DataFrame(
        {
            "session_open_utc": session_open,
            "session_close_utc": session_close,
            "is_trading_day": [True, True],
        },
        index=idx,
    )

    str_dates = pd.Series(idx.strftime(DATE_FMT))
    result = data_maps.build_session_dict(str_dates, nyse_cal)

    for i, day in enumerate(str_dates):
        open_expected = data_maps.to_seconds_int(session_open[i])
        close_expected = data_maps.to_seconds_int(session_close[i])
        assert result[day] == (open_expected, close_expected)


def test_build_data_maps_wires_cap_session_and_fraction_dicts_consistently() -> None:
    """
    Verify that `build_data_maps` constructs all internal dictionaries with
    aligned keys and correct values.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if:
            - `cap_dict`, `session_dict`, and `overnight_fraction_dict` share the
              same set of keys.
            - `valid_date_set` matches that shared key set.
            - A sample day's caps, session times, and overnight_fraction match
              the input calendar.

    Raises
    ------
    AssertionError
        If any dictionary is missing keys, has extra keys, or contains values
        that do not match the underlying calendar.

    Notes
    -----
    - Uses a three-day calendar with one non-trading day in the middle to
      exercise both trading and non-trading behavior.
    """

    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    session_open = pd.to_datetime(
        ["2024-01-01 14:30:00", "2024-01-02 14:30:00", "2024-01-03 14:30:00"],
        utc=True,
    )
    session_close = pd.to_datetime(
        ["2024-01-01 21:00:00", "2024-01-02 21:00:00", "2024-01-03 21:00:00"],
        utc=True,
    )

    nyse_cal = pd.DataFrame(
        {
            "intraday_cap": [8, 6, 10],
            "overnight_cap": [2, 4, 0],
            "session_open_utc": session_open,
            "session_close_utc": session_close,
            "is_trading_day": [True, False, True],
            "overnight_fraction": [0.2, 0.4, 0.1],
        },
        index=idx,
    )

    data = data_maps.build_data_maps(nyse_cal)

    expected_keys = set(idx.strftime(DATE_FMT))

    assert set(data.cap_dict.keys()) == expected_keys
    assert set(data.session_dict.keys()) == expected_keys
    assert set(data.overnight_fraction_dict.keys()) == expected_keys
    assert data.valid_date_set == expected_keys

    # Spot-check the first trading day.
    key0 = idx[0].strftime(DATE_FMT)
    assert data.cap_dict[key0] == (8, 2)
    assert data.overnight_fraction_dict[key0] == pytest.approx(0.2)

    open0, close0 = data.session_dict[key0]
    assert open0 == data_maps.to_seconds_int(session_open[0])
    assert close0 == data_maps.to_seconds_int(session_close[0])

    # Middle day is non-trading → session times should be (None, None).
    key1 = idx[1].strftime(DATE_FMT)
    assert data.session_dict[key1] == (None, None)


def test_to_seconds_and_to_seconds_int_are_consistent() -> None:
    """
    Check that `to_seconds` and `to_seconds_int` agree on epoch seconds.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if, for a small set of tz-aware timestamps, the
        vectorized conversion via `to_seconds` matches the scalar conversion
        via `to_seconds_int` element-wise.

    Raises
    ------
    AssertionError
        If any scalar/element-wise pair of epoch seconds differs between the
        two conversion paths.

    Notes
    -----
    - This test mainly guards against accidental divergence between the
      vectorized and scalar implementations if one is edited in the future.
    """

    ts = pd.to_datetime(
        ["2024-01-01 14:30:00", "2024-01-02 09:45:00"],
        utc=True,
    )
    series = pd.Series(ts)

    vec_seconds = data_maps.to_seconds(series)

    for i, t in enumerate(ts):
        scalar_sec = data_maps.to_seconds_int(t)
        assert vec_seconds.iloc[i] == scalar_sec
