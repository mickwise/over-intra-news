"""
Purpose
-------
Validate DST-aware day-length math in `calculate_minutes`.

Behavior validated
------------------
- `day_length_min` equals 1380 minutes on the spring-forward day (23h),
  1440 minutes on a normal day (24h), and 1500 minutes on the fall-back day (25h).
- `minutes_open` is 390 minutes (6.5h) for the provided session bounds,
  regardless of DST.

Conventions
-----------
- Session bounds are specified as UTC instants:
  * 2024-03-10 uses 14:30–21:00Z (EST trading day).
  * 2024-03-11 uses 13:30–20:00Z (EDT trading day).
  * 2024-11-03 uses 14:30–21:00Z (EST trading day).
- The DataFrame is indexed by `trading_day` and mutated in place by `calculate_minutes`.

Notes
-----
None
"""

import pandas as pd

from aws.ccnews_sampler.quota import calculate_minutes


def test_calculate_minutes_dst() -> None:
    """
    Verify day-length and session-minute calculations across DST boundaries.

    Behavior validated
    ------------------
    - 2024-03-10 (spring forward): day_length_min == 1380; minutes_open == 390.
    - 2024-03-11 (first EDT trading day): day_length_min == 1440; minutes_open == 390.
    - 2024-11-03 (fall back): day_length_min == 1500; minutes_open == 390.

    Notes
    -----
    - Session bounds are given in UTC and correspond to 9:30–16:00 NY time
      on each date (EST/EDT as appropriate).
    """

    nyse_cal = generate_test_nyse_cal()
    calculate_minutes(nyse_cal)
    assert nyse_cal.loc["2024-03-10", "minutes_open"] == 390  # EST
    assert nyse_cal.loc["2024-03-10", "day_length_min"] == 1380  # EST
    assert nyse_cal.loc["2024-03-11", "minutes_open"] == 390  # EDT
    assert nyse_cal.loc["2024-03-11", "day_length_min"] == 1440  # EDT
    assert nyse_cal.loc["2024-11-03", "minutes_open"] == 390  # EST
    assert nyse_cal.loc["2024-11-03", "day_length_min"] == 1500  # EST


def generate_test_nyse_cal() -> pd.DataFrame:
    """
    Build a minimal three-row calendar around 2024 DST transitions.

    Returns
    -------
    pandas.DataFrame
        Index: `trading_day` (Timestamp)
        Columns:
          - session_open_utc  (tz-aware UTC)
          - session_close_utc (tz-aware UTC)

    Composition
    -----------
    - 2024-03-10: 14:30–21:00Z  (EST 09:30–16:00)
    - 2024-03-11: 13:30–20:00Z  (EDT 09:30–16:00)
    - 2024-11-03: 14:30–21:00Z  (EST 09:30–16:00)

    Notes
    -----
    - `calculate_minutes` will add `minutes_open` and `day_length_min` in place.
    """

    return pd.DataFrame(
        {
            "trading_day": pd.to_datetime(
                [
                    "2024-03-10",  # Standard Time (EST)
                    "2024-03-11",  # Daylight Time (EDT)
                    "2024-11-03",  # Standard Time (EST)
                ]
            ),
            "session_open_utc": pd.to_datetime(
                [
                    "2024-03-10 14:30:00+00:00",
                    "2024-03-11 13:30:00+00:00",
                    "2024-11-03 14:30:00+00:00",
                ]
            ),
            "session_close_utc": pd.to_datetime(
                [
                    "2024-03-10 21:00:00+00:00",
                    "2024-03-11 20:00:00+00:00",
                    "2024-11-03 21:00:00+00:00",
                ]
            ),
        }
    ).set_index("trading_day")
