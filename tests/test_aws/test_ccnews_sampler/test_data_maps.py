"""
Purpose
-------
Verify that `build_session_dict` converts tz-aware session times to UTC epoch seconds
and returns `(None, None)` for non-trading days.

Key behaviors
-------------
- Timestamps in `session_open_utc` / `session_close_utc` map to integer epoch seconds.
- Dates flagged with `is_trading_day=False` produce `(None, None)` session tuples.
- The function respects the provided `str_dates` keys (no implicit reformatting).

Conventions
-----------
- Test data uses a weekday (trading) and a Saturday (non-trading).
- Times are specified in UTC to avoid DST ambiguity.

Downstream usage
----------------
Run with PyTest as part of the calendar utilities test suite:
`python -m pytest -q tests/.../test_calendar_utils.py`
"""

import pandas as pd

from aws.ccnews_sampler.data_maps import SessionTimes, build_session_dict

TEST_STR_DATES: pd.Series = pd.Series(
    [
        "2024-01-03",  # Wednesday
        "2024-01-06",  # Saturday
    ]
)
TEST_NYSE_CAL: pd.DataFrame = pd.DataFrame(
    {
        "is_trading_day": [True, False],
        "session_open_utc": [pd.Timestamp("2024-01-03 14:30:00+00:00"), pd.NaT],
        "session_close_utc": [pd.Timestamp("2024-01-03 21:00:00+00:00"), pd.NaT],
    },
    index=pd.to_datetime(
        [
            "2024-01-03",  # Wednesday
            "2024-01-06",  # Saturday
        ]
    ),
)


def test_build_session_dict() -> None:
    """
    Convert one trading day and one non-trading day into session-seconds mapping.

    Behavior validated
    ------------------
    - The trading Wednesday maps to concrete epoch seconds for open (14:30 UTC)
      and close (21:00 UTC).
    - The Saturday (is_trading_day=False) maps to `(None, None)`.

    Notes
    -----
    - The test frame must include `is_trading_day` (True for Wed, False for Sat);
      `build_session_dict` uses it to null out session times on non-trading days.
    - Expected epoch seconds for 2024-01-03 are:
        open  = 1704292200  (2024-01-03 14:30:00Z)
        close = 1704315600  (2024-01-03 21:00:00Z)
    """

    result_dict: dict[str, SessionTimes] = build_session_dict(TEST_STR_DATES, TEST_NYSE_CAL)
    expected_dict: dict[str, SessionTimes] = {
        "2024-01-03": (1704292200, 1704315600),
        "2024-01-06": (None, None),
    }
    assert result_dict == expected_dict
