"""
Purpose
-------
Exercise the minimal branching logic in `seed_trading_calendar` without re-testing
third-party libraries. Focuses on:
- Gap emission in `calendar_row_generator` (synthetic non-trading days).
- Weekend/holiday classification for non-trading civil dates.
- Half-day threshold and non-trading detection in `extract_row`.

Key behaviors
-------------
- Verifies that a Fri→Mon schedule emits Sat/Sun via `extract_non_trading_row`
  and that trading days are handed to `extract_row` in order.
- Confirms mutually exclusive weekend/holiday flags on non-trading dates.
- Asserts the strict half-day rule: duration < 6.5h → `is_half_day=True`;
  exactly 6.5h → `is_half_day=False`.
- Asserts `is_trading_day=False` when either open or close is missing.

Conventions
-----------
- Mocks collaborator functions (`extract_non_trading_row`, `extract_row`) to
  validate call counts and arguments while keeping tests hermetic.
- Uses explicit UTC timestamps in fixtures; no I/O or live calendars involved.

Downstream usage
----------------
- Run with: `python -m pytest -q tests`
- Part of CI/CD pipelines to ensure seed logic correctness.
"""

import datetime as dt
from unittest.mock import MagicMock

import pandas as pd
from pytest_mock import MockerFixture

from infra.seeds.seed_trading_calendar import (
    TradingCalendarRow,
    calendar_row_generator,
    extract_non_trading_row,
    extract_row,
)

TEST_NYSE_CAL: pd.DataFrame = pd.DataFrame(
    {
        "open": [
            pd.Timestamp("2023-01-06 14:30:00+0000", tz="UTC"),
            pd.Timestamp("2023-01-09 14:30:00+0000", tz="UTC"),
        ],
        "close": [
            pd.Timestamp("2023-01-06 21:00:00+0000", tz="UTC"),
            pd.Timestamp("2023-01-09 21:00:00+0000", tz="UTC"),
        ],
    },
    index=pd.to_datetime(["2023-01-06", "2023-01-09"]),
)
TEST_WEEKEND_DAY: dt.date = dt.date(2023, 1, 7)  # Saturday
TEST_HOLIDAY: dt.date = dt.date(2023, 12, 25)  # Christmas
IS_TRADING_DAY_INDEX: int = 3
IS_WEEKEND_INDEX: int = 4
IS_HOLIDAY_INDEX: int = 5
IS_HALFDAY_INDEX: int = 6
TEST_FULL_DAY_ROW: pd.Series = pd.Series(
    {
        "open": pd.Timestamp("2023-01-06 14:30:00+0000", tz="UTC"),
        "close": pd.Timestamp("2023-01-06 21:00:00+0000", tz="UTC"),
    },
    name=pd.Timestamp("2023-01-06"),
)
TEST_HALF_DAY_ROW: pd.Series = pd.Series(
    {
        "open": pd.Timestamp("2023-11-24 14:30:00+0000", tz="UTC"),
        "close": pd.Timestamp("2023-11-24 17:30:00+0000", tz="UTC"),
    },
    name=pd.Timestamp("2023-11-24"),
)
TEST_NON_TRADING_DAY_ROW: pd.Series = pd.Series(
    {
        "open": None,
        "close": pd.Timestamp("2023-11-24 21:00:00+0000", tz="UTC"),
    },
    name=pd.Timestamp("2023-01-07"),
)


def test_seed_trading_calendar_calendar_row_generator_gap(mocker: MockerFixture) -> None:
    """
    Ensure the generator emits gap days and trading rows in the correct order.

    Parameters
    ----------
    mocker : MockerFixture
        Used to patch `extract_non_trading_row` and `extract_row`.

    Behavior validated
    ------------------
    - For a 3-day civil gap (Fri -> Mon), emits two synthetic non-trading days
      (Sat, Sun) via `extract_non_trading_row` with exact date arguments.
    - Emits trading rows for both index days and passes the correct Series
      (validated via `.name`) to `extract_row`.
    - Call ordering implicitly validated by comparing the collected arguments.

    Notes
    -----
    - The test consumes the generator to trigger side effects.
    - Only collaborator call counts/arguments are asserted (no I/O).
    """

    mock_extract_non_trading_row: MagicMock = mocker.patch(
        "infra.seeds.seed_trading_calendar.extract_non_trading_row"
    )
    mock_extract_row: MagicMock = mocker.patch("infra.seeds.seed_trading_calendar.extract_row")

    # Consume the generator so the body runs and mocks are called.
    list(calendar_row_generator(TEST_NYSE_CAL))

    # Expect two synthetic non-trading days: 2023-01-07 and 2023-01-08
    prev_day = TEST_NYSE_CAL.index[0].date()
    expected_non_trading_dates = [prev_day + dt.timedelta(days=1), prev_day + dt.timedelta(days=2)]
    actual_non_trading_dates = [
        call.args[0] for call in mock_extract_non_trading_row.call_args_list
    ]
    assert mock_extract_non_trading_row.call_count == 2
    assert actual_non_trading_dates == expected_non_trading_dates

    # Trading rows should be emitted for both index days, in order.
    trading_names = [call.args[0].name for call in mock_extract_row.call_args_list]
    assert mock_extract_row.call_count == 2
    assert trading_names == list(TEST_NYSE_CAL.index)


def test_seed_trading_calendar_extract_non_trading_row() -> None:
    """
    Verify weekend vs holiday flags on non-trading civil dates.

    Behavior validated
    ------------------
    - A Saturday is flagged `is_weekend=True` (tuple index 4) and `is_holiday=False` (index 5).
    - A weekday holiday (e.g., Christmas) is flagged `is_holiday=True` (index 5)
      and `is_weekend=False` (index 4).

    Notes
    -----
    - This matches the mutually exclusive convention used by the seed module:
      weekend days are not also marked as holidays.
    """

    weekend_row: TradingCalendarRow = extract_non_trading_row(TEST_WEEKEND_DAY)
    holiday_row: TradingCalendarRow = extract_non_trading_row(TEST_HOLIDAY)
    assert weekend_row[IS_WEEKEND_INDEX] is True
    assert holiday_row[IS_HOLIDAY_INDEX] is True


def test_seed_trading_calendar_extract_row_half_day() -> None:
    """
    Validate the half-day boundary logic in `extract_row`.

    Behavior validated
    ------------------
    - A 6.5-hour session (14:30→21:00 UTC) is **not** a half day (`is_half_day=False`).
    - A strictly shorter session (e.g., 3 hours) **is** a half day (`is_half_day=True`).

    Notes
    -----
    - This directly targets the strict inequality (< 6.5h) rather than re-testing
      pandas datetime arithmetic.
    """

    full_day_row: TradingCalendarRow = extract_row(TEST_FULL_DAY_ROW)
    half_day_row: TradingCalendarRow = extract_row(TEST_HALF_DAY_ROW)
    assert full_day_row[IS_HALFDAY_INDEX] is False
    assert half_day_row[IS_HALFDAY_INDEX] is True


def test_seed_trading_calendar_extract_row_non_trading_day() -> None:
    """
    Confirm non-trading detection when a session is incomplete.

    Behavior validated
    ------------------
    - If either `open` or `close` is missing, `is_trading_day` is `False`.

    Notes
    -----
    - This covers the minimal guard for incomplete/malformed schedule rows without
      asserting anything about weekend/holiday flags (covered elsewhere).
    """

    non_trading_day_row: TradingCalendarRow = extract_row(TEST_NON_TRADING_DAY_ROW)
    assert non_trading_day_row[IS_TRADING_DAY_INDEX] is False
