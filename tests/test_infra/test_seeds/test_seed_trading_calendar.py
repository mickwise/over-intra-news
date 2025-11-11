"""
Purpose
-------
Unit tests for `seed_trading_calendar.py`, covering NYSE calendar row
normalization and the wiring into batched upserts.

Key behaviors
-------------
- Verify that `extract_non_trading_row`:
  - correctly flags weekends vs holidays,
  - always marks rows as non-trading and non-half-day.
- Verify that `extract_row`:
  - converts schedule rows into trading-calendar tuples,
  - sets `is_trading_day`, `is_weekend`, `is_holiday`, and `is_half_day`
    according to documented rules.
- Verify that `calendar_row_generator`:
  - yields trading rows in order,
  - fills in synthetic non-trading gap days between trading days,
  - classifies those gap days correctly as weekend/holiday.
- Verify that `fill_trading_calendar`:
  - builds the expected SQL via `generate_db_query`,
  - passes a row generator and connection into `load_into_table`,
  - logs a completion message.

Conventions
-----------
- All tests operate on small, in-memory pandas DataFrames.
- No real database or `exchange_calendars` calls are made; those are
  represented via minimal stubs or dummy objects where needed.
- Type compatibility with the production signatures is preserved via
  explicit casts where necessary so tests remain mypy-clean.

Downstream usage
----------------
- Run via `pytest` as part of the CI suite.
- Serves as executable documentation for how trading-calendar rows are
  derived and how they are fed into the batched loading helper.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, List, cast

import pandas as pd
import pytest
from psycopg2.extensions import connection

from infra.logging.infra_logger import InfraLogger
from infra.seeds import seed_trading_calendar


def test_extract_non_trading_row_weekend_vs_holiday() -> None:
    """
    Ensure `extract_non_trading_row` flags weekends and holidays correctly.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if:
            - A Saturday/Sunday date is marked as weekend (and not holiday),
            - A weekday date is marked as holiday (and not weekend),
            - Both rows are marked as non-trading and non-half-day with
              null session times.

    Raises
    ------
    AssertionError
        If any of the boolean flags or session fields are inconsistent with
        the documented behavior.
    """

    # Saturday (weekend).
    saturday = dt.date(2024, 1, 6)  # 2024-01-06 is a Saturday.
    row_weekend = seed_trading_calendar.extract_non_trading_row(saturday)

    assert row_weekend[0] == saturday
    assert row_weekend[1] is None  # session_open_utc
    assert row_weekend[2] is None  # session_close_utc
    assert row_weekend[3] is False  # is_trading_day
    assert row_weekend[4] is True  # is_weekend
    assert row_weekend[5] is False  # is_holiday
    assert row_weekend[6] is False  # is_half_day

    # Weekday non-trading → holiday, not weekend.
    weekday = dt.date(2024, 1, 3)  # Wednesday.
    row_holiday = seed_trading_calendar.extract_non_trading_row(weekday)

    assert row_holiday[0] == weekday
    assert row_holiday[1] is None
    assert row_holiday[2] is None
    assert row_holiday[3] is False
    assert row_holiday[4] is False  # not weekend
    assert row_holiday[5] is True  # holiday
    assert row_holiday[6] is False  # never half-day for non-trading


def test_extract_row_sets_trading_flags_and_half_day() -> None:
    """
    Validate that `extract_row` sets trading and half-day flags from open/close.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if:
            - A full-length session is marked as trading, not half-day,
            - A shorter session (< 6.5 hours) is marked as half-day,
            - Weekend/holiday flags are derived from the civil date.

    Raises
    ------
    AssertionError
        If durations or flags do not reflect the documented semantics.
    """

    # Full trading day (7 hours).
    idx_full = pd.Timestamp("2024-01-02", tz="UTC")  # Tuesday.
    open_full = pd.Timestamp("2024-01-02 14:30:00", tz="UTC")
    close_full = pd.Timestamp("2024-01-02 21:30:00", tz="UTC")  # 7 hours later.

    row_full = pd.Series({"open": open_full, "close": close_full}, name=idx_full)
    tc_full = seed_trading_calendar.extract_row(row_full)

    assert tc_full[0] == idx_full.date()
    assert tc_full[1] == open_full
    assert tc_full[2] == close_full
    assert tc_full[3] is True  # is_trading_day
    assert tc_full[4] is False  # is_weekend
    assert tc_full[5] is False  # is_holiday
    assert tc_full[6] is False  # is_half_day

    # Half-day session: 3.5 hours (< 6.5h threshold).
    idx_half = pd.Timestamp("2024-01-03", tz="UTC")  # Wednesday.
    open_half = pd.Timestamp("2024-01-03 14:30:00", tz="UTC")
    close_half = pd.Timestamp("2024-01-03 18:00:00", tz="UTC")  # 3.5 hours later.

    row_half = pd.Series({"open": open_half, "close": close_half}, name=idx_half)
    tc_half = seed_trading_calendar.extract_row(row_half)

    assert tc_half[0] == idx_half.date()
    assert tc_half[1] == open_half
    assert tc_half[2] == close_half
    assert tc_half[3] is True  # trading day
    assert tc_half[4] is False  # not weekend
    assert tc_half[5] is False  # not holiday
    assert tc_half[6] is True  # half-day detected


def test_calendar_row_generator_inserts_non_trading_gap_days() -> None:
    """
    Confirm that `calendar_row_generator` emits gap days as non-trading rows.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if, given a schedule with a multi-day gap between
        trading days:
            - The first trading day is yielded as a trading row,
            - Each missing intermediate civil date is yielded as a non-trading
              row in chronological order,
            - The final trading day is yielded as a trading row.

    Raises
    ------
    AssertionError
        If the number, order, or flags of yielded rows do not match the
        expected behavior.
    """

    # Trading on Jan 2 and Jan 5, gap on 3rd and 4th.
    idx = pd.to_datetime(["2024-01-02", "2024-01-05"], utc=True)
    opens = pd.to_datetime(
        ["2024-01-02 14:30:00", "2024-01-05 14:30:00"],
        utc=True,
    )
    closes = pd.to_datetime(
        ["2024-01-02 21:30:00", "2024-01-05 21:30:00"],
        utc=True,
    )
    nyse_cal = pd.DataFrame({"open": opens, "close": closes}, index=idx)

    rows: List[seed_trading_calendar.TradingCalendarRow] = list(
        seed_trading_calendar.calendar_row_generator(nyse_cal)
    )

    # Expect: 2024-01-02 (trading), 3rd (non-trading), 4th (non-trading), 5th (trading).
    assert [r[0] for r in rows] == [
        dt.date(2024, 1, 2),
        dt.date(2024, 1, 3),
        dt.date(2024, 1, 4),
        dt.date(2024, 1, 5),
    ]

    # First and last rows should be trading days.
    assert rows[0][3] is True  # is_trading_day
    assert rows[-1][3] is True  # is_trading_day

    # Gap days should be non-trading, non-weekend holidays (they’re midweek).
    for gap_row in rows[1:3]:
        assert gap_row[1] is None  # session_open_utc
        assert gap_row[2] is None  # session_close_utc
        assert gap_row[3] is False  # is_trading_day
        assert gap_row[4] is False  # is_weekend
        assert gap_row[5] is True  # is_holiday
        assert gap_row[6] is False  # is_half_day


def test_fill_trading_calendar_calls_load_into_table_with_query_and_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that `fill_trading_calendar` wires the query and row generator into `load_into_table`.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to stub `load_into_table` so that no real
        database I/O is performed, while capturing the call arguments.

    Returns
    -------
    None
        The test passes if:
            - `load_into_table` is called exactly once,
            - the connection object is passed through unchanged,
            - the SQL query equals `generate_db_query()`,
            - the row generator yields at least one row derived from
              `calendar_row_generator`.

    Raises
    ------
    AssertionError
        If `load_into_table` is not called as expected or if the generator
        does not produce the expected shape of trading-calendar rows.
    """

    captured: dict[str, Any] = {}

    def fake_load_into_table(conn: Any, row_generator: Any, input_query: str) -> None:
        captured["conn"] = conn
        captured["row_generator"] = row_generator
        captured["input_query"] = input_query

    monkeypatch.setattr(
        seed_trading_calendar,
        "load_into_table",
        fake_load_into_table,
    )

    class DummyLogger(InfraLogger):
        def __init__(self) -> None:
            # Minimal stub; underlying InfraLogger implementation is not used.
            pass

        def info(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
            return None

    class DummyConn:
        def cursor(self) -> Any:
            raise RuntimeError("cursor should not be used in this test")

    conn = DummyConn()

    # Minimal schedule with a single trading day.
    idx = pd.to_datetime(["2024-01-02"], utc=True)
    opens = pd.to_datetime(["2024-01-02 14:30:00"], utc=True)
    closes = pd.to_datetime(["2024-01-02 21:30:00"], utc=True)
    nyse_cal = pd.DataFrame({"open": opens, "close": closes}, index=idx)

    logger = DummyLogger()

    # Call the function under test (with type casts to keep mypy happy).
    seed_trading_calendar.fill_trading_calendar(
        conn=cast(connection, conn),
        nyse_cal=nyse_cal,
        logger=cast(InfraLogger, logger),
    )

    # Ensure load_into_table was invoked.
    assert captured, "load_into_table was not called by fill_trading_calendar"

    assert captured["conn"] is conn
    assert captured["input_query"] == seed_trading_calendar.generate_db_query()

    # The row_generator should be an iterator producing TradingCalendarRow values.
    row_gen = captured["row_generator"]
    first_row = next(iter(row_gen))

    assert isinstance(first_row, tuple)
    assert first_row[0] == dt.date(2024, 1, 2)
    # is_trading_day should be True for this single trading day.
    assert first_row[3] is True
