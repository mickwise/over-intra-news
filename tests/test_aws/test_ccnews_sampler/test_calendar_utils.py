"""
Purpose
-------
Unit tests for `aws.ccnews_sampler.calendar_utils`.

Key behaviors
-------------
- Verify that `month_bounds` applies left/right padding correctly, including
  special-case handling at the sampling horizon edges.
- Check that `calculate_overnight_fraction_trading_days` computes overnight
  fractions from trading-day session times and respects the Aug-2016 edge case.
- Check that `calculate_overnight_fraction`:
  - wires in the trading-day fractions,
  - backfills over non-trading days, and
  - drops padded dates outside the core `(year, month)`.
- Ensure `extract_nyse_cal` orchestrates DB access and delegates overnight
  computation without hitting a real database.

Conventions
-----------
- All external dependencies (DB connections, exchange_calendars, etc.) are
  mocked or patched; tests operate on small in-memory DataFrames.
- Numeric assertions for fractions use `pytest.approx` to avoid brittle
  comparisons.
- Tests focus on module logic and orchestration, not on pandas internals.

Downstream usage
----------------
- Run via `pytest` as part of the CI suite.
- Serves as executable documentation of calendar/overnight-fraction behavior
  for the CC-NEWS sampler.
"""

import datetime as dt
from typing import Any, Dict

import pandas as pd
import pytest

from aws.ccnews_sampler import calendar_utils


def test_month_bounds_interior_month_applies_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that `month_bounds` applies both left and right padding for interior months.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to temporarily patch `calendar_utils.PAD_DAYS` so that
        the padding behavior is deterministic and independent of the real config.

    Returns
    -------
    None
        The test passes if `month_bounds` returns a `(start, end)` pair where:
            - `start` is shifted left by the configured `PAD_DAYS`, and
            - `end` is shifted right by the same `PAD_DAYS` for a non-edge month.

    Raises
    ------
    AssertionError
        If the returned start or end dates do not reflect the expected padding
        relative to the first day of the month and the first day of the next month.

    Notes
    -----
    - Uses a small, fixed padding (`3` days) to keep expectations simple.
    - Treats June 2020 as a generic interior month, i.e. not affected by any
      special-casing of the sampling horizon edges.
    """

    # Use a small deterministic padding.
    monkeypatch.setattr(calendar_utils, "PAD_DAYS", dt.timedelta(days=3), raising=True)

    start, end = calendar_utils.month_bounds(2020, 6)

    assert start == dt.date(2020, 6, 1) - dt.timedelta(days=3)
    # Next month is July; end is padded by +3 days.
    assert end == dt.date(2020, 7, 1) + dt.timedelta(days=3)


def test_month_bounds_left_edge_disables_left_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that `month_bounds` does not left-pad August 2016 (left horizon edge).

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to patch `calendar_utils.PAD_DAYS` so the right-padding
        behavior can be asserted independently of the actual configuration.

    Returns
    -------
    None
        The test passes if:
            - The returned `start` date for `(2016, 8)` is exactly 2016-08-01
              (no left padding).
            - The returned `end` date is the first day of the next month plus
              the configured right padding.

    Raises
    ------
    AssertionError
        If `start` is shifted earlier than 2016-08-01 or if the right padding is
        not applied as expected.

    Notes
    -----
    - Models the “left edge” of the sampling horizon where historical data does
      not extend far enough to support left padding.
    - Confirms that only the left side is suppressed; right padding should still
      be applied at this edge.
    """
    monkeypatch.setattr(calendar_utils, "PAD_DAYS", dt.timedelta(days=5), raising=True)

    start, end = calendar_utils.month_bounds(2016, 8)

    # No left padding at the left horizon.
    assert start == dt.date(2016, 8, 1)
    # Right padding still applies.
    assert end == dt.date(2016, 9, 1) + dt.timedelta(days=5)


def test_month_bounds_right_edge_disables_right_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that `month_bounds` does not right-pad July 2025 (right horizon edge).

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to patch `calendar_utils.PAD_DAYS` so expected padding
        behavior at the right edge can be checked explicitly.

    Returns
    -------
    None
        The test passes if:
            - `start` for `(2025, 7)` is shifted left by the configured padding.
            - `end` is exactly 2025-08-01 (no right padding added).

    Raises
    ------
    AssertionError
        If left padding is omitted unexpectedly, or if `end` is extended beyond
        the first day of the next month.

    Notes
    -----
    - Models one of the right-edge months of the sampling horizon where future
      data is not available for right padding.
    - Confirms asymmetric behavior: left padding is allowed, right padding is not.
    """
    monkeypatch.setattr(calendar_utils, "PAD_DAYS", dt.timedelta(days=2), raising=True)

    start, end = calendar_utils.month_bounds(2025, 7)

    # Left padding still applies.
    assert start == dt.date(2025, 7, 1) - dt.timedelta(days=2)
    # No right padding at the right horizon.
    assert end == dt.date(2025, 8, 1)


def test_calculate_overnight_fraction_trading_days_basic() -> None:
    """
    Verify basic overnight fraction computation for consecutive trading days.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if `calculate_overnight_fraction_trading_days` returns a
        Series whose values on the second and third trading days match the expected
        ratio:
            overnight_minutes / (overnight_minutes + intraday_minutes)
        for a simple, repeated intraday session.

    Raises
    ------
    AssertionError
        If the resulting Series index is not aligned with the input calendar
        index, or if the computed fractions deviate from the expected value
        beyond numerical tolerance.

    Notes
    -----
    - Constructs a minimal calendar of three consecutive trading days with
      identical session open/close times.
    - Ignores the first trading day’s fraction in the assertion because its
      previous close is NaN; focuses on days where the previous close is
      well-defined via `.shift(1)`.
    - Uses `pytest.approx` to avoid brittle comparisons on floating-point results.
    """
    # Three consecutive trading days with identical intraday sessions.
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    open_times = pd.to_datetime(
        ["2024-01-02 14:30", "2024-01-03 14:30", "2024-01-04 14:30"], utc=True
    )
    close_times = pd.to_datetime(
        ["2024-01-02 21:00", "2024-01-03 21:00", "2024-01-04 21:00"], utc=True
    )
    calendar = pd.DataFrame(
        {
            "session_open_utc": open_times,
            "session_close_utc": close_times,
            "is_trading_day": [True, True, True],
        },
        index=idx,
    )

    overnight_frac = calendar_utils.calculate_overnight_fraction_trading_days(
        calendar, year=2024, month=1
    )

    # First day has no previous_close_utc, so overnight_minutes is 0 → fraction NaN or 0.
    # We only assert the second and third days where the math is well-defined.
    # intraday: 6.5h = 390 minutes
    intraday_minutes = 6.5 * 60
    # Overnight = 17.5h = 1050 minutes between 21:00 and 14:30 next day.
    overnight_minutes = 17.5 * 60
    expected_fraction = overnight_minutes / (overnight_minutes + intraday_minutes)

    assert overnight_frac.index.equals(idx)
    assert overnight_frac.loc["2024-01-03"] == pytest.approx(expected_fraction)
    assert overnight_frac.loc["2024-01-04"] == pytest.approx(expected_fraction)


def test_calculate_overnight_fraction_trading_days_aug2016_special_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify the Aug-2016 special-case logic for injecting a prior close.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to patch `exchange_calendars.get_calendar` so that
        the “previous close” for 2016-08-01 comes from a deterministic dummy
        schedule rather than the real exchange calendar.

    Returns
    -------
    None
        The test passes if:
            - The overnight fraction for `2016-08-01` is computed using the injected
              close timestamp from `2016-07-29`.
            - The overnight fraction for `2016-08-02` is finite, confirming that the
              standard previous-close logic operates normally after the patched day.

    Raises
    ------
    AssertionError
        If `get_calendar` is not called with the expected arguments, or if the
        fraction for `2016-08-01` does not match the hand-computed ratio based on
        the injected prior close.

    Notes
    -----
    - Models the left-horizon edge where the first sampling day (`2016-08-01`)
      needs a previous trading close from outside the main data range.
    - The dummy calendar is kept intentionally small, with a single “close” value
      used for the injected prior close.
    """
    # Stub out exchange_calendars.get_calendar to return a known prior close.
    last_close = pd.Timestamp("2016-07-29 20:00:00", tz="UTC")

    class DummyCalendar:
        def __init__(self, close_ts: pd.Timestamp) -> None:
            self.schedule = pd.DataFrame({"close": [close_ts]})

    def fake_get_calendar(name: str, start: str, end: str) -> DummyCalendar:
        assert name == "XNYS"
        assert start == "2016-07-29"
        assert end == "2016-07-29"
        return DummyCalendar(last_close)

    monkeypatch.setattr(calendar_utils.xcals, "get_calendar", fake_get_calendar, raising=True)

    idx = pd.to_datetime(["2016-08-01", "2016-08-02"])
    open_times = pd.to_datetime(["2016-08-01 13:30", "2016-08-02 13:30"], utc=True)
    close_times = pd.to_datetime(["2016-08-01 20:00", "2016-08-02 20:00"], utc=True)
    calendar = pd.DataFrame(
        {
            "session_open_utc": open_times,
            "session_close_utc": close_times,
            "is_trading_day": [True, True],
        },
        index=idx,
    )

    overnight_frac = calendar_utils.calculate_overnight_fraction_trading_days(
        calendar, year=2016, month=8
    )

    # For 2016-08-01: overnight from 2016-07-29 20:00 to 2016-08-01 13:30.
    overnight_minutes_first = (
        pd.Timestamp("2016-08-01 13:30:00", tz="UTC") - last_close
    ).total_seconds() / 60
    intraday_minutes = (
        pd.Timestamp("2016-08-01 20:00:00", tz="UTC")
        - pd.Timestamp("2016-08-01 13:30:00", tz="UTC")
    ).total_seconds() / 60
    expected_first = overnight_minutes_first / (overnight_minutes_first + intraday_minutes)

    assert overnight_frac.loc["2016-08-01"] == pytest.approx(expected_first)
    # Second day uses the regular previous-close logic; just assert it is finite.
    assert pd.notna(overnight_frac.loc["2016-08-02"])


def test_calculate_overnight_fraction_filters_core_month_trading_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that `calculate_overnight_fraction` wires trading-day fractions and drops
    both padded and non-trading dates outside the core month.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to patch
        `calendar_utils.calculate_overnight_fraction_trading_days` with a stub
        that returns known fractions on a subset of dates, including dates outside
        the core (year, month).

    Returns
    -------
    None
        The test passes if:
            - The result index contains only the requested month’s trading dates.
            - Trading-day fractions from the stub are preserved on trading days
              within the month.
            - Non-trading days and padded boundary days are fully excluded.

    Raises
    ------
    AssertionError
        If padded or non-trading dates are retained, or if trading-day fractions
        are altered unexpectedly.

    Notes
    -----
    - Builds a small padded calendar around January 2024, including:
        - a trading day in the prior month,
        - a non-trading day inside the month, and
        - a trading day after the month.
    - The stubbed trading-day fractions make it easy to see whether month filtering
      and non-trading-day exclusion behave as intended.
    """
    # Padded window across month boundaries:
    idx = pd.to_datetime(["2023-12-31", "2024-01-01", "2024-01-02", "2024-02-01"])
    calendar = pd.DataFrame(
        {
            "session_open_utc": pd.to_datetime(
                ["2023-12-31 14:30", "2024-01-01 14:30", "2024-01-02 14:30", "2024-02-01 14:30"],
                utc=True,
            ),
            "session_close_utc": pd.to_datetime(
                ["2023-12-31 21:00", "2024-01-01 21:00", "2024-01-02 21:00", "2024-02-01 21:00"],
                utc=True,
            ),
            "is_trading_day": [True, False, True, True],
        },
        index=idx,
    )

    # Stub: fractions only for trading days; non-trading days are omitted.
    def fake_trading_frac(_cal: pd.DataFrame, _year: int, _month: int) -> pd.Series:
        values = [0.1, 0.3, 0.9]
        index = pd.to_datetime(["2023-12-31", "2024-01-02", "2024-02-01"])
        return pd.Series(values, index=index, name="overnight_fraction")

    monkeypatch.setattr(
        calendar_utils, "calculate_overnight_fraction_trading_days", fake_trading_frac, raising=True
    )

    result = calendar_utils.calculate_overnight_fraction(calendar, year=2024, month=1)

    # Only the core January 2024 trading days should remain (non-trading days removed).
    expected_index = pd.to_datetime(["2024-01-02"])
    assert result.index.equals(expected_index)

    # 2024-01-02 keeps its direct fraction (0.3).
    assert result.loc["2024-01-02", "overnight_fraction"] == pytest.approx(0.3)


def test_extract_nyse_cal_uses_db_and_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that `extract_nyse_cal` orchestrates DB access and delegates overnight computation.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to patch:
        - `month_bounds` to a fixed `(start, end)` pair,
        - `build_calendar_query` to a known SQL string,
        - `pandas.read_sql` to return a synthetic calendar DataFrame,
        - `connect_to_db` to a dummy context manager, and
        - `calculate_overnight_fraction` to a stub that records its inputs and
        returns the frame unchanged.

    Returns
    -------
    None
        The test passes if:
            - `read_sql` is called with the expected SQL, bounds, index column, and
              parse_dates configuration.
            - `calculate_overnight_fraction` receives:
                - the DataFrame returned by `read_sql`,
                - the correctly parsed integer year and month.
            - The final result returned by `extract_nyse_cal` is exactly the object
              returned by the stubbed `calculate_overnight_fraction`.

    Raises
    ------
    AssertionError
        If any of the orchestration steps are mis-wired, such as incorrect query
        string, parameter bounds, index configuration, or year/month values.

    Notes
    -----
    - Ensures that `extract_nyse_cal` remains a thin coordinator that:
    - builds the query,
    - delegates to the DB layer to load a padded calendar,
    - normalizes time columns to tz-aware UTC, and
    - hands off to `calculate_overnight_fraction` for business logic.
    - Avoids touching a real database or the real exchange calendar, keeping the
      test fast and deterministic.
    """
    # Stub month_bounds to avoid relying on real PAD_DAYS logic here.
    fake_bounds = (dt.date(2020, 1, 1), dt.date(2020, 2, 1))
    monkeypatch.setattr(calendar_utils, "month_bounds", lambda y, m: fake_bounds, raising=True)

    # Stub build_calendar_query to a known sentinel string.
    fake_query = "SELECT * FROM trading_calendar WHERE trading_day >= %s AND trading_day < %s;"
    monkeypatch.setattr(calendar_utils, "build_calendar_query", lambda: fake_query, raising=True)

    # Capture arguments passed to read_sql.
    captured_sql: Dict[str, Any] = {}

    def fake_read_sql(
        sql: str,
        _: Any,
        index_col: str | None = None,
        params: Any | None = None,
        parse_dates: list[str] | None = None,
    ) -> pd.DataFrame:
        captured_sql["sql"] = sql
        captured_sql["params"] = params
        captured_sql["index_col"] = index_col
        captured_sql["parse_dates"] = parse_dates
        # Minimal calendar frame; index set to trading_day.
        df = pd.DataFrame(
            {
                "trading_day": pd.to_datetime(["2020-01-15"]),
                "session_open_utc": ["2020-01-15 14:30:00"],
                "session_close_utc": ["2020-01-15 21:00:00"],
                "is_trading_day": [True],
            }
        ).set_index("trading_day")
        return df

    monkeypatch.setattr(pd, "read_sql", fake_read_sql, raising=True)

    # Stub DB connection context manager.
    class DummyConn:
        pass

    class DummyContext:
        def __enter__(self) -> DummyConn:
            return DummyConn()

        def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
            return None

    monkeypatch.setattr(calendar_utils, "connect_to_db", DummyContext, raising=True)

    # Stub calculate_overnight_fraction to capture its input and return it.
    captured_calendar: Dict[str, Any] = {}

    def fake_calc_overnight(cal: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
        captured_calendar["df"] = cal.copy()
        captured_calendar["year"] = year
        captured_calendar["month"] = month
        return cal

    monkeypatch.setattr(
        calendar_utils, "calculate_overnight_fraction", fake_calc_overnight, raising=True
    )

    result = calendar_utils.extract_nyse_cal("2020", "01")

    # Check that the SQL query and bounds were wired correctly.
    assert captured_sql["sql"] == fake_query
    assert captured_sql["params"] == fake_bounds
    assert captured_sql["index_col"] == "trading_day"
    assert captured_sql["parse_dates"] == ["trading_day"]

    # Check that calculate_overnight_fraction saw the correct year/month.
    assert captured_calendar["year"] == 2020
    assert captured_calendar["month"] == 1

    # Session columns should have been converted to tz-aware UTC.
    cal_df = captured_calendar["df"]
    assert cal_df.index.name == "trading_day"
    assert cal_df["session_open_utc"].dt.tz is not None
    assert cal_df["session_close_utc"].dt.tz is not None

    # The final result is whatever calculate_overnight_fraction returned.
    assert isinstance(result, pd.DataFrame)
    assert result.equals(cal_df)
