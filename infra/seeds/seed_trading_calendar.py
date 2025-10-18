"""
seed_trading_calendar.py

Purpose
-------
Seed the `trading_calendar` reference table for OverIntraNews.
Fetches the NYSE schedule from `exchange_calendars`, derives per-day session
metadata (open/close UTC, weekend/holiday flags, half-day flag), and writes rows
into Postgres in batched upserts.

Key behaviors
-------------
- Single pass over the exchange schedule.
- Fills *non-trading civil dates* (weekends/holidays) missing between consecutive trading days.
- Uses batched inserts via psycopg2.extras.execute_values for speed.
- Transaction semantics: either all rows in the run are committed, or none.

Conventions
-----------
- All timestamps stored in UTC (`timestamptz`).
- `is_half_day` is True for sessions strictly shorter than 6.5 hours.
- On non-trading days, open/close columns are NULL and `is_trading_day=False`.

Downstream usage
----------------
Run this script to initialize or refresh the `trading_calendar` table.
Other ingestion scripts rely on this table for date alignment of external data
(news articles, membership histories, prices).
"""

import datetime as dt
import os
from typing import Iterator, Optional, Tuple, TypeAlias

import exchange_calendars as xcals
import pandas as pd
from dotenv import load_dotenv
from psycopg2.extensions import connection

from infra.logging.infra_logger import InfraLogger, initialize_logger
from infra.utils.db_utils import connect_to_db, load_into_table

TradingCalendarRow: TypeAlias = Tuple[
    dt.date,
    Optional[dt.datetime],
    Optional[dt.datetime],
    bool,
    bool,
    bool,
    bool,
]


SECONDS_IN_HOUR: int = 3600
SATURDAY: int = 5  # Python's datetime module: Monday is 0 and Sunday is 6


def ingest_nyse() -> None:
    """
    Orchestrate one full ingestion run.

    Steps
    -----
    1) Load environment variables (for DB and date span).
    2) Connect to Postgres.
    3) Fetch the NYSE schedule for [START_DATE, END_DATE] from exchange_calendars.
    4) Stream rows into a fixed-size batch, upserting when the batch fills.
    5) Commit on success; rollback on any exception.

    Side Effects
    ------------
    Writes to the `trading_calendar` table via INSERT ... ON CONFLICT DO UPDATE.

    Environment variables used
    --------------------------
    START_DATE : str (YYYY-MM-DD)
    END_DATE   : str (YYYY-MM-DD)

    Notes
    -----
    - Since NYSE has no lunch breaks, only open/close times are stored.
    """
    load_dotenv()
    start_date: str = os.environ["START_DATE"]
    end_date: str = os.environ["END_DATE"]
    logger: InfraLogger = initialize_logger("seed_trading_calendar")
    logger.info("START_RUN", f"Seeding trading_calendar for NYSE from {start_date} to {end_date}")
    nyse_cal: pd.DataFrame = xcals.get_calendar("XNYS", start_date, end_date).schedule[
        ["open", "close"]
    ]
    logger.info(
        "SCHEDULE_FETCHED",
        f"Fetched {len(nyse_cal)} trading days from exchange_calendars",
        {
            "trading_days": len(nyse_cal),
            "first_day": nyse_cal.index[0].date().isoformat(),
            "last_day": nyse_cal.index[-1].date().isoformat(),
        },
    )
    with connect_to_db() as conn:
        logger.info("UPSERT_BEGIN", "Connected to Postgres")
        fill_trading_calendar(conn, nyse_cal, logger)
    conn.close()
    logger.info("RUN_COMPLETE", "Completed seeding trading_calendar", {"status": "success"})


def fill_trading_calendar(conn: connection, nyse_cal: pd.DataFrame, logger: InfraLogger) -> None:
    """
    Upsert a NYSE calendar snapshot into Postgres using batched inserts.

    Parameters
    ----------
    conn : psycopg2.extensions.connection
        Open database connection; caller manages commit/rollback.
    nyse_cal : pd.DataFrame
        Exchange schedule indexed by trading day with 'open' and 'close'
        tz-aware UTC timestamps.

    Returns
    -------
    None

    Raises
    ------
    Exception
        Propagates any database or adapter errors during load.

    Notes
    -----
    - Uses `ON CONFLICT (trading_day) DO UPDATE` to keep rows fresh.
    - Rows are streamed by `calendar_row_generator` for gap filling.
    """

    input_query: str = generate_db_query()
    row_generator: Iterator[TradingCalendarRow] = calendar_row_generator(nyse_cal)
    load_into_table(conn, row_generator, input_query)
    logger.info("UPSERT_COMPLETE", "Upserted trading_calendar rows")


def generate_db_query() -> str:
    """
    Generate the SQL input query for upserting into `trading_calendar`.

    Returns
    -------
    str
        The SQL query string with placeholders for psycopg2.extras.execute_values.

    Notes
    -----
    - Uses ON CONFLICT to handle duplicate trading_day entries.
    - Updates all fields on conflict to ensure data freshness.
    """
    return """
        INSERT INTO trading_calendar (
            trading_day,
            session_open_utc,
            session_close_utc,
            is_trading_day,
            is_weekend,
            is_holiday,
            is_half_day
        ) VALUES %s
        ON CONFLICT (trading_day) DO UPDATE SET
            session_open_utc = EXCLUDED.session_open_utc,
            session_close_utc = EXCLUDED.session_close_utc,
            is_trading_day = EXCLUDED.is_trading_day,
            is_weekend = EXCLUDED.is_weekend,
            is_holiday = EXCLUDED.is_holiday,
            is_half_day = EXCLUDED.is_half_day;
    """


def calendar_row_generator(nyse_cal: pd.DataFrame) -> Iterator[TradingCalendarRow]:
    """
    Yield normalized calendar rows including synthetic non-trading dates.

    Parameters
    ----------
    nyse_cal : pd.DataFrame
        Schedule from `exchange_calendars` with index = trading day (Timestamp)
        and columns 'open'/'close' (tz-aware UTC).

    Returns
    -------
    Iterator[TradingCalendarRow]
        Tuples of:
        (trading_day: date,
         session_open_utc: datetime | None,
         session_close_utc: datetime | None,
         is_trading_day: bool,
         is_weekend: bool,
         is_holiday: bool,
         is_half_day: bool)

    Raises
    ------
    TypeError
        If the schedule index is not `pd.Timestamp`.

    Notes
    -----
    - Emits non-trading gap days between consecutive trading days.
    - `is_half_day` is True for sessions strictly < 6.5 hours.
    - Weekend uses civil weekday (Mon=0..Sun=6); holidays are non-weekend,
      non-trading days.
    """
    prev_day = nyse_cal.index[0].date()
    for day in nyse_cal.index:
        curr_day = day.date()
        delta_days = (curr_day - prev_day).days
        if delta_days > 1:
            for i in range(1, delta_days):
                gap_date = prev_day + dt.timedelta(days=i)
                yield extract_non_trading_row(gap_date)
        yield extract_row(nyse_cal.loc[day])
        prev_day = curr_day


def extract_row(row: pd.Series) -> TradingCalendarRow:
    """
    Convert a schedule row into a `trading_calendar` tuple.

    Parameters
    ----------
    row : pandas.Series
        A single row from the schedule with:
        - index name = trading day (Timestamp)
        - columns: 'open' (tz-aware UTC), 'close' (tz-aware UTC)

    Returns
    -------
    TradingCalendarRow
        (
          trading_day: date,
          session_open_utc: datetime,
          session_close_utc: datetime,
          is_trading_day: bool,
          is_weekend: bool,
          is_holiday: bool,
          is_half_day: bool
        )

    Notes
    -----
    - `is_trading_day` is True iff both open and close are present.
    - `is_half_day` is True iff session duration < 6.5 hours.
    - Weekend is computed on the civil date (Mon=0,...,Sun=6).
    """

    if not isinstance(row.name, pd.Timestamp):
        raise TypeError("Row index must be a pd.Timestamp")
    trading_day: dt.date = row.name.date()
    session_open_utc: dt.datetime = row["open"]
    session_close_utc: dt.datetime = row["close"]
    is_trading_day: bool = not pd.isna(session_open_utc) and not pd.isna(session_close_utc)
    is_half_day: bool = False
    if is_trading_day:
        curr_day_intra_seconds: float = (session_close_utc - session_open_utc).total_seconds()
        is_half_day = curr_day_intra_seconds < 6.5 * SECONDS_IN_HOUR
    is_weekend: bool = trading_day.weekday() >= SATURDAY
    is_holiday: bool = not is_trading_day and not is_weekend

    return (
        trading_day,
        session_open_utc,
        session_close_utc,
        is_trading_day,
        is_weekend,
        is_holiday,
        is_half_day,
    )


def extract_non_trading_row(date: dt.date) -> TradingCalendarRow:
    """
    Build a `trading_calendar` row for a non-trading civil date.

    Parameters
    ----------
    date : datetime.date
        Civil date to represent.

    Returns
    -------
    TradingCalendarRow
        (
          trading_day=date,
          session_open_utc=None,
          session_close_utc=None,
          is_trading_day=False,
          is_weekend=bool,
          is_holiday=not is_weekend,
          is_half_day=False
        )

    Notes
    -----
    - Weekend is Saturday/Sunday.
    - Non-weekend and non-trading implies `is_holiday=True`.
    """
    trading_day: dt.date = date
    is_weekend: bool = trading_day.weekday() >= SATURDAY
    return (
        trading_day,
        None,
        None,
        False,
        is_weekend,
        not is_weekend,
        False,
    )
