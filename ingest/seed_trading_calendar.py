"""
Seed the `trading_calendar` reference table for OverIntraNews.

This script pulls the NYSE schedule from `exchange_calendars`, derives per-day
session metadata (open/close UTC, weekend/holiday flags, half-day flag), and
writes rows into Postgres in batched upserts.

Key behaviors
-------------
- Single pass over the exchange schedule.
- Fills *non-trading civil dates* (weekends/holidays) that are missing between
  consecutive trading days so the table has one row per calendar date.
- Uses batched inserts via psycopg2.extras.execute_values for speed.
- Transaction semantics: either all rows in the run are committed, or none.

Conventions
-----------
- All timestamps are stored in UTC (`timestamptz`).
- `is_half_day` is True for any session strictly shorter than 6.5 hours.
- On non-trading days, open/close columns are NULL and `is_trading_day=False`.
"""

import datetime
import os
from typing import List, Optional, Tuple, TypeAlias

import exchange_calendars as xcals
import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from psycopg2.extensions import connection

TradingCalendarRow: TypeAlias = Tuple[
    datetime.date,
    Optional[datetime.datetime],
    Optional[datetime.datetime],
    bool,
    bool,
    bool,
    bool,
]


SECONDS_IN_HOUR: int = 3600
SATURDAY: int = 5  # Python's datetime module: Monday is 0 and Sunday is 6
BATCH_SIZE: int = 1000


def connect_to_db() -> connection:
    """
    Open a PostgreSQL connection using credentials from environment variables.

    Environment variables expected
    ------------------------------
    POSTGRES_DB : str
        Database name to connect to.
    POSTGRES_USER : str
        Database user.
    POSTGRES_PASSWORD : str
        Password for the database user.
    DB_HOST : str
        Hostname of the database server.
    DB_PORT : int
        Port number of the database server.

    Returns
    -------
    psycopg2.extensions.connection
        An open psycopg2 connection. Caller is responsible for closing it.

    Raises
    ------
    KeyError
        If any required environment variable is missing.
    psycopg2.OperationalError
        If the connection cannot be established
        (bad credentials, host unreachable, etc.).
    """
    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=os.environ["DB_PORT"],
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )


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
    """
    load_dotenv()
    conn: connection = connect_to_db()
    start_date: str = os.environ["START_DATE"]
    end_date: str = os.environ["END_DATE"]
    nyse_cal: pd.DataFrame = xcals.get_calendar("XNYS", start_date, end_date).schedule
    try:
        left_over_batch = fill_db(conn, nyse_cal)
        execute_batch(conn, left_over_batch)
        conn.commit()

    except Exception as e:
        conn.rollback()
        raise e

    finally:
        conn.close()


def fill_db(
    conn: connection, nyse_cal: pd.DataFrame
) -> List[TradingCalendarRow | None]:
    """
    Iterate the NYSE schedule and build batched rows for `trading_calendar`.

    For each pair of consecutive schedule dates, this function:
      - Detects gaps in civil dates and inserts the missing *non-trading* days.
      - Inserts the current trading day.
      - Flushes the batch to Postgres whenever it reaches BATCH_SIZE.

    Parameters
    ----------
    conn : psycopg2.extensions.connection
        Open connection (transaction controlled by caller).
    nyse_cal : pandas.DataFrame
        The exchange schedule. Must have DatetimeIndex and 'open'/'close' columns.

    Returns
    -------
    list[TradingCalendarRow]
        The final partially filled batch (tail) to be flushed by the caller.

    Notes
    -----
    - Gaps are computed on civil dates (date part), not timestamps.
    - Days *between* consecutive trading dates are inserted as non-trading days.
    - The caller is responsible for flushing the returned tail and committing.
    """
    curr_batch_idx: int = 0
    batch: List[TradingCalendarRow | None] = [None for _ in range(BATCH_SIZE)]
    prev_day: datetime.date = nyse_cal.index[0].date()
    for day in nyse_cal.index:
        time_delta: datetime.timedelta = day.date() - prev_day
        if curr_batch_idx == BATCH_SIZE:
            execute_batch(conn, batch)
            curr_batch_idx = 0
        if time_delta.days > 1:
            gap_days = [
                prev_day + datetime.timedelta(days=i) for i in range(1, time_delta.days)
            ]
            curr_batch_idx = (
                handle_gap_days(gap_days, batch, conn, curr_batch_idx) % BATCH_SIZE
            )
        batch[curr_batch_idx] = extract_row(nyse_cal.loc[day])
        curr_batch_idx += 1
        prev_day = day.date()
    return batch[:curr_batch_idx]


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
    # Fix typing
    if not isinstance(row.name, pd.Timestamp):
        raise TypeError("Row index must be a pd.Timestamp")
    trading_day: datetime.date = row.name.date()
    session_open_utc: datetime.datetime = row["open"]
    session_close_utc: datetime.datetime = row["close"]
    is_trading_day: bool = not pd.isna(session_open_utc) and not pd.isna(
        session_close_utc
    )
    is_half_day: bool = False
    if is_trading_day:
        curr_day_intra_seconds: float = (
            session_close_utc - session_open_utc
        ).total_seconds()
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


def execute_batch(conn: connection, batch: List[TradingCalendarRow | None]) -> None:
    """
    Upsert a batch of rows into `trading_calendar`.

    Parameters
    ----------
    conn : psycopg2.extensions.connection
        Open connection; cursor is created and closed inside.
    batch : list[TradingCalendarRow]
        Batch of tuples to insert. Must contain only fully populated rows.

    Behavior
    --------
    - Uses `psycopg2.extras.execute_values` to send a multi-VALUES insert.
    - Conflicts on primary key (trading_day) are resolved with an UPDATE.

    Raises
    ------
    psycopg2.DatabaseError
        On SQL or connectivity errors (transaction is controlled by caller).
    """
    with conn.cursor() as cur:
        input_query = """
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
        psycopg2.extras.execute_values(cur, input_query, batch)


def handle_gap_days(
    non_trading_days: List[datetime.date],
    batch: List[TradingCalendarRow | None],
    conn: connection,
    curr_batch_idx: int,
) -> int:
    """
    Append missing *non-trading* civil dates into the batch, flushing as needed.

    Parameters
    ----------
    non_trading_days : list[date]
        Civil dates strictly between two consecutive trading dates.
    batch : list[TradingCalendarRow]
        Fixed-size batch buffer to fill.
    conn : psycopg2.extensions.connection
        Open connection for flushing full batches.
    curr_buff_idx : int
        Current write index within the batch.

    Returns
    -------
    int
        Updated write index after inserting the gap days.

    Notes
    -----
    - For non-trading days, open/close are NULL and `is_trading_day=False`.
    - Function may flush a full batch and reset index to 0.
    """
    for row in non_trading_days:
        if curr_batch_idx == BATCH_SIZE:
            execute_batch(conn, batch)
            curr_batch_idx = 0
        batch[curr_batch_idx] = extract_non_trading_row(row)
        curr_batch_idx += 1
    return curr_batch_idx


def extract_non_trading_row(date: datetime.date) -> tuple:
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
    trading_day: datetime.date = date
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
