"""
Purpose
-------
Database utility helpers for ingestion and ETL tasks.

Key behaviors
-------------
- Opens PostgreSQL connections from environment variables.
- Streams iterable rows into batched INSERT/UPSERT calls.
- Provides strict ISO → UTC timestamp parsing.
- Fetches remote JSON into DataFrames with optional post-processing.

Conventions
-----------
- PostgreSQL credentials are read from environment variables.
- Batch loads use `psycopg2.extras.execute_values` under a cursor.
- Batching uses a fixed `BATCH_SIZE`; the final partial batch is flushed.
- All timestamps are UTC; parsing expects "YYYY-MM-DD".
- Network fetch in `process_chunk` is delegated to `pandas.read_json`.

Downstream usage
----------------
Import and use:
- `connect_to_db()` for connection setup.
- `load_into_table()` + `flush_values_batch()` to load iterables efficiently.
- `str_to_timestamp()` for consistent UTC date parsing.
- `process_chunk()` to retrieve JSON data and apply a local transformer.
"""

import datetime as dt
import os
from typing import Callable, Iterable, List

import pandas as pd
import psycopg2
from psycopg2.extensions import connection
from psycopg2.extras import execute_values

BATCH_SIZE: int = 1000
CHUNK_SIZE: int = 100


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
    DB_PORT : str
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

    Notes
    -----
    - Sets the connection timezone to UTC.
    """
    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=os.environ["DB_PORT"],
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        options="-c timezone=utc",
    )


def load_into_table(conn: connection, row_generator: Iterable[tuple], input_query: str) -> None:
    """
    Stream rows from an iterable into the database in fixed-size batches.

    Parameters
    ----------
    conn : psycopg2.extensions.connection
        Open PostgreSQL connection; caller manages commit/rollback/close.
    row_generator : Iterable[tuple]
        Iterable of value tuples matching the `input_query` VALUES template.
    input_query : str
        SQL INSERT/UPSERT statement compatible with `execute_values` (e.g., "INSERT ... VALUES %s").

    Returns
    -------
    None

    Raises
    ------
    Exception
        Propagates any database or driver errors raised by `flush_values_batch`.

    Notes
    -----
    - Uses a preallocated list of length `BATCH_SIZE` to collect rows.
    - Calls `flush_values_batch` every time the batch fills; flushes the final partial batch.
    - Optimized for large streams; avoids holding the entire dataset in memory.
    """

    batch: List[tuple | None] = [None] * BATCH_SIZE
    index = 0
    for row in row_generator:
        batch[index] = row
        index += 1
        if index == BATCH_SIZE:
            flush_values_batch(conn, batch, input_query)
            index = 0
    if index > 0:
        flush_values_batch(conn, batch[:index], input_query)


def str_to_timestamp(date_str: str) -> pd.Timestamp:
    """
    Convert an ISO "YYYY-MM-DD" string into a UTC pandas.Timestamp.

    Parameters
    ----------
    date_str : str
        Date string in "YYYY-MM-DD" format.

    Returns
    -------
    pandas.Timestamp
        UTC timestamp at 00:00:00 of the provided date.

    Raises
    ------
    ValueError
        If `date_str` is not in the expected "YYYY-MM-DD" format.

    Notes
    -----
    - Output is timezone-aware (UTC).
    - Intended for normalizing environment/config dates for queries.
    """

    return pd.Timestamp(dt.datetime.strptime(date_str, "%Y-%m-%d"), tz="UTC")


def flush_values_batch(conn: connection, batch: List, input_query: str) -> None:
    """
    Execute a single batched INSERT/UPSERT using `psycopg2.extras.execute_values`.

    Parameters
    ----------
    conn : psycopg2.extensions.connection
        Open PostgreSQL connection.
    batch : list
        List (or list-like) of value tuples to insert.
    input_query : str
        SQL statement with a `%s` placeholder for `execute_values`.

    Returns
    -------
    None

    Raises
    ------
    Exception
        Propagates database or driver errors; no internal retry logic.

    Notes
    -----
    - Opens a cursor via context manager to ensure timely cleanup.
    - Suitable for use by higher-level batching functions like `load_into_table`.
    """

    with conn.cursor() as cur:
        execute_values(cur, input_query, batch)


def process_chunk(
    chunk: pd.Series,
    api_key: str,
    url: str,
    column_names: List[str],
    processor: Callable | None = None,
) -> pd.DataFrame:
    """
    Fetch a JSON payload into a DataFrame, select columns, and optionally post-process.

    Parameters
    ----------
    chunk : pandas.Series
        Series of tokens (e.g., tickers) joined by commas into the request URL.
    api_key : str
        API key appended as a query parameter.
    url : str
        Base URL/prefix to which the comma-joined `chunk` and `apikey` are appended.
    column_names : list[str]
        Columns to select from the loaded DataFrame (order is preserved).
    processor : Callable | None, optional
        Optional callable `processor(df) -> DataFrame` applied after column selection.

    Returns
    -------
    pandas.DataFrame
        The selected (and optionally transformed) DataFrame.

    Raises
    ------
    KeyError
        If requested `column_names` are not present in the returned DataFrame.
    Exception
        Any exception raised by `pandas.read_json` or the `processor` callable.

    Notes
    -----
    - Builds the request as `f"{url}{','.join(chunk)}?apikey={api_key}"`.
    - This function does not perform retries or network error handling.
    - Keep `processor` pure (avoid side effects) for testability and reuse.
    """

    profiles: pd.DataFrame = pd.read_json(f"{url}{','.join(chunk)}?apikey={api_key}")[column_names]
    if processor:
        profiles = processor(profiles)
    return profiles
