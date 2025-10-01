"""
db_utils.py

Utility functions for database connectivity.

Purpose
-------
Centralizes PostgreSQL connection logic so it can be reused across all
ingestion scripts (e.g., trading calendar, security master, prices).
Encapsulates environment variable handling and psycopg2 connection setup.

Conventions
-----------
- Reads credentials from environment variables:
    POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD,
    DB_HOST, DB_PORT.
- Returns a live psycopg2 connection; caller is responsible for closing it.
- Raises KeyError if any required variable is missing.
- Raises psycopg2.OperationalError if the connection fails.

Downstream usage
----------------
Import `connect_to_db()` from this module in any ingestion script
to open a Postgres session within a context that handles
transactions and cleanup explicitly.
"""
import os
from typing import Callable, Iterable, List, Optional

import pandas as pd
import datetime as dt
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.extensions import connection


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


def load_into_table(conn: connection, row_generator: Iterable[tuple], input_query: str) -> None:
    batch: List[tuple | None] = [None] * BATCH_SIZE
    index = 0
    for row in row_generator:
        batch[index] = row
        index += 1
        if index == BATCH_SIZE:
            execute_batch(conn, batch, input_query)
            index = 0
    if index > 0:
        execute_batch(conn, batch[:index], input_query)


def str_to_timestamp(date_str: str) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.strptime(date_str, '%Y-%m-%d'), tz="UTC")


def execute_batch(conn: connection, batch: List, input_query: str) -> None:
    with conn.cursor() as cur:
        execute_values(cur, input_query, batch)


def process_chunk(
        chunk: pd.Series,
        api_key: str,
        url: str,
        column_names: List[str],
        processor: Optional[Callable] = None
    ) -> pd.DataFrame:
    profiles: pd.DataFrame = pd.read_json(
        f"{url}{','.join(chunk)}?apikey={api_key}"
        )[column_names]
    if processor:
        profiles = processor(profiles)
    return profiles
