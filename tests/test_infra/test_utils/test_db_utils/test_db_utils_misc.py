"""
Purpose
-------
Unit tests for `infra.utils.db_utils` functions that handle DB connectivity
and timestamp parsing. Tests ensure predictable behavior without relying
on a real Postgres instance.

Key behaviors
-------------
- Verifies `connect_to_db` constructs a psycopg2 connection from environment
  variables and propagates missing-variable errors.
- Ensures `str_to_timestamp` parses valid ISO date strings into UTC timestamps
  and rejects invalid formats.

Conventions
-----------
- `psycopg2.connect` is always patched; returned connection is a MagicMock.
- Environment variables are seeded explicitly in tests for reproducibility.
- Uses pytest parameterization to cover missing-variable cases.
- Timezone in timestamp parsing is always UTC.

Downstream usage
----------------
Guards assumptions that ingestion scripts can rely on environment-driven
DB wiring and strict ISO timestamp parsing, without involving external systems.
"""

from typing import List
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pytest import MonkeyPatch
from pytest_mock import MockerFixture

from infra.utils.db_utils import connect_to_db, str_to_timestamp

TEST_HOST: str = "localhost"
TEST_PORT: str = "5432"
TEST_DBNAME: str = "testdb"
TEST_USER: str = "user"
TEST_PASSWORD: str = "password"
ENV_VARS: List[str] = ["DB_HOST", "DB_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"]


def test_connect_to_db_happy(mocker: MockerFixture, monkeypatch: MonkeyPatch) -> None:
    """
    Test that `connect_to_db` successfully constructs a psycopg2 connection
    when all required environment variables are present.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest-mock fixture to patch `psycopg2.connect`.

    Raises
    ------
    AssertionError
        If the returned connection is None or psycopg2.connect was not called
        with the expected keyword arguments.
    """

    mock_conn = set_up_env_and_conn(mocker, monkeypatch)

    conn = connect_to_db()
    assert conn is mock_conn.return_value
    mock_conn.assert_called_once_with(
        host=TEST_HOST,
        port=TEST_PORT,
        dbname=TEST_DBNAME,
        user=TEST_USER,
        password=TEST_PASSWORD,
    )


@pytest.mark.parametrize("missing_var", ENV_VARS)
def test_connect_to_db_missing_env_var(
    mocker: MockerFixture, missing_var: str, monkeypatch: MonkeyPatch
) -> None:
    """
    Test that `connect_to_db` raises KeyError if any required environment
    variable is missing.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest-mock fixture to patch `psycopg2.connect`.
    missing_var : str
        Environment variable removed during the test.

    Raises
    ------
    KeyError
        When the specific environment variable is missing.
    """

    set_up_env_and_conn(mocker, monkeypatch)
    monkeypatch.delenv(missing_var, raising=True)
    with pytest.raises(KeyError):
        connect_to_db()


def set_up_env_and_conn(mocker: MockerFixture, monkeypatch: MonkeyPatch) -> MagicMock:
    """
    Helper to seed required environment variables and patch psycopg2.connect.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest-mock fixture to patch psycopg2 internals.

    Returns
    -------
    MagicMock
        The patched `psycopg2.connect` mock.
    """

    mock_conn: MagicMock = mocker.patch(
        "infra.utils.db_utils.psycopg2.connect", return_value=mocker.MagicMock()
    )
    monkeypatch.setenv("DB_HOST", TEST_HOST)
    monkeypatch.setenv("DB_PORT", TEST_PORT)
    monkeypatch.setenv("POSTGRES_DB", TEST_DBNAME)
    monkeypatch.setenv("POSTGRES_USER", TEST_USER)
    monkeypatch.setenv("POSTGRES_PASSWORD", TEST_PASSWORD)
    return mock_conn


def test_str_to_timestamp() -> None:
    """
    Test that `str_to_timestamp` converts valid ISO "YYYY-MM-DD" strings into
    UTC pandas.Timestamp values and raises ValueError for invalid formats.

    Raises
    ------
    AssertionError
        If valid dates are parsed incorrectly.
    ValueError
        If invalid formats (e.g., non-ISO) or an invalid
        date is passed.
    """

    # Correct format
    assert str_to_timestamp("2023-10-01") == pd.Timestamp("2023-10-01 00:00:00", tz="UTC")

    # Leap year
    assert str_to_timestamp("2020-02-29") == pd.Timestamp("2020-02-29 00:00:00", tz="UTC")

    # Invalid format
    with pytest.raises(ValueError):
        str_to_timestamp("2023/10/01")

    # Invalid date
    with pytest.raises(ValueError):
        str_to_timestamp("2023-02-30")
