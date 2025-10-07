"""
Purpose
-------
Unit tests for `infra.utils.db_utils.execute_batch`. Ensures correct cursor usage,
SQL execution, and error propagation when batching inserts into Postgres.

Key behaviors
-------------
- Verifies that `execute_values` is called with the expected cursor, query, and batch.
- Confirms that raised exceptions are not swallowed by the function.
- Uses mocks to isolate database behavior from real Postgres.

Conventions
-----------
- Always patches `psycopg2.extras.execute_values` to avoid I/O.
- Cursor object is provided via a MagicMock with context manager support.
- Batches are small, deterministic lists of tuples for clarity.

Downstream usage
----------------
Run automatically under pytest when validating database utilities.
No external setup (e.g., Docker Postgres) is required for these tests.
"""

from typing import List
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from infra.utils.db_utils import execute_batch


def test_execute_batch_happy(mocker: MockerFixture) -> None:
    """
    Test the happy path: `execute_batch` calls `execute_values` once
    with the correct cursor, SQL query, and batch.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest-mock fixture used to patch psycopg2 internals.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the call arguments do not match the expected cursor, query, or batch.

    Notes
    -----
    - Cursor identity is asserted with `is` to confirm the exact object is used.
    """

    mock_conn, mock_cursor, mock_execute_vals = create_mock_conn_cursor_and_patcher(mocker)
    test_query, test_batch = generate_query_and_batch()
    execute_batch(mock_conn, test_batch, test_query)
    args, _ = mock_execute_vals.call_args
    assert args[0] is mock_cursor
    assert args[1] == test_query
    assert list(args[2]) == test_batch


def test_execute_batch_erroneous(mocker: MockerFixture) -> None:
    """
    Test the erroneous path: `execute_batch` propagates exceptions raised
    by `execute_values`.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest-mock fixture used to patch psycopg2 internals.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        When `execute_values` is patched with a ValueError side effect.

    Notes
    -----
    - Confirms that `execute_batch` does not swallow database errors.
    """

    mock_conn, _, mock_execute_vals = create_mock_conn_cursor_and_patcher(mocker)
    mock_execute_vals.side_effect = ValueError("boom")
    test_query, test_batch = generate_query_and_batch()
    with pytest.raises(ValueError) as _:
        execute_batch(mock_conn, test_batch, test_query)


def create_mock_conn_cursor_and_patcher(
    mocker: MockerFixture,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    """
    Helper to create a mocked Postgres connection, cursor, and patched
    `execute_values` function.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest-mock fixture used to build MagicMocks and apply patches.

    Returns
    -------
    tuple[MagicMock, MagicMock, MagicMock]
        (mock_conn, mock_cursor, mock_execute_vals) for test use.

    Notes
    -----
    - The returned cursor supports context manager (`__enter__`, `__exit__`).
    - `execute_values` is patched in `psycopg2.extras`.
    """

    mock_conn: MagicMock = mocker.MagicMock()
    mock_cursor: MagicMock = mocker.MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.__enter__.return_value = mock_cursor
    mock_execute_vals: MagicMock = mocker.patch("infra.utils.db_utils.execute_values")
    return mock_conn, mock_cursor, mock_execute_vals


def generate_query_and_batch() -> tuple[str, List[tuple[int, str]]]:
    """
    Helper to generate a test SQL query string and a small batch of rows.

    Returns
    -------
    tuple[str, list[tuple[int, str]]]
        A parameterized INSERT query and a deterministic batch of (int, str) tuples.

    Notes
    -----
    - Used to avoid duplication across test cases.
    """

    test_query: str = "INSERT INTO test_table (col1, col2) VALUES %s ON CONFLICT DO NOTHING"
    test_batch: List[tuple[int, str]] = [(1, "a"), (2, "b"), (3, "c")]
    return test_query, test_batch
