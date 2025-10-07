"""
Purpose
-------
Unit tests for `infra.utils.db_utils.load_into_table`. Validates batching
logic and ensures rows are grouped correctly before being passed to
`execute_batch`.

Key behaviors
-------------
- Confirms that `execute_batch` is never called for an empty iterator.
- Verifies that batches are emitted with the configured batch size.
- Ensures the final batch size is correct when rows don't divide evenly.
- Asserts that the same connection and query are passed to all calls.

Conventions
-----------
- Uses `mocker.spy` to observe calls into `execute_batch`.
- Patches `BATCH_SIZE` to test different batch configurations.
- Row inputs are simple tuples for deterministic validation.

Downstream usage
----------------
Run automatically under pytest when validating database utilities.
These tests isolate logic from Postgres and never perform I/O.
"""

from math import ceil
from typing import Iterable, List
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

import infra.utils.db_utils as dbu
from infra.utils.db_utils import load_into_table


def test_load_into_table_empty_iter(mocker: MockerFixture) -> None:
    """
    Verify that `load_into_table` performs no work when given an empty iterator.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest-mock fixture used to spy on `execute_batch`.

    Raises
    ------
    AssertionError
        If `execute_batch` is called despite an empty iterator.
    """

    spy_execute_batch: MagicMock = mocker.spy(dbu, "execute_batch")
    mock_conn: MagicMock = MagicMock()
    empty_iter: Iterable = iter([])
    input_query = "INSERT INTO test_table (col) VALUES %s"
    load_into_table(mock_conn, empty_iter, input_query)
    spy_execute_batch.assert_not_called()


@pytest.mark.parametrize(
    "row_list",
    [[(1,), (2,)], [(1,), (2,), (3,)], [(i,) for i in range(10)]],
)
@pytest.mark.parametrize("batch_size", [2, 3, 5])
def test_load_into_table_row_generator(
    mocker: MockerFixture,
    row_list: List[tuple[int]],
    batch_size: int,
) -> None:
    """
    Verify that `load_into_table` correctly batches rows and calls
    `execute_batch` the expected number of times across different batch sizes.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest-mock fixture used to spy on `execute_batch`.
    row_list : list[tuple[int]]
        Input rows to feed into `load_into_table`, parameterized by pytest.
    batch_size : int
        The patched batch size to validate grouping logic, parameterized by pytest.

    Raises
    ------
    AssertionError
        If the number of calls or batch sizes differ from expectations.

    Notes
    -----
    - Expected number of calls is ceil(len(row_list) / batch_size).
    - Last batch size is `(len(row_list) % batch_size) or batch_size`.
    """

    # Set up mockers
    spy_execute_batch: MagicMock = mocker.spy(dbu, "execute_batch")
    mocker.patch("infra.utils.db_utils.BATCH_SIZE", batch_size)
    mocker.patch("infra.utils.db_utils.execute_values")
    row_amount: int = len(row_list)
    mock_conn: MagicMock = MagicMock()
    input_query = "INSERT INTO test_table (col) VALUES %s"
    load_into_table(mock_conn, iter(row_list), input_query)

    # Assertions
    assert spy_execute_batch.call_count == ceil(row_amount / batch_size)
    expected_last_batch_size = (row_amount % batch_size) or batch_size
    for i in range(spy_execute_batch.call_count):
        args, _ = spy_execute_batch.call_args_list[i]
        assert args[0] is mock_conn
        batch_arg: List[tuple[int]] = args[1]
        assert args[2] == input_query
        if i < spy_execute_batch.call_count - 1:
            assert len(batch_arg) == batch_size
        else:
            assert len(batch_arg) == expected_last_batch_size
