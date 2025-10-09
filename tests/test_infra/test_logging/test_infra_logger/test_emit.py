"""
Purpose
-------
Validate the `InfraLogger.emit` behavior in isolation. Ensures level thresholding
(below-threshold fast exit), field normalization for missing `msg`/`context`,
and the happy-path wiring from `emit` → `format_entry` → `write_entry`.

Key behaviors
-------------
- Verifies that below-threshold events produce no timestamping, formatting, or writes.
- Confirms `msg=None`→"" and `context=None`→{} normalization is applied before formatting.
- Asserts a single formatted line is handed off to `write_entry` for at/above-threshold events.

Conventions
-----------
- Time is patched at the module symbol (`infra.logging.infra_logger.dt.datetime`) and
  expected to be called with `timezone.utc`.
- Formatting is stubbed to return a fixed sentinel string; no real serialization is tested here.
- I/O is not performed: `write_entry` is stubbed to avoid touching stderr or the filesystem.

Downstream usage
----------------
- Run locally with PyTest: `pytest -q`; these tests are hermetic (time/IO patched).
- In CI (GitHub Actions), PyTest runs with: pytest -q tests/test_infra/test_utils.
"""

import datetime as dt
from typing import List
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from infra.logging.infra_logger import InfraLogger
from tests.test_infra.test_logging.test_infra_logger.infra_logger_testing_utils import (
    TEST_CONTEXT,
    TEST_EVENT,
    TEST_FORMATTED_ENTRY,
    TEST_MESSAGE,
    Context,
    init_logger_for_test,
    mock_datetime_now,
)

LOWER_LEVEL_TUPLES: List[tuple[str, str]] = [
    ("INFO", "DEBUG"),
    ("WARNING", "INFO"),
    ("WARNING", "DEBUG"),
    ("ERROR", "WARNING"),
    ("ERROR", "INFO"),
    ("ERROR", "DEBUG"),
]

TEST_MESSAGE_CONTEXT_TUPLES: List[tuple[str | None, str, Context | None, Context]] = [
    (None, "", None, {}),
    (TEST_MESSAGE, TEST_MESSAGE, None, {}),
    (None, "", TEST_CONTEXT, TEST_CONTEXT),
    (TEST_MESSAGE, TEST_MESSAGE, TEST_CONTEXT, TEST_CONTEXT),
]


@pytest.mark.parametrize("log_level, message_level", LOWER_LEVEL_TUPLES)
def test_infra_logger_emit_lower_level(
    mocker: MockerFixture,
    log_level: str,
    message_level: str,
) -> None:
    """
    Verify that below-threshold events are dropped without side effects.

    Parameters
    ----------
    mocker : MockerFixture
        pytest-mock fixture used to patch `format_entry`, `write_entry`, and time.
    log_level : str
        Logger's configured minimum level (e.g., "INFO", "WARNING", "ERROR").
    message_level : str
        Event level that is strictly lower than `log_level`.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If `datetime.now`, `format_entry`, or `write_entry` are called for a dropped event.

    Notes
    -----
    - This test asserts the "fast exit" contract: `emit` returns before timestamping
    or any further processing when the event level is below the threshold.
    """

    # Mocking and setup
    logger: InfraLogger = init_logger_for_test(log_level=log_level)
    mock_format_entry, mock_write_entry = mock_format_and_write_entry(mocker, logger)
    mock_dt: MagicMock = mock_datetime_now(mocker)
    logger.emit(TEST_EVENT, message_level, TEST_MESSAGE, TEST_CONTEXT)

    # Method level assertions
    mock_dt.now.assert_not_called()
    mock_format_entry.assert_not_called()
    mock_write_entry.assert_not_called()


@pytest.mark.parametrize(
    "input_message,expected_message,input_context,expected_context", TEST_MESSAGE_CONTEXT_TUPLES
)
def test_infra_logger_emit_correct_level(
    mocker: MockerFixture,
    input_message: str | None,
    expected_message: str,
    input_context: Context | None,
    expected_context: dict,
) -> None:
    """
    Validate normalization and wiring for at/above-threshold events.

    Parameters
    ----------
    mocker : MockerFixture
        pytest-mock fixture used to patch collaborators.
    input_message : str or None
        Incoming message argument to `emit`.
    expected_message : str
        Expected normalized message written into the entry ("" when input is None).
    input_context : dict or None
        Incoming context argument to `emit`.
    expected_context : dict
        Expected normalized context written into the entry ({} when input is None).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If time is not called with UTC, if formatting is not invoked exactly once,
        if the normalized fields differ from expectations, or if `write_entry`
        does not receive the sentinel string.

    Notes
    -----
    - The test stubs `format_entry` to a constant sentinel and asserts `write_entry`
    receives it exactly once, proving the emit→format→write flow.
    - Normalization is asserted by inspecting the `entry` dict passed to `format_entry`.
    """

    # Mocking and setup
    logger: InfraLogger = init_logger_for_test()
    mock_format_entry, mock_write_entry = mock_format_and_write_entry(mocker, logger)
    mock_dt: MagicMock = mock_datetime_now(mocker)
    logger.emit(TEST_EVENT, "INFO", input_message, input_context)

    # Method level assertions
    mock_dt.now.assert_called_once_with(dt.timezone.utc)
    mock_format_entry.assert_called_once()
    mock_write_entry.assert_called_once_with(TEST_FORMATTED_ENTRY)

    # Entry content assertions
    mock_entry: dict = mock_format_entry.call_args[0][0]
    assert mock_entry["level"] == "INFO"
    assert mock_entry["event"] == TEST_EVENT
    assert mock_entry["message"] == expected_message
    assert mock_entry["context"] == expected_context


def mock_format_and_write_entry(
    mocker: MockerFixture,
    infra_logger: InfraLogger,
) -> tuple[MagicMock, MagicMock]:
    """
    Stub `format_entry` and `write_entry` on a logger instance.

    Parameters
    ----------
    mocker : MockerFixture
        pytest-mock fixture used to patch instance methods.
    infra_logger : InfraLogger
        The logger under test whose methods are patched.

    Returns
    -------
    tuple[MagicMock, MagicMock]
        (mock_format_entry, mock_write_entry), allowing assertions on calls and arguments.

    Raises
    ------
    None

    Notes
    -----
    - `format_entry` is stubbed, not wrapped, to isolate `emit` from formatting logic here.
    - `write_entry` is stubbed to avoid any real I/O; tests assert it receives the sentinel.
    """

    mock_format_entry: MagicMock = mocker.patch.object(
        infra_logger, "format_entry", return_value=TEST_FORMATTED_ENTRY, spec_set=True
    )
    mock_write_entry: MagicMock = mocker.patch.object(infra_logger, "write_entry")
    return mock_format_entry, mock_write_entry
