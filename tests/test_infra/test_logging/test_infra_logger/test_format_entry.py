"""
Purpose
-------
Validate `InfraLogger.format_entry` across all formatting paths:
- JSON success
- JSON fallback on serialization error
- Human-readable text formatting

Key behaviors
-------------
- Asserts the JSON path delegates to `json.dumps`.
- On JSON serialization error, confirms fallback call with `default=str` and
  the returned fallback value is surfaced.
- In text mode, verifies `json.dumps` is not called and the exact one-line
  string matches the centralized expectation.

Conventions
-----------
- `json.dumps` is patched at 'infra.logging.infra_logger.json.dumps'.
- Expected text is imported as `TEST_FORMATTED_TEXT` from test utils for a single
  source of truth.
- No real I/O or time access is performed.

Downstream usage
----------------
- Run locally with PyTest: `pytest -q`
- CI picks this up via `[tool.pytest.ini_options].testpaths = ["tests"]`.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from infra.logging.infra_logger import InfraLogger, LogEntry
from tests.test_infra.test_logging.test_infra_logger.infra_logger_testing_utils import (
    TEST_FORMATTED_TEXT,
    create_test_entry,
    init_logger_for_test,
)

FALL_BACK_STRING = "<FALLBACK>"


def test_infra_logger_format_entry_json_happy(mocker: MockerFixture) -> None:
    """
    Verify the JSON formatting path on success.

    Parameters
    ----------
    mocker : MockerFixture
        Fixture used to patch `infra.logging.infra_logger.json.dumps`.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If `json.dumps` is not called exactly once.

    Notes
    -----
    None

    """

    logger: InfraLogger = init_logger_for_test()
    mock_json: MagicMock = mock_json_dumps(mocker)
    test_entry: LogEntry = create_test_entry()
    logger.format_entry(test_entry)
    mock_json.assert_called_once()


@pytest.mark.parametrize("error_type", [TypeError("boom"), ValueError("kaboom")])
def test_infra_logger_format_entry_json_erroneous(
    mocker: MockerFixture, error_type: TypeError | ValueError
) -> None:
    """
    Verify JSON fallback behavior when serialization fails.

    Parameters
    ----------
    mocker : MockerFixture
        Fixture used to patch `infra.logging.infra_logger.json.dumps` with a
        side effect that raises on the first call and returns a sentinel on the second.
    error_type : TypeError | ValueError
        The exception type to raise on the first call to `json.dumps`. Both types
        are exercised since the implementation handles either.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If two calls are not observed, if the second call lacks `default=str`,
        or if the returned value is not the sentinel.

    Notes
    -----
    - This models a non-JSON-serializable entry member triggering an error.
    """
    logger: InfraLogger = init_logger_for_test()
    mock_json: MagicMock = mock_json_dumps(mocker, [error_type, FALL_BACK_STRING])
    test_entry: LogEntry = create_test_entry()
    result: str = logger.format_entry(test_entry)
    assert mock_json.call_count == 2
    second_call_kwargs = mock_json.call_args_list[1].kwargs
    assert second_call_kwargs.get("default") is str
    assert result == FALL_BACK_STRING


def test_infra_logger_format_entry_text(mocker: MockerFixture) -> None:
    """
    Verify the human-readable text formatting path.

    Parameters
    ----------
    mocker : MockerFixture
        Provided for consistency; no patches are required in this test.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If `json.dumps` is called, or if the returned string does not match
        `TEST_FORMATTED_TEXT`.

    Notes
    -----
    - The expected string is centralized in test utils to keep assertions stable
      and avoid duplication.
    """
    logger: InfraLogger = init_logger_for_test(log_format="text")
    mock_json: MagicMock = mock_json_dumps(mocker)
    test_entry: LogEntry = create_test_entry()
    result: str = logger.format_entry(test_entry)
    mock_json.assert_not_called()
    assert result == TEST_FORMATTED_TEXT


def mock_json_dumps(mocker: MockerFixture, side_effect: Any = None) -> MagicMock:
    """
    Patch `infra.logging.infra_logger.json.dumps` for a test case.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest-mock fixture.
    side_effect : Any, optional
        If provided, used as the `side_effect` for the patched function. This may be a single
        exception/value or a list to sequence multiple behaviors
        (e.g., [TypeError(), "<FALLBACK>"]).

    Returns
    -------
    MagicMock
        The patched `json.dumps` mock for call/argument assertions.

    Raises
    ------
    None

    Notes
    -----
    - When `side_effect` is None, the patched function returns a `MagicMock` value; tests that
      only care about call-count/dispatch can rely on that without asserting specific return data.
    """

    return mocker.patch("infra.logging.infra_logger.json.dumps", side_effect=side_effect)
