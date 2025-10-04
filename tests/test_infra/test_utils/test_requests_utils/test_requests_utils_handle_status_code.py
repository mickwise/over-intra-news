"""
Purpose
-------
Unit tests for `handle_status_code`, covering retryable 5xx paths, 429 with/without
`Retry-After`, and non-retryable 4xx behavior. Ensures backoff math, jitter usage,
and `Retry-After` delegation are correct and deterministic.

Key behaviors
-------------
- 5xx → exponential backoff with jitter (jitter mocked to 1.0).
- 429 + valid `Retry-After` → returns parsed delay (no jitter fallback).
- 429 + missing/invalid `Retry-After` → falls back to exponential backoff (with jitter).
- 4xx non-retryable → raises `requests.HTTPError`.

Conventions
-----------
- `random.uniform` is patched in the module namespace to return 1.0 for determinism.
- `extract_retry_after` is patched where needed to isolate `handle_status_code`.
- Expected sleep values are computed from `backoff_factor * (2 ** attempt)`.

Downstream usage
----------------
Treat these tests as a specification for the transport layer’s retry timing policy.
Any adjustment to jitter strategy or `Retry-After` handling should update these tests.
"""

from typing import Any, List
from unittest.mock import MagicMock

import pytest
import requests
from pytest_mock import MockerFixture

from infra.utils.requests_utils import handle_status_code

RETRYABLE_STATUS_CODES: set[int] = {500, 502, 503, 504}
NON_RETRYABLE_STATUS_CODES: set[int] = {400, 401, 403, 404, 418}
TEST_ATTEMPT: int = 2
TEST_BACKOFF_FACTOR: float = 0.5
EXPECTED_SLEEP_TIME: float = TEST_BACKOFF_FACTOR * (2**TEST_ATTEMPT)
TEST_RETRY_AFTER: str = "5.0"
TEST_CORRECT_HEADER: dict[str, str] = {"Retry-After": TEST_RETRY_AFTER}
TEST_NON_STRING_HEADERS: List[dict[str, Any]] = [
    {},
    {"Retry-After": None},
    {"Retry-After": 10},
    {"Retry-After": 3.5},
]


@pytest.mark.parametrize("status_code", RETRYABLE_STATUS_CODES)
def test_handle_status_code_retryable(mocker: MockerFixture, status_code: int) -> None:
    """
    Asserts that retryable 5xx statuses use exponential backoff (with jitter=1.0).

    Parameters
    ----------
    mocker : MockerFixture
        Used to patch `random.uniform` and `extract_retry_after`.
    status_code : int
        One of {500, 502, 503, 504}.

    Returns
    -------
    None

    Notes
    -----
    - Verifies returned sleep equals `backoff_factor * (2 ** attempt)` and
      that jitter was invoked once.
    """

    mock_retry_after, mock_uniform = mock_extract_retry_after_and_uniform(mocker, 0.0)
    sleep_time: float = handle_status_code(status_code, {}, TEST_ATTEMPT, TEST_BACKOFF_FACTOR)
    general_asserts(mock_retry_after, mock_uniform, EXPECTED_SLEEP_TIME, sleep_time)


def test_handle_status_code_retry_after_with_header(mocker: MockerFixture) -> None:
    """
    Asserts that 429 with a valid `Retry-After` header returns that exact delay.

    Parameters
    ----------
    mocker : MockerFixture
        Used to patch `random.uniform` and `extract_retry_after`.

    Returns
    -------
    None

    Notes
    -----
    - Confirms `extract_retry_after` is called once and its result is returned.
    - Jitter is still called (computed once upstream) but not used to override
      `Retry-After`.
    """

    mock_retry_after, mock_uniform = mock_extract_retry_after_and_uniform(
        mocker, float(TEST_RETRY_AFTER)
    )
    sleep_time: float = handle_status_code(
        429, TEST_CORRECT_HEADER, TEST_ATTEMPT, TEST_BACKOFF_FACTOR
    )
    general_asserts(mock_retry_after, mock_uniform, float(TEST_RETRY_AFTER), sleep_time, True)


@pytest.mark.parametrize("non_string_header", TEST_NON_STRING_HEADERS)
def test_handle_status_code_retry_after_non_string_header(
    mocker: MockerFixture,
    non_string_header: dict[str, Any],
) -> None:
    """
    Asserts that 429 a non string `Retry-After` falls back to exponential backoff.

    Parameters
    ----------
    mocker : MockerFixture
        Used to patch `random.uniform` and `extract_retry_after`.

    Returns
    -------
    None

    Notes
    -----
    - Verifies `extract_retry_after` is not called.
    - Ensures fallback sleep equals `backoff_factor * (2 ** attempt)`.
    """

    mock_extract_retry_after, mock_uniform = mock_extract_retry_after_and_uniform(mocker, 0.0)
    sleep_time: float = handle_status_code(
        429, non_string_header, TEST_ATTEMPT, TEST_BACKOFF_FACTOR
    )
    general_asserts(mock_extract_retry_after, mock_uniform, EXPECTED_SLEEP_TIME, sleep_time)


@pytest.mark.parametrize("status_code", NON_RETRYABLE_STATUS_CODES)
def test_handle_status_code_non_retryable(mocker: MockerFixture, status_code: int) -> None:
    """
    Asserts that non-retryable 4xx statuses raise `requests.HTTPError`.

    Parameters
    ----------
    mocker : MockerFixture
        Used to patch `random.uniform` for deterministic call counting.
    status_code : int
        One of {400, 401, 403, 404, 418}.

    Returns
    -------
    None

    Raises
    ------
    requests.HTTPError
        Always, for non-retryable statuses.

    Notes
    -----
    - Confirms jitter was computed once, matching the function’s “compute-once”
      pattern even when raising.
    """

    _, mock_uniform = mock_extract_retry_after_and_uniform(mocker, 0.0)
    with pytest.raises(requests.HTTPError):
        handle_status_code(status_code, {}, TEST_ATTEMPT, TEST_BACKOFF_FACTOR)
    mock_uniform.assert_called_once()


def mock_extract_retry_after_and_uniform(
    mocker: MockerFixture, retry_after_return: float
) -> tuple[MagicMock, MagicMock]:
    """
    Patch helpers for isolating `handle_status_code` timing logic.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest-mock fixture to apply patches.
    retry_after_return : float
        Value that the patched `extract_retry_after` should return.

    Returns
    -------
    tuple[MagicMock, MagicMock]
        (mock_extract_retry_after, mock_uniform) for assertions.

    Notes
    -----
    - `random.uniform` is patched to return 1.0 for deterministic backoff.
    - `extract_retry_after` is patched to a known value to avoid date parsing.
    """

    mock_extract_retry_after: MagicMock = mocker.patch(
        "infra.utils.requests_utils.extract_retry_after", return_value=retry_after_return
    )
    mock_uniform: MagicMock = mocker.patch(
        "infra.utils.requests_utils.random.uniform", return_value=1.0
    )
    return mock_extract_retry_after, mock_uniform


def general_asserts(
    mock_extract_retry_after: MagicMock,
    mock_uniform: MagicMock,
    expected_sleep_time: float,
    actual_sleep_time: float,
    retry_branch: bool = False,
) -> None:
    """
    Shared assertions for sleep value and collaborator call counts.

    Parameters
    ----------
    mock_extract_retry_after : MagicMock
        The patched `extract_retry_after` mock.
    mock_uniform : MagicMock
        The patched `random.uniform` mock.
    expected_sleep_time : float
        The expected number of seconds to sleep.
    actual_sleep_time : float
        The value returned from `handle_status_code`.
    retry_branch : bool, default=False
        True when testing the 429+header path that must call `extract_retry_after`.

    Returns
    -------
    None

    Notes
    -----
    - Always asserts jitter was computed once.
    - Asserts `extract_retry_after` was called iff `retry_branch` is True.
    """

    assert actual_sleep_time == expected_sleep_time
    mock_uniform.assert_called_once()
    if retry_branch:
        mock_extract_retry_after.assert_called_once()
    else:
        mock_extract_retry_after.assert_not_called()
