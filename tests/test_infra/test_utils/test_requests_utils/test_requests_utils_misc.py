"""
Purpose
-------
Misc tests for request helpers: header construction and retry sleep control.
Exercises `create_header` and `check_response` without real I/O.

Key behaviors
-------------
- `create_header`: builds headers from env; optional JSON Accept; fails without USER_AGENT.
- `check_response`: computes exponential backoff, delegates status handling
  and skips sleeping on the final attempt.

Conventions
-----------
- Environment variables are isolated with `monkeypatch`.
- Sleep and jitter are patched in the module namespace: `time.sleep`, `random.uniform`.
- `handle_status_code` is wrapped (spy) to verify delegation.

Downstream usage
----------------
Run with pytest. Treat these tests as the specification for header policy and
retry/backoff timing semantics in the transport layer.
"""

from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch
from pytest_mock import MockerFixture

from infra.utils.requests_utils import check_response, create_header
from infra.utils.requests_utils import handle_status_code as real_handle_status_code

TEST_AGENT: str = "test-agent"
EXPECTED_ACCEPT: str = "application/json; charset=utf-8"
TEST_MAX_RETRIES: int = 5
TEST_BACKOFF_FACTOR: float = 0.5
TEST_STATUS_CODE: int = 500


def test_create_header_happy_no_json(monkeypatch: MonkeyPatch) -> None:
    """
    Builds headers without JSON Accept when `expect_json=False`.

    Parameters
    ----------
    monkeypatch : MonkeyPatch
        Used to seed USER_AGENT for the duration of the test.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the header dict does not match the exact expected keys/values.
    """

    monkeypatch.setenv("USER_AGENT", TEST_AGENT)
    header = create_header(expect_json=False)
    assert header == {"User-Agent": TEST_AGENT}


def test_create_header_happy_json(monkeypatch: MonkeyPatch) -> None:
    """
    Builds headers with JSON Accept when `expect_json=True`.

    Parameters
    ----------
    monkeypatch : MonkeyPatch
        Used to seed USER_AGENT for the duration of the test.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the header dict does not match the exact expected keys/values.
    """

    monkeypatch.setenv("USER_AGENT", TEST_AGENT)
    header = create_header(expect_json=True)
    assert header == {"User-Agent": TEST_AGENT, "Accept": EXPECTED_ACCEPT}


def test_create_header_missing_user_agent(monkeypatch: MonkeyPatch) -> None:
    """
    Raises KeyError when USER_AGENT is absent.

    Parameters
    ----------
    monkeypatch : MonkeyPatch
        Used to remove USER_AGENT from the environment.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        Expected when USER_AGENT is not set.
    """

    monkeypatch.delenv("USER_AGENT", raising=False)
    with pytest.raises(KeyError):
        create_header()


def test_check_response(mocker: MockerFixture) -> None:
    """
    Sleeps using exponential backoff for retryable failures with a response.

    Parameters
    ----------
    mocker : MockerFixture
        Used to patch module-level `time.sleep`, `random.uniform`,
        and wrap `handle_status_code`.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If sleep duration, jitter call, or `handle_status_code` delegation
        do not match expectations.
    """

    mock_sleep, mock_uniform, mock_response, mock_handle_status_code = init_check_response_test(
        mocker
    )
    test_attempt: int = 1
    check_response(
        mock_response,
        attempt=test_attempt,
        max_retries=TEST_MAX_RETRIES,
        backoff_factor=TEST_BACKOFF_FACTOR,
    )
    mock_sleep.assert_called_once_with(TEST_BACKOFF_FACTOR * (2**test_attempt))
    mock_uniform.assert_called_once()
    mock_handle_status_code.assert_called_once_with(
        TEST_STATUS_CODE, {}, test_attempt, TEST_BACKOFF_FACTOR
    )


def test_check_response_none(mocker: MockerFixture) -> None:
    """
    Sleeps using exponential backoff for network-level errors (no response).

    Parameters
    ----------
    mocker : MockerFixture
        Used to patch module-level `time.sleep` and `random.uniform`.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the computed sleep is incorrect.
    """

    mock_sleep, mock_uniform = mock_sleep_and_random(mocker)
    test_attempt: int = 2
    check_response(
        None, attempt=test_attempt, max_retries=TEST_MAX_RETRIES, backoff_factor=TEST_BACKOFF_FACTOR
    )
    mock_sleep.assert_called_once_with(TEST_BACKOFF_FACTOR * (2**test_attempt))
    mock_uniform.assert_not_called()


def test_check_response_last_attempt(mocker: MockerFixture) -> None:
    """
    Does not sleep on the final attempt; still delegates to `handle_status_code`.

    Parameters
    ----------
    mocker : MockerFixture
        Used to patch module-level `time.sleep`/`random.uniform` and wrap `handle_status_code`.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If sleep occurs on the last attempt, or delegation is missing.
    """

    mock_sleep, mock_uniform, mock_response, mock_handle_status_code = init_check_response_test(
        mocker
    )
    test_attempt: int = TEST_MAX_RETRIES - 1
    check_response(
        mock_response,
        attempt=test_attempt,
        max_retries=TEST_MAX_RETRIES,
        backoff_factor=TEST_BACKOFF_FACTOR,
    )
    mock_sleep.assert_not_called()
    mock_uniform.assert_called_once()
    mock_handle_status_code.assert_called_once_with(
        TEST_STATUS_CODE, {}, test_attempt, TEST_BACKOFF_FACTOR
    )


def init_check_response_test(
    mocker: MockerFixture,
) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
    """
    Initialize common patches and fixtures for `check_response` tests.

    Parameters
    ----------
    mocker : MockerFixture
        pytest-mock fixture used to apply patches within the
        `infra.utils.requests_utils` module namespace.

    Returns
    -------
    tuple[MagicMock, MagicMock, MagicMock, MagicMock]
        A 4-tuple:
        - mock_sleep : MagicMock
            Patched `time.sleep` in `infra.utils.requests_utils` (no-op).
        - mock_uniform : MagicMock
            Patched `random.uniform` in `infra.utils.requests_utils` (fixed to 1.0).
        - mock_response : MagicMock
            Minimal Response-like mock with `status_code=TEST_STATUS_CODE`
            and empty `headers` dict.
        - mock_handle_status_code : MagicMock
            Spy wrapping the real `handle_status_code` to assert delegation
            while preserving behavior.

    Notes
    -----
    - Jitter is fixed to 1.0 so expected sleep equals the base backoff.
    """

    mock_sleep, mock_uniform = mock_sleep_and_random(mocker)
    mock_response: MagicMock = MagicMock()
    mock_response.status_code = TEST_STATUS_CODE
    mock_response.headers = {}
    mock_handle_status_code: MagicMock = mocker.patch(
        "infra.utils.requests_utils.handle_status_code", wraps=real_handle_status_code
    )
    return mock_sleep, mock_uniform, mock_response, mock_handle_status_code


def mock_sleep_and_random(mocker: MockerFixture) -> tuple[MagicMock, MagicMock]:
    """
    Patch `time.sleep` and `random.uniform` in the requests_utils module and return the mocks.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest-mock fixture to apply patches.

    Returns
    -------
    tuple[MagicMock, MagicMock]
        The mocked `time.sleep` and `random.uniform` callables for assertions.

    Notes
    -----
    - Jitter is fixed to 1.0 so expected sleep equals the base backoff.
    """

    mock_sleep = mocker.patch("infra.utils.requests_utils.time.sleep", return_value=None)
    mock_uniform = mocker.patch("infra.utils.requests_utils.random.uniform", return_value=1.0)
    return mock_sleep, mock_uniform
