"""
Purpose
-------
Exercise the retry/backoff control-flow of `make_request` without real I/O.
Proves that the function retries on the right failures, fails fast on the wrong
ones, forwards parameters correctly, and respects `max_retries` semantics.

Key behaviors
-------------
- Happy path: returns the response, no backoff invoked.
- Retryables: 429/5xx and network-level exceptions are retried with the final
  exception re-raised on exhaustion.
- Non-retryables: 4xx (e.g., 400/401/403/404/418) and non-whitelisted exceptions
  (e.g., TypeError) fail fast with no sleep.
- `max_retries=1`: a retryable failure raises immediately.
- Optional behavioral check: `check_response` can be wrapped to assert a single
  invocation on fail-fast status codes.

Conventions
-----------
- No real HTTP calls: `try_request` and `check_response` are patched.
- `wraps` is used where we want real behavior plus call counting.
- Jitter and sleep timing are not asserted in unit tests; `time.sleep` is
  patched in fail-fast paths to ensure zero calls.
- The same header object is injected via a patched `create_header`.

Downstream usage
----------------
Run with `pytest`. Treat these tests as the specification for `make_request`’s
public contract (retry policy, exception propagation, and argument forwarding).
"""

from unittest.mock import MagicMock

import pytest
import requests
from pytest_mock import MockerFixture
from requests import Response

from infra.utils.requests_utils import check_response as real_check_response
from infra.utils.requests_utils import make_request

TEST_URL: str = "https://example.com/api"
TEST_HEADER: dict[str, str] = {"User-Agent": "Moshe"}
TEST_TIMEOUT: tuple[float, float] = (3.5, 10.0)
TEST_EXPECT_JSON: bool = True
TEST_MAX_RETRIES: int = 3
RETRYABLE_STATUS_CODES: list[int] = [429, 500, 502, 503, 504]
NON_RETRYABLE_STATUS_CODES: list[int] = [400, 401, 403, 404, 418]
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    requests.Timeout,
    requests.ConnectionError,
    requests.RequestException,
    ValueError,
)


def test_make_request_happy(mocker: MockerFixture) -> None:
    """
    Happy path: returns the response on the first attempt with no backoff.

    Parameters
    ----------
    mocker : MockerFixture
        Fixture used to patch collaborators (`create_header`, `try_request`, `check_response`).

    Returns
    -------
    None
        Asserts identity of the returned response and that backoff logic is not touched.

    Raises
    ------
    AssertionError
        If `try_request` is not called exactly once with expected args or if
        `check_response` is invoked.

    Notes
    -----
    - Ensures `make_request` forwards `(url, header, timeout, expect_json)` correctly.
    """

    mock_resp: MagicMock = MagicMock()
    mock_try_request, mock_check_response = mock_requests_utils_methods(
        mocker, try_request_return=mock_resp
    )
    resp: Response = make_request(TEST_URL)
    assert resp is mock_resp
    mock_try_request.assert_called_once_with(
        TEST_URL, TEST_HEADER, TEST_TIMEOUT, TEST_EXPECT_JSON, None
    )
    mock_check_response.assert_not_called()


@pytest.mark.parametrize("status_code", RETRYABLE_STATUS_CODES)
@pytest.mark.parametrize("succeed_last", [False, True])
def test_make_request_retryable(
    mocker: MockerFixture,
    status_code: int,
    succeed_last: bool,
) -> None:
    """
    Retryable status codes: retries then either succeeds on last attempt or exhausts.

    Parameters
    ----------
    mocker : MockerFixture
        Patching utility.
    status_code : int
        One of the retryable HTTP statuses (429, 500, 502, 503, 504).
    succeed_last : bool
        If True, final attempt returns success; otherwise all attempts fail.

    Returns
    -------
    None
        Asserts correct call counts and final outcome for both branches.

    Raises
    ------
    AssertionError
        If retries/backoff counts or final outcome deviate from the contract.

    Notes
    -----
    - Drives status-driven retry path via `requests.HTTPError(response=...)`.
    """

    err_response: MagicMock = MagicMock()
    err_response.status_code = status_code
    mock_try_request, mock_check_response = mock_requests_utils_methods(
        mocker, try_request_return=err_response
    )
    if succeed_last:
        check_succeed_last(mock_try_request, mock_check_response, err_response)
    else:
        check_fail_all_retries(mock_try_request, mock_check_response, err_response)


@pytest.mark.parametrize("status_code", NON_RETRYABLE_STATUS_CODES)
def test_make_request_non_retryable(
    mocker: MockerFixture,
    status_code: int,
) -> None:
    """
    Non-retryable status codes: fail fast on first attempt without sleeping.

    Parameters
    ----------
    mocker : MockerFixture
        Patching utility (used to wrap `check_response` and patch `sleep`).
    status_code : int
        One of {400, 401, 403, 404, 418} treated as non-retryable.

    Returns
    -------
    None
        Asserts single attempt, one `check_response` call, and zero sleeps.

    Raises
    ------
    AssertionError
        If more than one attempt occurs or backoff is invoked.

    Notes
    -----
    - Uses `wraps` on `check_response` to preserve real behavior while counting calls.
    """

    err_response: MagicMock = MagicMock()
    err_response.status_code = status_code
    mock_try_request, mock_check_response = mock_requests_utils_methods(
        mocker, try_request_return=err_response, wrap_check_response=True
    )
    check_non_retryable(
        mocker,
        mock_try_request,
        mock_check_response,
        err_response,
    )


@pytest.mark.parametrize("current_exception", RETRYABLE_EXCEPTIONS)
@pytest.mark.parametrize("succeed_last", [False, True])
def test_make_request_retryable_exceptions(
    mocker: MockerFixture,
    current_exception: type[BaseException],
    succeed_last: bool,
) -> None:
    """
    Retryable exceptions: retries on network/validation errors and either succeed or exhaust.

    Parameters
    ----------
    mocker : MockerFixture
        Patching utility.
    current_exception : type[BaseException]
        Exception class among {requests.Timeout, requests.ConnectionError,
        requests.RequestException, ValueError}.
    succeed_last : bool
        If True, success on last attempt; else all attempts raise the exception.

    Returns
    -------
    None
        Asserts correct retry counts and final outcome for exception-driven path.

    Raises
    ------
    AssertionError
        If call counts or raised exception do not match expectations.

    Notes
    -----
    - Drives exception path by setting `mock_try_request.side_effect` to classes/instances.
    """

    mock_try_request, mock_check_response = mock_requests_utils_methods(mocker)
    if succeed_last:
        check_succeed_last(
            mock_try_request,
            mock_check_response,
            current_exception=current_exception,
            is_exception=True,
        )
    else:
        check_fail_all_retries(
            mock_try_request,
            mock_check_response,
            current_exception=current_exception,
            is_exception=True,
        )


def test_make_request_non_retryable_exception(
    mocker: MockerFixture,
) -> None:
    """
    Non-retryable exception path: immediate propagation with no backoff.

    Parameters
    ----------
    mocker : MockerFixture
        Patching utility.

    Returns
    -------
    None
        Asserts single attempt, no `check_response`, and no sleeps when an
        out-of-policy exception (e.g., TypeError) occurs.

    Raises
    ------
    AssertionError
        If more than one attempt occurs or backoff logic is invoked.

    Notes
    -----
    - Demonstrates that only whitelisted exceptions are retried.
    """

    non_ret_exception: type[BaseException] = TypeError
    mock_try_request, mock_check_response = mock_requests_utils_methods(mocker)
    check_non_retryable(
        mocker,
        mock_try_request,
        mock_check_response,
        current_exception=non_ret_exception,
        is_exception=True,
    )


def test_make_request_no_retries(mocker: MockerFixture) -> None:
    """
    `max_retries=1` semantics: retryable status raises immediately with one attempt.

    Parameters
    ----------
    mocker : MockerFixture
        Patching utility.

    Returns
    -------
    None
        Asserts a single `try_request` call and one `check_response` invocation.

    Raises
    ------
    AssertionError
        If additional attempts occur or call counts differ.

    Notes
    -----
    - Models the strictest retry budget: first failure leads to immediate raise.
    """

    err_response: MagicMock = MagicMock()
    err_response.status_code = 500
    mock_try_request, mock_check_response = mock_requests_utils_methods(
        mocker, try_request_return=err_response
    )
    mock_try_request.side_effect = requests.HTTPError("boom")
    with pytest.raises(requests.HTTPError):
        make_request(TEST_URL, max_retries=1)
    assert mock_try_request.call_count == 1
    mock_try_request.assert_called_with(TEST_URL, TEST_HEADER, TEST_TIMEOUT, TEST_EXPECT_JSON, None)
    assert mock_check_response.call_count == 1


def check_succeed_last(
    mock_try_request: MagicMock,
    mock_check_response: MagicMock,
    err_response: MagicMock | None = None,
    current_exception: type[BaseException] | None = None,
    is_exception: bool = False,
) -> None:
    """
    Helper: program retryable failures then success on the last attempt.

    Parameters
    ----------
    mock_try_request : MagicMock
        Patched `try_request` to receive a side-effect sequence.
    mock_check_response : MagicMock
        Patched `check_response` used to count per-failure backoff calls.
    err_response : MagicMock, optional
        Response carrying a retryable status for HTTPError-driven failures.
    current_exception : type[BaseException], optional
        Retryable exception class to raise on each failed attempt.
    is_exception : bool
        If True, drive the exception path; otherwise drive the status-code path.

    Returns
    -------
    None
        Asserts final success, expected retry count, and expected backoff invocations.

    Raises
    ------
    AssertionError
        If success is not returned at the last attempt or counts are wrong.

    Notes
    -----
    - Side effect: `[fail] * (TEST_MAX_RETRIES - 1) + [success]`.
    """

    # Set side effect to raise the retryable error/exception
    success_response: MagicMock = MagicMock()
    if is_exception:
        assert current_exception is not None
        mock_try_request.side_effect = [current_exception("boom")] * (TEST_MAX_RETRIES - 1) + [
            success_response
        ]
    else:
        mock_try_request.side_effect = [requests.HTTPError(response=err_response)] * (
            TEST_MAX_RETRIES - 1
        ) + [success_response]

    # Call the function under test
    resp: Response = make_request(TEST_URL, max_retries=TEST_MAX_RETRIES)

    # Assertions
    assert resp is success_response
    assert mock_try_request.call_count == TEST_MAX_RETRIES
    mock_try_request.assert_called_with(TEST_URL, TEST_HEADER, TEST_TIMEOUT, TEST_EXPECT_JSON, None)
    assert mock_check_response.call_count == TEST_MAX_RETRIES - 1


def check_fail_all_retries(
    mock_try_request: MagicMock,
    mock_check_response: MagicMock,
    err_response: MagicMock | None = None,
    current_exception: type[BaseException] | None = None,
    is_exception: bool = False,
) -> None:
    """
    Helper: program all attempts as retryable failures and assert exhaustion behavior.

    Parameters
    ----------
    mock_try_request : MagicMock
        Patched `try_request` to raise on every call.
    mock_check_response : MagicMock
        Patched `check_response` used to count backoff calls.
    err_response : MagicMock, optional
        Response carrying a retryable status for the HTTPError path.
    current_exception : type[BaseException], optional
        Retryable exception class to raise on each attempt.
    is_exception : bool
        If True, exception-driven path; else status-code-driven path.

    Returns
    -------
    None
        Asserts the correct exception is re-raised and that retry/backoff counts match.

    Raises
    ------
    AssertionError
        If the number of attempts or the raised exception type is incorrect.

    Notes
    -----
    - Verifies `try_request.call_count == TEST_MAX_RETRIES` and that the last
    retryable error is propagated.
    """

    # Set side effect to raise the retryable error/exception
    if is_exception:
        assert current_exception is not None
        mock_try_request.side_effect = current_exception()
        with pytest.raises(current_exception):
            make_request(TEST_URL, max_retries=TEST_MAX_RETRIES)
    else:
        mock_try_request.side_effect = requests.HTTPError(response=err_response)
        with pytest.raises(requests.HTTPError):
            make_request(TEST_URL, max_retries=TEST_MAX_RETRIES)

    # Assertions
    assert mock_try_request.call_count == TEST_MAX_RETRIES
    assert mock_check_response.call_count == TEST_MAX_RETRIES
    mock_try_request.assert_called_with(TEST_URL, TEST_HEADER, TEST_TIMEOUT, TEST_EXPECT_JSON, None)


def check_non_retryable(
    mocker: MockerFixture,
    mock_try_request: MagicMock,
    mock_check_response: MagicMock,
    err_response: MagicMock | None = None,
    current_exception: type[BaseException] | None = None,
    is_exception: bool = False,
) -> None:
    """
    Helper: program a non-retryable failure and assert immediate propagation.

    Parameters
    ----------
    mocker : MockerFixture
        Used to patch `time.sleep` so we can assert zero backoff sleeps.
    mock_try_request : MagicMock
        Patched `try_request` to raise once.
    mock_check_response : MagicMock
        Patched or wrapped `check_response` for call assertions.
    err_response : MagicMock, optional
        Response with a non-retryable status for HTTPError-driven paths.
    current_exception : type[BaseException], optional
        Non-retryable exception class (e.g., TypeError) to raise directly.
    is_exception : bool
        If True, exception path (no `check_response` call); else status path
        (expects a single `check_response` call).

    Returns
    -------
    None
        Asserts single attempt, correct `check_response` usage, and zero sleeps.

    Raises
    ------
    AssertionError
        If multiple attempts occur or backoff sleeps are triggered.

    Notes
    -----
    - Guarantees fast-fail semantics for both non-retryable statuses and exceptions.
    """

    # Patch sleep to avoid delays during testing
    mock_sleep: MagicMock = mocker.patch("infra.utils.requests_utils.time.sleep")

    # Set side effect to raise the non-retryable error/exception
    if is_exception:
        assert current_exception is not None
        mock_try_request.side_effect = current_exception()
        with pytest.raises(current_exception):
            make_request(TEST_URL, max_retries=TEST_MAX_RETRIES)
        mock_check_response.assert_not_called()
    else:
        mock_try_request.side_effect = requests.HTTPError(response=err_response)
        with pytest.raises(requests.HTTPError):
            make_request(TEST_URL, max_retries=TEST_MAX_RETRIES)
        assert mock_check_response.call_count == 1

    # Assertions
    assert mock_try_request.call_count == 1
    mock_try_request.assert_called_with(TEST_URL, TEST_HEADER, TEST_TIMEOUT, TEST_EXPECT_JSON, None)
    assert mock_sleep.call_count == 0


def mock_requests_utils_methods(
    mocker: MockerFixture,
    try_request_return: MagicMock | None = None,
    wrap_check_response: bool = False,
) -> tuple[MagicMock, MagicMock]:
    """
    Patch `requests_utils` collaborators to isolate `make_request` in unit tests.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest-mock fixture to apply patches.
    try_request_return : MagicMock or None
        Default return value of `try_request` when no side effect is provided.
    wrap_check_response : bool
        If True, wrap real `check_response` (via `wraps`) to preserve behavior
        and count calls; else stub to a no-op.

    Returns
    -------
    tuple[MagicMock, MagicMock]
        `(mock_try_request, mock_check_response)` ready for use in tests.
        `create_header` is also patched to return `TEST_HEADER`.

    Raises
    ------
    AssertionError
        If a downstream test relies on unpatched network behavior.

    Notes
    -----
    - Ensures deterministic, I/O-free testing by stubbing all external effects.
    """

    mocker.patch("infra.utils.requests_utils.create_header", return_value=TEST_HEADER)
    mock_try_request: MagicMock = mocker.patch(
        "infra.utils.requests_utils.try_request", return_value=try_request_return
    )
    if wrap_check_response:
        mock_check_response: MagicMock = mocker.patch(
            "infra.utils.requests_utils.check_response", wraps=real_check_response
        )
    else:
        mock_check_response = mocker.patch("infra.utils.requests_utils.check_response")

    return mock_try_request, mock_check_response
