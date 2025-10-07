"""
Purpose
-------
Validate `try_request` behavior in isolation from real I/O. Ensures the function
forwards arguments correctly, propagates HTTP errors, and enforces (or bypasses)
JSON content-type validation according to `expect_json`.

Key behaviors
-------------
- Returns successfully for 2xx with JSON content-type when `expect_json=True`.
- Raises ValueError for non-JSON or missing Content-Type when `expect_json=True`.
- Bypasses content-type checks and returns for non-JSON when `expect_json=False`.
- Propagates HTTP errors raised by `raise_for_status()`.

Conventions
-----------
- No network: `requests.get` is patched to return a Response-like mock.
- Assertions target call plumbing and our own validation logic, not `requests`.
- `raise_for_status()` is asserted via the instance method on the mocked response.

Downstream usage
----------------
Run with `pytest`. Treat these tests as the executable spec for `try_request`’s
contract (argument plumbing, error propagation, and JSON-content validation).
"""

from typing import cast
from unittest.mock import MagicMock

import pytest
import requests
from pytest_mock import MockerFixture

from infra.utils.requests_utils import try_request

TEST_URL: str = "https://example.com/api"
TEST_HEADER: dict[str, str] = {"User-Agent": "Moshe", "Accept": "application/json; charset=utf-8"}
TEST_TIMEOUT: tuple[float, float] = (3.5, 10.0)
NON_JSON_VARIANTS: list[str | None] = [
    "text/html; charset=utf-8",
    "application/xml",
    "text/plain",
    "image/png",
    "",
    None,
]


def test_try_request_happy_json(mocker: MockerFixture) -> None:
    """
    Happy path: 200 OK with JSON content-type returns the same response.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest-mock fixture used to patch `requests.get` and inspect the mocked
        response's `raise_for_status()`.

    Returns
    -------
    None
        Asserts that the returned object is the exact mocked response and that
        `requests.get` was called with the provided URL, headers, and timeout.

    Raises
    ------
    AssertionError
        If the response is not returned as-is, or if plumbing calls differ.

    Notes
    -----
    - This test also confirms that `raise_for_status()` was invoked exactly once.
    """

    mock_response: MagicMock = MagicMock(status_code=200)
    mock_response.headers = {"Content-Type": "application/json; charset=utf-8"}
    mock_get, mock_raise_for_status = mock_requests_methods(
        # Cast to appease mypy about MagicMock vs requests.Response.
        mocker,
        get_return_value=cast(requests.Response, mock_response),
    )
    resp: requests.Response = try_request(TEST_URL, TEST_HEADER, TEST_TIMEOUT)
    assert resp is mock_response
    general_assertions(mock_get, mock_raise_for_status)


@pytest.mark.parametrize("non_json_variant", NON_JSON_VARIANTS)
def test_try_request_happy_non_json(mocker: MockerFixture, non_json_variant: str) -> None:
    """
    Bypass path: when `expect_json=False`, non-JSON content-types return successfully.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest-mock patching utility.
    non_json_variant : str
        A non-JSON Content-Type (or empty string/None) to attach to the response.

    Returns
    -------
    None
        Asserts that `try_request` returns the mocked response and that
        plumbing calls match expectations.

    Raises
    ------
    AssertionError
        If a ValueError is raised despite `expect_json=False` or if plumbing is wrong.

    Notes
    -----
    - Demonstrates that content-type enforcement is disabled when requested.
    """

    mock_response: MagicMock = MagicMock(status_code=200)
    mock_response.headers = {"Content-Type": non_json_variant}
    mock_get, mock_raise_for_status = mock_requests_methods(
        # Cast to appease mypy about MagicMock vs requests.Response.
        mocker,
        get_return_value=cast(requests.Response, mock_response),
    )
    resp: requests.Response = try_request(TEST_URL, TEST_HEADER, TEST_TIMEOUT, expect_json=False)
    assert resp is mock_response
    general_assertions(mock_get, mock_raise_for_status)


@pytest.mark.parametrize("non_json_variant", NON_JSON_VARIANTS)
def test_try_request_non_json_response(mocker: MockerFixture, non_json_variant: str) -> None:
    """
    Validation path: non-JSON or missing Content-Type raises ValueError when `expect_json=True`.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest-mock patching utility.
    non_json_variant : str
        A Content-Type string that is not JSON, or None to simulate missing header.

    Returns
    -------
    None
        Asserts that a ValueError is raised with an informative message.

    Raises
    ------
    ValueError
        Expected when Content-Type is non-JSON or missing and `expect_json=True`.

    Notes
    -----
    - Covers both explicit non-JSON types and an absent header case.
    """

    mock_response: MagicMock = MagicMock(status_code=200)
    if non_json_variant is None:
        mock_response.headers = {}
    else:
        mock_response.headers = {"Content-Type": non_json_variant}
    mock_get, mock_raise_for_status = mock_requests_methods(mocker, get_return_value=mock_response)
    with pytest.raises(ValueError, match="Expected JSON response"):
        try_request(TEST_URL, TEST_HEADER, TEST_TIMEOUT)
    general_assertions(mock_get, mock_raise_for_status)


def test_try_request_http_error(mocker: MockerFixture) -> None:
    """
    HTTP error propagation: `raise_for_status()` exceptions bubble up unmodified.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest-mock patching utility.

    Returns
    -------
    None
        Asserts that a `requests.HTTPError` raised by `raise_for_status()` is
        propagated and that plumbing is correct.

    Raises
    ------
    requests.HTTPError
        Propagated from the mocked response's `raise_for_status()`.

    Notes
    -----
    - Confirms `try_request` does not catch or transform HTTP errors.
    """

    mock_response: MagicMock = MagicMock(status_code=404)
    mock_get, mock_raise_for_status = mock_requests_methods(mocker, get_return_value=mock_response)
    mock_raise_for_status.side_effect = requests.HTTPError("HTTP Error")
    with pytest.raises(requests.HTTPError, match="HTTP Error"):
        try_request(TEST_URL, TEST_HEADER, TEST_TIMEOUT)
    general_assertions(mock_get, mock_raise_for_status)


def mock_requests_methods(
    mocker: MockerFixture,
    get_return_value: MagicMock | requests.Response,
) -> tuple[MagicMock, MagicMock]:
    """
    Patch `requests.get` to return a provided Response-like mock and expose its
    instance `raise_for_status` for assertions.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest-mock fixture for applying patches.
    get_return_value : MagicMock
        The Response-like mock to be returned by `requests.get`.

    Returns
    -------
    tuple[MagicMock, MagicMock]
        `(mock_get, mock_raise_for_status)`:
        - `mock_get` is the patched `requests.get`.
        - `mock_raise_for_status` is the instance method on `get_return_value`.

    Raises
    ------
    AssertionError
        If downstream tests rely on unpatched network I/O.

    Notes
    -----
    - This avoids patching the class method; we assert against the actual instance.
    """

    mock_get: MagicMock = mocker.patch(
        "infra.utils.requests_utils.requests.get", return_value=get_return_value
    )
    mock_raise_for_status: MagicMock = cast(MagicMock, get_return_value.raise_for_status)
    return mock_get, mock_raise_for_status


def general_assertions(
    mock_get: MagicMock,
    mock_raise_for_status: MagicMock,
) -> None:
    """
    Common plumbing checks for `try_request` tests.

    Parameters
    ----------
    mock_get : MagicMock
        The patched `requests.get` mock used to validate call arguments.
    mock_raise_for_status : MagicMock
        The response instance's `raise_for_status` mock to validate it was called.

    Returns
    -------
    None
        Performs shared assertions on call signatures and call counts.

    Raises
    ------
    AssertionError
        If `requests.get` was not called with `(url, headers, timeout)` or if
        `raise_for_status()` was not called exactly once.

    Notes
    -----
    - Keeps individual tests focused by centralizing repeated plumbing checks.
    """

    mock_get.assert_called_with(TEST_URL, headers=TEST_HEADER, timeout=TEST_TIMEOUT)
    mock_raise_for_status.assert_called_once()
