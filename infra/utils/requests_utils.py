"""
Purpose
-------
Provide a resilient HTTP request helper with retry, backoff, and content-type
validation. Centralizes transport-layer logic for interacting with SEC/EDGAR
JSON endpoints and other APIs, ensuring consistent behavior across the project.

Key behaviors
-------------
- Issues GET requests with a user-defined User-Agent header.
- Retries on transient network errors or server-side errors (429, 5xx).
- Implements exponential backoff with jitter and honors `Retry-After` headers.
- Validates response status and optionally enforces JSON content-type.
- Surfaces the final `requests.Response` for downstream parsing and handling.

Conventions
-----------
- Retries use exponential backoff: `backoff_factor * (2 ** attempt)`.
- Random jitter (± ~1s) is added to spread concurrent retries.
- Treats 400–404 as non-retryable (fails fast).
- Always requires a `USER_AGENT` environment variable.
- Timeout is a tuple `(connect_timeout, read_timeout)` passed directly to
  `requests`.

Downstream usage
----------------
Import `make_request` for all HTTP fetches that need robust retry behavior.
Do not parse or interpret JSON in this module; call `response.json()` in
the caller after a validated response is returned.
"""

import os
import random
from time import sleep
from typing import Any, TypeAlias
import requests


ExceptionTypes: TypeAlias = tuple[type[BaseException], ...]


RETRYABLE_STATUS_CODES: set[int] = {500, 502, 503, 504}
TOO_MANY_REQUESTS_STATUS_CODE: int = 429
RETRYABLE_EXCEPTIONS: ExceptionTypes = (
    requests.Timeout,
    requests.ConnectionError,
    requests.RequestException,
    ValueError,
)


def make_request(
        url: str,
        expect_json: bool = True,
        max_retries: int = 5,
        backoff_factor: float = 0.5,
        timeout: tuple[float, float] = (3.05, 10),
    ) -> requests.Response:
    """
    Perform a GET request with retry and backoff handling.

    Parameters
    ----------
    url : str
        The target URL to fetch.
    expect_json : bool, default=True
        If True, validates that the response has a JSON Content-Type.
    max_retries : int, default=5
        Maximum number of attempts before failing.
    backoff_factor : float, default=0.5
        Base multiplier for exponential backoff.
    timeout : tuple[float, float], default=(3.05, 10)
        (connect_timeout, read_timeout) passed to `requests.get`.

    Returns
    -------
    requests.Response
        The final successful response object.

    Raises
    ------
    requests.HTTPError
        If a non-retryable HTTP status (e.g., 400–404) is encountered.
    RuntimeError
        If no exception is available after retries are exhausted.
    BaseException
        Re-raises the last retryable exception after retries are exhausted.

    Notes
    -----
    - Retries on `requests.Timeout`, `requests.ConnectionError`,
    `requests.RequestException`, and `ValueError` from validation.
    - Response JSON is not parsed here; caller should handle content.
    """

    header: dict[str, str] = create_header(expect_json)
    last_exception: BaseException | None = None
    for attempt in range(max_retries):
        try:
            response: requests.Response = try_request(
                url,
                header,
                timeout,
                expect_json
            )
            return response
        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e
            exception_response: Any | None = (
                e.response if isinstance(e, requests.RequestException) else None
            )
            check_response(exception_response, attempt, max_retries, backoff_factor)
    raise last_exception if last_exception else RuntimeError("Request failed without exception")


def try_request(
        url: str,
        header: dict[str, str],
        timeout: tuple[float, float],
        expect_json: bool = True
    ) -> requests.Response:
    """
    Send a single GET request and perform immediate validation.

    Parameters
    ----------
    url : str
        The target URL to fetch.
    header : dict[str, str]
        Headers to include in the request.
    timeout : tuple[float, float]
        (connect_timeout, read_timeout) passed to `requests.get`.
    expect_json : bool, default=True
        If True, enforces that the Content-Type contains JSON.

    Returns
    -------
    requests.Response
        The response object, guaranteed to have passed status checks.

    Raises
    ------
    requests.HTTPError
        If the response status code indicates a client or server error.
    ValueError
        If JSON is expected but Content-Type is not JSON.
    """

    response = requests.get(
        url,
        headers=header,
        timeout=timeout
    )
    response.raise_for_status()
    if expect_json:
        if not 'json' in response.headers.get('Content-Type', '').lower():
            raise ValueError(
                f"Expected JSON response, got {response.headers.get('Content-Type')}"
            )
    return response


def create_header(expect_json: bool = True) -> dict[str, str]:
    """
    Construct a request header with User-Agent and optional Accept.

    Parameters
    ----------
    expect_json : bool, default=True
        If True, adds `Accept: application/json; charset=utf-8`.

    Returns
    -------
    dict[str, str]
        Header dictionary including User-Agent (from environment) and optional Accept.

    Raises
    ------
    KeyError
        If USER_AGENT is not set in environment variables.
    """

    header: dict[str, str] = {
        "User-Agent": os.environ['USER_AGENT']
    }
    if expect_json:
        header["Accept"] = "application/json; charset=utf-8"
    return header


def check_response(
        response: Any | None,
        attempt: int,
        max_retries: int,
        backoff_factor: float
    ) -> None:
    """
    Handle retry decisions and sleep between attempts.

    Parameters
    ----------
    response : requests.Response | None
        The response object if available; None for network-level errors.
    attempt : int
        Current attempt number (zero-based).
    max_retries : int
        Total retry limit.
    backoff_factor : float
        Base multiplier for exponential backoff.

    Returns
    -------
    None

    Notes
    -----
    - Computes sleep duration based on status code or exponential backoff.
    - Applies jitter to reduce synchronized retries.
    - Does not raise; caller is responsible for re-raising the last exception.
    """

    sleep_time: float = backoff_factor * (2 ** attempt)
    if response is not None:
        status_code: Any | None = getattr(response, 'status_code', None)
        headers: dict = getattr(response, 'headers', {})
        sleep_time = handle_status_code(status_code, headers, attempt, backoff_factor)
    if attempt < max_retries - 1:
        sleep(sleep_time*random.uniform(0.9, 1.1))


def handle_status_code(
        status_code:  Any | None,
        headers: dict,
        attempt: int,
        backoff_factor: float
    ) -> float:
    """
    Compute backoff sleep duration based on HTTP status.

    Parameters
    ----------
    status_code : int
        The HTTP status code from the response.
    headers : dict
        Response headers, used to check `Retry-After` on 429.
    attempt : int
        Current attempt number (zero-based).
    backoff_factor : float
        Base multiplier for exponential backoff.

    Returns
    -------
    float
        Number of seconds to sleep before retry.

    Raises
    ------
    requests.HTTPError
        If the status code is not retryable (e.g., 400–404, unexpected values).

    Notes
    -----
    - Retryable: 500, 502, 503, 504.
    - Too Many Requests (429): honors `Retry-After` if numeric, else backoff.
    """

    default_sleep = backoff_factor * (2 ** attempt)

    if status_code in RETRYABLE_STATUS_CODES:
        return default_sleep

    elif status_code == TOO_MANY_REQUESTS_STATUS_CODE:
        retry_after = headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            return int(retry_after)
        else:
            return default_sleep

    else:
        raise requests.HTTPError(f"Non-retryable status code: {status_code}")
