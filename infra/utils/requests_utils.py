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
- Random jitter (± ~10%) is added to spread concurrent retries.
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

import datetime as dt
import email.utils as eu
import math
import os
import random
import time
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
RETRY_AFTER_ERRORS: ExceptionTypes = (ValueError, TypeError, AttributeError)


def make_request(
    url: str,
    expect_json: bool = True,
    max_retries: int = 5,
    backoff_factor: float = 0.5,
    timeout: tuple[float, float] = (3.5, 10.0),
    sleep_time: float = 0.0,
    session: requests.Session | None = None,
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
    timeout : tuple[float, float], default=(3.5, 10)
        (connect_timeout, read_timeout) passed to `requests.get`.
    sleep_time : float, default=0.0
        Initial sleep time (seconds) before the first request.
    session : requests.Session | None, default=None
        Optional `requests.Session` to use for the request.

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
    - If `sleep_time` is provided, it is applied once before the first attempt
      with a small random jitter.
    """

    if sleep_time > 0.0:
        time.sleep(sleep_time + random.uniform(0, 0.05))
    header: dict[str, str] = create_header(expect_json)
    last_exception: BaseException | None = None
    for attempt in range(max_retries):
        try:
            response: requests.Response = try_request(url, header, timeout, expect_json, session)
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
    timeout: tuple[float, float] = (3.5, 10),
    expect_json: bool = True,
    session: requests.Session | None = None,
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
    session : requests.Session | None, default=None
        Optional `requests.Session` to use for the request.

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

    if session is not None:
        response = session.get(url, headers=header, timeout=timeout)
    else:
        response = requests.get(url, headers=header, timeout=timeout)
    response.raise_for_status()
    if expect_json:
        if "json" not in response.headers.get("Content-Type", "").lower():
            raise ValueError(f"Expected JSON response, got {response.headers.get('Content-Type')}")
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

    header: dict[str, str] = {"User-Agent": os.environ["USER_AGENT"]}
    if expect_json:
        header["Accept"] = "application/json; charset=utf-8"
    return header


def check_response(
    response: Any | None, attempt: int, max_retries: int, backoff_factor: float
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

    Raises
    ------
    requests.HTTPError
        If the status code is non-retryable (e.g., 400–404) or otherwise invalid.

    Notes
    -----
    - Computes sleep duration based on status code or exponential backoff.
    - Jitter is applied by handle_status_code for 5xx or 429 without a usable Retry-After.
      When Retry-After is honored, no jitter is added.
    """

    sleep_time: float = backoff_factor * (2**attempt)
    if response is not None:
        status_code: Any | None = getattr(response, "status_code", None)
        headers: dict = getattr(response, "headers", {})
        sleep_time = handle_status_code(status_code, headers, attempt, backoff_factor)
    if attempt < max_retries - 1:
        time.sleep(sleep_time)


def handle_status_code(
    status_code: Any | None, headers: dict, attempt: int, backoff_factor: float
) -> float:
    """
    Compute the retry sleep duration (seconds) from an HTTP status, applying jitter
    for exponential backoff paths and honoring `Retry-After` exactly for 429.

    Parameters
    ----------
    status_code : int | None
        HTTP status code from the response (or None).
    headers : dict
        Response headers; consulted for `Retry-After` when status is 429.
    attempt : int
        Zero-based attempt index used for exponential backoff.
    backoff_factor : float
        Base multiplier for exponential backoff.

    Returns
    -------
    float
        Number of seconds to sleep before the next retry:
        - For retryable 5xx (500, 502, 503, 504): `backoff_factor * (2 ** attempt)` with
          jitter applied in-range ~±10% (via `random.uniform(0.9, 1.1)`).
        - For 429: if `Retry-After` is present and parseable (delta-seconds or HTTP-date),
          return that exact delay (no jitter); otherwise fall back to the jittered
          exponential backoff above.

    Raises
    ------
    requests.HTTPError
        If the status code is non-retryable (e.g., 400–404) or otherwise invalid.

    Notes
    -----
    - Jitter is applied **only** on exponential backoff paths (5xx or 429 without a
      usable `Retry-After`). When `Retry-After` is honored, its value is returned as-is.
    """

    default_sleep: float = backoff_factor * (2**attempt) * random.uniform(0.9, 1.1)

    if status_code in RETRYABLE_STATUS_CODES:
        return default_sleep

    elif status_code == TOO_MANY_REQUESTS_STATUS_CODE:
        retry_after = headers.get("Retry-After")
        if isinstance(retry_after, str):
            return extract_retry_after(retry_after, default_sleep)
        else:
            return default_sleep

    else:
        raise requests.HTTPError(f"Non-retryable status code: {status_code}")


def extract_retry_after(retry_after: str, default_sleep: float) -> float:
    """
    Parse an HTTP `Retry-After` value and return a sleep duration (seconds).

    Parameters
    ----------
    retry_after : str
        Raw `Retry-After` header value. Per RFC 7231, this may be delta-seconds
        (e.g., "120") or an HTTP-date (e.g., "Wed, 21 Oct 2015 07:28:00 GMT").
    default_sleep : float
        Fallback number of seconds to return if parsing fails or yields an unusable value.

    Returns
    -------
    float
        Sleep seconds computed as follows:
        1) If `retry_after` parses as a finite float:
        - If > 0.0, return that value (no jitter).
        - If <= 0.0, return `default_sleep`.
        2) Else, if it parses as an HTTP-date:
        - Compute `(parsed_datetime - now_utc)` in seconds.
        - If > 0.0, return that value; otherwise return `default_sleep`.
        3) Otherwise, return `default_sleep`.

    Notes
    -----
    - Non-finite numerics (`NaN`, `inf`, `-inf`) are treated as
      invalid and fall back to `default_sleep`.
    - This function never raises; it always returns a float suitable for `time.sleep`.
    """

    try:
        retry_after = retry_after.strip()
        float_retry_after: float = float(retry_after)
        if not math.isfinite(float_retry_after) or float_retry_after <= 0.0:
            return default_sleep
        return float_retry_after
    except RETRY_AFTER_ERRORS:
        try:
            possible_date: dt.datetime = eu.parsedate_to_datetime(retry_after)
            if possible_date is None:
                return default_sleep
            if possible_date.tzinfo is None:
                possible_date = possible_date.replace(tzinfo=dt.timezone.utc)
            now: dt.datetime = dt.datetime.now(dt.timezone.utc)
            secs: float = (possible_date - now).total_seconds()
            if secs <= 0.0:
                return default_sleep
            return secs
        except RETRY_AFTER_ERRORS:
            return default_sleep
