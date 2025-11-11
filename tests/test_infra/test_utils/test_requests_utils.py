"""
Purpose
-------
Unit tests for the requests_utils module that provides `make_request` and its
supporting functions (`try_request`, `create_header`, `check_response`,
`handle_status_code`, and `extract_retry_after`).

Key behaviors
-------------
- Verify that `create_header`:
  - pulls `USER_AGENT` from the environment,
  - conditionally adds an `Accept` header when JSON is expected.
- Verify that `try_request`:
  - uses a provided `requests.Session` when given,
  - raises on non-2xx responses via `raise_for_status`,
  - enforces JSON Content-Type when `expect_json=True`.
- Verify that `handle_status_code` and `extract_retry_after`:
  - compute backoff-based sleep durations with jitter for retryable 5xx,
  - honor `Retry-After` headers for 429 if parseable,
  - raise on non-retryable status codes.
- Verify that `check_response`:
  - delegates status handling to `handle_status_code`,
  - calls `time.sleep` with the computed duration when more retries remain.
- Verify that `make_request`:
  - succeeds on the first attempt when no errors occur,
  - retries on configured retryable exceptions and errors,
  - propagates the last exception when retries are exhausted,
  - fails fast on non-retryable HTTP status codes (400–404, etc.).

Conventions
-----------
- All network calls are stubbed; no real HTTP traffic is generated.
- `time.sleep` and random jitter are monkeypatched to keep tests fast
  and deterministic where timing matters.
- Tests use small dummy response/session objects rather than real
  `requests` implementations, except where the type is checked.

Downstream usage
----------------
- Run via `pytest` as part of the CI suite.
- Serves as executable documentation of how retry, backoff, and header
  logic are expected to behave across different failure modes.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, cast

import pytest
import requests

from infra.utils import requests_utils


def test_create_header_includes_user_agent_and_optional_accept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure `create_header` reads USER_AGENT and adds Accept when JSON is expected.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to manipulate environment variables for the
        duration of the test.

    Returns
    -------
    None
        The test passes if:
            - USER_AGENT is required and propagated to the header,
            - `Accept` is present when `expect_json=True`,
            - and `Accept` is omitted when `expect_json=False`.

    Raises
    ------
    AssertionError
        If headers do not match the expected shape or contents.
    """

    monkeypatch.setenv("USER_AGENT", "my-agent/1.0")

    header_json = requests_utils.create_header(expect_json=True)
    assert header_json["User-Agent"] == "my-agent/1.0"
    assert header_json["Accept"] == "application/json; charset=utf-8"

    header_no_json = requests_utils.create_header(expect_json=False)
    assert header_no_json["User-Agent"] == "my-agent/1.0"
    assert "Accept" not in header_no_json


def test_create_header_raises_if_user_agent_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that `create_header` fails fast when USER_AGENT is not set.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to clear USER_AGENT from the environment.

    Returns
    -------
    None
        The test passes if accessing `create_header` without USER_AGENT
        raises a `KeyError`.

    Raises
    ------
    AssertionError
        If no exception or an unexpected exception type is raised.
    """

    monkeypatch.delenv("USER_AGENT", raising=False)
    with pytest.raises(KeyError):
        requests_utils.create_header(expect_json=True)


def test_try_request_uses_session_and_validates_json_content_type() -> None:
    """
    Check that `try_request` uses a provided session and enforces JSON Content-Type.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture; not used directly here but kept for consistency and
        future extension if needed.

    Returns
    -------
    None
        The test passes if:
            - `session.get` is called instead of `requests.get`,
            - `raise_for_status` is invoked,
            - and a non-JSON Content-Type triggers a `ValueError` when
            `expect_json=True`.

    Raises
    ------
    AssertionError
        If the wrong call path is taken or the validation behavior is not
        observed as expected.
    """

    class DummyResponse:
        def __init__(self, status_code: int, content_type: str) -> None:
            self.status_code = status_code
            self.headers = {"Content-Type": content_type}
            self.raise_called = False

        def raise_for_status(self) -> None:
            self.raise_called = True

    class DummySession:
        def __init__(self, resp: DummyResponse) -> None:
            self.resp = resp
            self.calls: list[dict[str, Any]] = []

        def get(
            self, url: str, headers: dict[str, str], timeout: tuple[float, float]
        ) -> DummyResponse:
            self.calls.append({"url": url, "headers": headers, "timeout": timeout})
            return self.resp

    # First, a valid JSON response.
    good_resp = DummyResponse(200, "application/json; charset=utf-8")
    session = DummySession(good_resp)
    header = {"User-Agent": "x"}

    result = requests_utils.try_request(
        url="https://example.test/json",
        header=header,
        timeout=(1.0, 2.0),
        expect_json=True,
        session=cast(requests.Session, session),
    )

    assert result is good_resp
    assert good_resp.raise_called
    assert session.calls[0]["url"] == "https://example.test/json"

    # Now a non-JSON Content-Type should raise ValueError when expect_json=True.
    bad_resp = DummyResponse(200, "text/html")
    session_bad = DummySession(bad_resp)

    with pytest.raises(ValueError):
        requests_utils.try_request(
            url="https://example.test/html",
            header=header,
            timeout=(1.0, 2.0),
            expect_json=True,
            session=cast(requests.Session, session_bad),
        )


def test_try_request_raises_http_error_on_non_2xx(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Ensure `try_request` propagates HTTPError for non-2xx responses.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to patch `requests.get` for this test.

    Returns
    -------
    None
        The test passes if a 500-like dummy response causes `try_request`
        to raise `requests.HTTPError`.

    Raises
    ------
    AssertionError
        If no HTTPError is raised for the simulated failing response.
    """

    class DummyResponse:
        def __init__(self) -> None:
            self.status_code = 500
            self.headers: dict[str, str] = {}

        def raise_for_status(self) -> None:
            raise requests.HTTPError("boom", response=cast(requests.Response, self))

    def fake_get(url: str, headers: dict[str, str], timeout: tuple[float, float]) -> DummyResponse:
        return DummyResponse()

    monkeypatch.setattr(requests_utils.requests, "get", fake_get)
    header = {"User-Agent": "x"}

    with pytest.raises(requests.HTTPError):
        requests_utils.try_request(
            url="https://example.test/fail",
            header=header,
            timeout=(1.0, 2.0),
            expect_json=False,
            session=None,
        )


def test_handle_status_code_retryable_5xx_uses_backoff_and_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that `handle_status_code` returns a jittered exponential backoff for 5xx.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to override `random.uniform` for deterministic
        jitter behavior.

    Returns
    -------
    None
        The test passes if, for a retryable 500 status, the returned sleep
        duration equals `backoff_factor * (2 ** attempt) * jitter`, where
        jitter is the monkeypatched value.

    Raises
    ------
    AssertionError
        If the computed sleep time does not match the expected formula.
    """

    # Fix jitter at exactly 1.0 for predictability.
    monkeypatch.setattr(requests_utils.random, "uniform", lambda a, b: 1.0)

    sleep = requests_utils.handle_status_code(
        status_code=500,
        headers={},
        attempt=2,
        backoff_factor=0.5,
    )

    # backoff_factor * (2**attempt) * jitter = 0.5 * 4 * 1.0 = 2.0
    assert sleep == 2.0


def test_handle_status_code_429_honors_retry_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Ensure `handle_status_code` honors a valid numeric Retry-After for 429.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture; jitter is irrelevant when Retry-After is valid, but we
        still patch `random.uniform` defensively.

    Returns
    -------
    None
        The test passes if a 429 status with `Retry-After: "10"` yields a
        sleep duration of exactly 10 seconds.

    Raises
    ------
    AssertionError
        If the returned value does not match the header’s numeric value.
    """

    monkeypatch.setattr(requests_utils.random, "uniform", lambda a, b: 1.0)

    sleep = requests_utils.handle_status_code(
        status_code=requests_utils.TOO_MANY_REQUESTS_STATUS_CODE,
        headers={"Retry-After": "10"},
        attempt=0,
        backoff_factor=1.0,
    )

    assert sleep == 10.0


def test_handle_status_code_raises_for_non_retryable_status() -> None:
    """
    Check that `handle_status_code` raises HTTPError for non-retryable statuses.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if a status code such as 400 causes a
        `requests.HTTPError` to be raised.

    Raises
    ------
    AssertionError
        If no HTTPError is raised for a non-retryable status code.
    """

    with pytest.raises(requests.HTTPError):
        requests_utils.handle_status_code(
            status_code=400,
            headers={},
            attempt=0,
            backoff_factor=1.0,
        )


def test_extract_retry_after_numeric_and_http_date(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Validate `extract_retry_after` for numeric and HTTP-date Retry-After values.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to control "now" when testing HTTP-date parsing.

    Returns
    -------
    None
        The test passes if:
            - positive numeric values are returned as-is,
            - zero or negative numeric values fall back to the default,
            - HTTP-date values in the future return a positive delta,
            - HTTP-date values in the past fall back to the default.

    Raises
    ------
    AssertionError
        If any of the cases above do not match the documented behavior.
    """

    default_sleep = 5.0

    # Positive numeric.
    assert requests_utils.extract_retry_after("12", default_sleep) == 12.0

    # Zero → fallback.
    assert requests_utils.extract_retry_after("0", default_sleep) == default_sleep

    # Negative → fallback.
    assert requests_utils.extract_retry_after("-3", default_sleep) == default_sleep

    # Non-finite → fallback.
    assert requests_utils.extract_retry_after("NaN", default_sleep) == default_sleep

    # HTTP-date in the future.
    now = dt.datetime(2025, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    future = now + dt.timedelta(seconds=30)
    http_date = future.strftime("%a, %d %b %Y %H:%M:%S GMT")

    class DummyDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(requests_utils.dt, "datetime", DummyDateTime)

    sleep_future = requests_utils.extract_retry_after(http_date, default_sleep)
    assert 29.0 <= sleep_future <= 31.0

    # HTTP-date in the past → fallback.
    past = now - dt.timedelta(seconds=30)
    past_http_date = past.strftime("%a, %d %b %Y %H:%M:%S GMT")
    sleep_past = requests_utils.extract_retry_after(past_http_date, default_sleep)
    assert sleep_past == default_sleep


def test_check_response_delegates_to_handle_status_code_and_sleeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure `check_response` calls `handle_status_code` and then `time.sleep`.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to stub `handle_status_code` and `time.sleep`
        for deterministic verification.

    Returns
    -------
    None
        The test passes if:
            - `handle_status_code` is invoked with the response’s status/headers,
            - `time.sleep` is called exactly once with the value returned by
              `handle_status_code` when more retries remain.

    Raises
    ------
    AssertionError
        If delegation or sleeping behavior does not match expectations.
    """

    class DummyResponse:
        def __init__(self) -> None:
            self.status_code = 503
            self.headers = {"X-Dummy": "1"}

    dummy_resp = DummyResponse()

    recorded: dict[str, Any] = {"sleep": None, "handled": False}

    def fake_handle_status_code(
        status_code: Any, headers: dict, attempt: int, backoff_factor: float
    ) -> float:
        recorded["handled"] = True
        assert status_code == 503
        assert headers == dummy_resp.headers
        assert attempt == 1
        assert backoff_factor == 0.5
        return 2.5

    def fake_sleep(secs: float) -> None:
        recorded["sleep"] = secs

    monkeypatch.setattr(requests_utils, "handle_status_code", fake_handle_status_code)
    monkeypatch.setattr(requests_utils.time, "sleep", fake_sleep)

    requests_utils.check_response(
        response=dummy_resp,
        attempt=1,
        max_retries=3,
        backoff_factor=0.5,
    )

    assert recorded["handled"] is True
    assert recorded["sleep"] == 2.5


def test_make_request_retries_on_exception_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that `make_request` retries on retryable exceptions and eventually succeeds.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to stub `create_header`, `try_request`, and
        `time.sleep` for deterministic behavior.

    Returns
    -------
    None
        The test passes if:
            - the first call to `try_request` raises a retryable exception,
            - a subsequent call succeeds,
            - `time.sleep` is called once with a backoff-derived delay,
            - and the successful response is returned.

    Raises
    ------
    AssertionError
        If retry logic does not follow the expected sequence.
    """

    monkeypatch.setenv("USER_AGENT", "test-agent")

    header = {"User-Agent": "test-agent"}

    monkeypatch.setattr(requests_utils, "create_header", lambda expect_json: header)

    class DummyResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self.headers = {"Content-Type": "application/json"}

    responses: list[Any] = [
        requests.Timeout("first failure"),  # retryable
        DummyResponse(),  # success
    ]

    calls: dict[str, int | float] = {"try_request": 0, "sleep": 0.0}

    def fake_try_request(
        url: str,
        header_arg: dict[str, str],
        timeout: tuple[float, float],
        expect_json: bool,
        session: Any,
    ) -> DummyResponse:
        calls["try_request"] = int(calls["try_request"]) + 1
        outcome = responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def fake_sleep(secs: float) -> None:
        calls["sleep"] = secs

    monkeypatch.setattr(requests_utils, "try_request", fake_try_request)
    monkeypatch.setattr(requests_utils.time, "sleep", fake_sleep)

    result = requests_utils.make_request(
        url="https://example.test/retry",
        expect_json=True,
        max_retries=3,
        backoff_factor=1.0,
        timeout=(1.0, 2.0),
        sleep_time=0.0,
        session=None,
    )

    assert isinstance(result, DummyResponse)
    assert calls["try_request"] == 2
    # First retry uses attempt=0 in check_response →
    # sleep(backoff_factor * 2**0) = 1.0 (modulo jitter),
    # but our fake_sleep simply records the value passed in from check_response.
    assert calls["sleep"] > 0.0


def test_make_request_stops_on_non_retryable_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure `make_request` fails fast on non-retryable HTTP errors (e.g., 400).

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to stub `create_header`, `try_request`, and
        `handle_status_code` to simulate a 400 error.

    Returns
    -------
    None
        The test passes if a single HTTPError is raised without exhausting
        all retry attempts when a non-retryable status is encountered.

    Raises
    ------
    AssertionError
        If `make_request` continues retrying instead of propagating the
        non-retryable error.
    """

    monkeypatch.setenv("USER_AGENT", "test-agent")
    monkeypatch.setattr(
        requests_utils, "create_header", lambda expect_json: {"User-Agent": "test-agent"}
    )

    class DummyResponse:
        def __init__(self) -> None:
            self.status_code = 400
            self.headers: dict[str, str] = {}

    http_error = requests.HTTPError(
        "bad request", response=cast(requests.Response, DummyResponse())
    )

    def fake_try_request(
        url: str,
        header: dict[str, str],
        timeout: tuple[float, float],
        expect_json: bool,
        session: Any,
    ) -> None:
        raise http_error

    def fake_check_response(
        response: Any, attempt: int, max_retries: int, backoff_factor: float
    ) -> None:
        # Simulate handle_status_code raising immediately for 400.
        raise http_error

    monkeypatch.setattr(requests_utils, "try_request", fake_try_request)
    monkeypatch.setattr(requests_utils, "check_response", fake_check_response)

    with pytest.raises(requests.HTTPError) as excinfo:
        requests_utils.make_request(
            url="https://example.test/bad",
            expect_json=False,
            max_retries=3,
            backoff_factor=1.0,
            timeout=(1.0, 2.0),
            sleep_time=0.0,
            session=None,
        )

    assert excinfo.value is http_error
