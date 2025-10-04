"""
Purpose
-------
Unit tests for `extract_retry_after`, covering numeric seconds, HTTP-date parsing,
and failure/edge cases. Ensures the helper always returns a sane, non-negative
sleep duration per RFC 7231 semantics and local policy.

Key behaviors
-------------
- Numeric input: accepts trimmed integers/floats; defaults on NaN/Inf/negatives/zero.
- HTTP-date input: converts to UTC, computes time delta in seconds against a fixed "now".
    Defaults if the date is in the past.

Conventions
-----------
- `email.utils.parsedate_to_datetime` and `datetime.datetime` are patched
  and datetime.now's return value is stubbed in the
  module-under-test namespace to keep tests deterministic.
- UTC-aware `TEST_NOW_DATE` is used to avoid naive/aware arithmetic errors.
- Table-driven tests via `pytest.mark.parametrize` increase coverage and readability.

Downstream usage
----------------
Treat these tests as the behavioral contract for backoff timing derived from
`Retry-After` headers. If policy changes (e.g., handling of zero/negative values),
update both implementation and these tests.
"""

import datetime as dt
from typing import List
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from infra.utils.requests_utils import extract_retry_after

TEST_NUMERICAL_VALS: List[str] = ["5", "5.0", " 5 ", "  5.0  "]
TEST_BAD_NUMERICAL_VALS: List[str] = ["inf", "INF", "nan", "Nan", "-5", "-5.0", "0", "0.0"]
TEST_CORRECT_DATES: List[tuple[str, dt.datetime, float]] = [
    ("Wed, 21 Oct 2015 07:30:00 GMT", dt.datetime(2015, 10, 21, 7, 30, 0), 120.0),
    ("Wed, 21 Oct 2015 08:28:00 GMT", dt.datetime(2015, 10, 21, 8, 28, 0), 3600.0),
    ("Wed, 21 Oct 2015 09:28:00 GMT", dt.datetime(2015, 10, 21, 9, 28, 0), 7200.0),
]
TEST_NON_POSITIVE_DATES: List[tuple[str, dt.datetime]] = [
    ("Wed, 21 Oct 2015 07:00:00 GMT", dt.datetime(2015, 10, 21, 7, 0, 0)),
    ("Wed, 21 Oct 2015 07:28:00 GMT", dt.datetime(2015, 10, 21, 7, 28, 0)),
]
TEST_ERRONEOUS_DATES: List[str] = ["Wed, 21 Oct 2015 07:28:00", "", "Not a date"]
TEST_NOW_DATE: dt.datetime = dt.datetime(2015, 10, 21, 7, 28, 0, tzinfo=dt.timezone.utc)
EXPECTED_NUMERICAL_VAL: float = 5.0
DEFAULT_SLEEP: float = 60.0


@pytest.mark.parametrize("val", TEST_NUMERICAL_VALS)
def test_extract_retry_after_happy_numerical(mocker: MockerFixture, val: str) -> None:
    """
    Accepts well-formed numeric seconds (int/float with whitespace) and returns that value.

    Parameters
    ----------
    mocker : MockerFixture
        Patches date parsing and `now` to prove they are not touched for numeric paths.
    val : str
        A numeric variant expected to parse successfully.

    Returns
    -------
    None
        Asserts exact equality with EXPECTED_NUMERICAL_VAL and that parsing/now were unused.

    Raises
    ------
    AssertionError
        If the value is not returned verbatim or date/now were called unexpectedly.

    Notes
    -----
    - Confirms the fast path (numeric) short-circuits before date parsing.
    """
    mock_parse_to_datetime, mock_datetime_now = mock_parse_to_datetime_and_now(mocker)
    result: float = extract_retry_after(val, DEFAULT_SLEEP)
    general_asserts(result, EXPECTED_NUMERICAL_VAL, mock_parse_to_datetime, mock_datetime_now)


@pytest.mark.parametrize("val", TEST_BAD_NUMERICAL_VALS)
def test_extract_retry_after_bad_numerical(mocker: MockerFixture, val: str) -> None:
    """
    Rejects non-finite or non-positive numeric strings and falls back to default_sleep.

    Parameters
    ----------
    mocker : MockerFixture
        Patches date parsing and `now` to ensure they are not touched on bad numerics.
    val : str
        A bad numeric variant (NaN/Inf/negative/zero).

    Returns
    -------
    None
        Asserts result equals DEFAULT_SLEEP and parsing/now were unused.

    Raises
    ------
    AssertionError
        If a bad numeric is not mapped to DEFAULT_SLEEP.
    """
    mock_parse_to_datetime, mock_datetime_now = mock_parse_to_datetime_and_now(mocker)
    result: float = extract_retry_after(val, DEFAULT_SLEEP)
    general_asserts(result, DEFAULT_SLEEP, mock_parse_to_datetime, mock_datetime_now)


@pytest.mark.parametrize("date_str, parsed_date, expected_delta", TEST_CORRECT_DATES)
def test_extract_retry_after_happy_date(
    mocker: MockerFixture, date_str: str, parsed_date: dt.datetime, expected_delta: float
) -> None:
    """
    Parses a valid HTTP-date, converts to UTC if needed, and returns the positive delta.

    Parameters
    ----------
    mocker : MockerFixture
        Patches `parsedate_to_datetime` and `datetime.now`.
    date_str : str
        RFC 7231 HTTP-date string.
    parsed_date : datetime
        The datetime to be returned by the parser.
    expected_delta : float
        Expected seconds between parsed_date and TEST_NOW_DATE.

    Returns
    -------
    None
        Asserts exact delta, and that both parser and now were called once.

    Raises
    ------
    AssertionError
        If delta is incorrect or required calls were not performed.

    Notes
    -----
    - Uses a fixed UTC "now" to keep arithmetic deterministic.
    """

    mock_parse_to_datetime, mock_datetime_now = mock_parse_to_datetime_and_now(
        mocker, parse_return_value=parsed_date, now_return_value=TEST_NOW_DATE
    )
    result: float = extract_retry_after(date_str, DEFAULT_SLEEP)
    general_asserts(
        result, expected_delta, mock_parse_to_datetime, mock_datetime_now, correct_date_path=True
    )


@pytest.mark.parametrize("returned_date_str, returned_date", TEST_NON_POSITIVE_DATES)
def test_extract_retry_after_non_positive_date(
    mocker: MockerFixture, returned_date_str: str, returned_date: dt.datetime
) -> None:
    """
    Returns default_sleep when the parsed HTTP-date is at/before now (non-positive delta).

    Parameters
    ----------
    mocker : MockerFixture
        Patches `parsedate_to_datetime` and `datetime.now`.

    Returns
    -------
    None
        Asserts DEFAULT_SLEEP and that both parser and now were called.

    Raises
    ------
    AssertionError
        If the non-positive delta is not mapped to DEFAULT_SLEEP.
    """
    mock_parse_to_datetime, mock_datetime_now = mock_parse_to_datetime_and_now(
        mocker, parse_return_value=returned_date, now_return_value=TEST_NOW_DATE
    )
    result: float = extract_retry_after(returned_date_str, DEFAULT_SLEEP)
    general_asserts(
        result, DEFAULT_SLEEP, mock_parse_to_datetime, mock_datetime_now, correct_date_path=True
    )


@pytest.mark.parametrize("erroneous_str_val", TEST_ERRONEOUS_DATES)
def test_extract_retry_after_erroneous_date(mocker: MockerFixture, erroneous_str_val: str) -> None:
    """
    Falls back to default_sleep when the HTTP-date string cannot be parsed.

    Parameters
    ----------
    mocker : MockerFixture
        Patches `parsedate_to_datetime` to return None.
    erroneous_str_val : str
        A string that is not an RFC 7231 date.

    Returns
    -------
    None
        Asserts DEFAULT_SLEEP, verifies parser called once and `now` not called.

    Raises
    ------
    AssertionError
        If unparsable date does not lead to DEFAULT_SLEEP.
    """

    mock_parse_to_datetime, mock_datetime_now = mock_parse_to_datetime_and_now(mocker)
    result: float = extract_retry_after(erroneous_str_val, DEFAULT_SLEEP)
    general_asserts(
        result, DEFAULT_SLEEP, mock_parse_to_datetime, mock_datetime_now, erroneous_date_path=True
    )


def mock_parse_to_datetime_and_now(
    mocker: MockerFixture,
    parse_return_value: dt.datetime | None = None,
    now_return_value: dt.datetime | None = None,
) -> tuple[MagicMock, MagicMock]:
    """
    Patch `parsedate_to_datetime` and `datetime.now` in the module-under-test.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest-mock fixture for applying patches.
    parse_return_value : datetime or None, optional
        Value returned by the patched parser.
    now_return_value : datetime or None, optional
        Value returned by the patched `datetime.now`.

    Returns
    -------
    tuple[MagicMock, MagicMock]
        (mock_parse_to_datetime, mock_datetime_now) for call assertions.

    Raises
    ------
    AssertionError
        If downstream tests assume unpatched time-dependent behavior.

    Notes
    -----
    - Patch targets use the module namespace: 'infra.utils.requests_utils.eu.parsedate_to_datetime'
      and 'infra.utils.requests_utils.dt.datetime'.
    """

    mock_parse_to_datetime: MagicMock = mocker.patch(
        "infra.utils.requests_utils.eu.parsedate_to_datetime", return_value=parse_return_value
    )
    mock_datetime: MagicMock = mocker.patch("infra.utils.requests_utils.dt.datetime")
    mock_datetime.now.return_value = now_return_value
    return mock_parse_to_datetime, mock_datetime.now


def general_asserts(
    result: float,
    expected: float,
    mock_parse_to_datetime: MagicMock,
    mock_datetime_now: MagicMock,
    correct_date_path: bool = False,
    erroneous_date_path: bool = False,
) -> None:
    """
    Shared assertions for result equality and call expectations.

    Parameters
    ----------
    result : float
        Actual result returned by `extract_retry_after`.
    expected : float
        Expected result for the scenario.
    mock_parse_to_datetime : MagicMock
        Spy for the parser to assert call counts.
    mock_datetime_now : MagicMock
        Spy for `datetime.now` to assert call counts.
    correct_date_path : bool, default=False
        When True, asserts both parser and now were called once.
    erroneous_date_path : bool, default=False
        When True, asserts parser called once and now not called.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the numerical result is wrong or call patterns differ from expectations.

    Notes
    -----
    - In the numeric fast path, both parser and now should be untouched.
    """

    assert result == expected
    if correct_date_path:
        mock_parse_to_datetime.assert_called_once()
        mock_datetime_now.assert_called_once()
    elif erroneous_date_path:
        mock_parse_to_datetime.assert_called_once()
        mock_datetime_now.assert_not_called()
    else:
        mock_parse_to_datetime.assert_not_called()
        mock_datetime_now.assert_not_called()
