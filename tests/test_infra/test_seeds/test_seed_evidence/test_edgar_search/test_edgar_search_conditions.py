"""
Purpose
-------
Unit tests for EDGAR Atom feed gating and pagination conditions (edgar_search_conditions.py).

Key behaviors
-------------
- Verify that `evaluate_updated_timestamp`:
  - returns a parsed timestamp when `<updated>` is within the window, and
  - logs a `loop_break` and returns None when `<updated>` is older than the window
    start, or when `<updated>` is missing.
- Confirm that `check_entry_form_type_eligibility`:
  - accepts entries whose form type is in `ELIGIBLE_FORM_TYPES`, and
  - rejects entries whose form type is not eligible or cannot be resolved.
- Validate that `within_validity_window`:
  - implements half-open semantics `start <= ts < end` on UTC timestamps.
- Ensure that `evaluate_page_break_conditions`:
  - flags pages with no entries as terminal, and
  - stops paging when the oldest filing date is older than the window start,
  - otherwise continues paging.

Conventions
-----------
- Timezone-aware `pandas.Timestamp` with UTC tz is used throughout.
- Logging is captured via a minimal `_DummyLogger` that mimics the subset
  of the `InfraLogger` interface used by the module.
- XML/Atom elements are replaced with lightweight dummy objects and
  stubs for `find_element(...)` so tests exercise only the gating logic.

Downstream usage
----------------
Run this module with `pytest` as part of the CI suite. The tests act as
executable documentation for how EDGAR search conditions gate entries,
apply validity windows, and determine when pagination should stop.
"""

from __future__ import annotations

from typing import Any, Dict, List, cast

import pandas as pd
import pytest

from infra.logging.infra_logger import InfraLogger
from infra.seeds.seed_evidence.edgar_search import edgar_search_conditions as esc
from infra.seeds.seed_evidence.edgar_search.edgar_search_utils import RunData
from infra.seeds.seed_evidence.seed_evidence_types import NameSpaceBindings, ValidityWindow


class _DummyLogger:
    """
    Purpose
    -------
    Minimal stand-in for `InfraLogger` capturing info/debug/warning calls.

    Key behaviors
    -------------
    - Records `.info(...)`, `.warning(...)`, and `.debug(...)` invocations.
    - Stores event name, positional args, and keyword args for assertions.

    Parameters
    ----------
    None
        Instances are constructed without additional initialization.

    Attributes
    ----------
    infos : list[dict[str, Any]]
        Recorded info-level events with event name, args, and kwargs.
    warnings : list[dict[str, Any]]
        Recorded warning-level events.
    debugs : list[dict[str, Any]]
        Recorded debug-level events.

    Notes
    -----
    - The real `InfraLogger` supports structured logging with a message
      identifier and optional `msg` and `context` fields; this dummy class
      mirrors that calling pattern loosely so tests can assert on both the
      event identifier and the context payload.
    """

    def __init__(self) -> None:
        self.infos: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.debugs: List[Dict[str, Any]] = []

    def info(self, event: str, *args: Any, **kwargs: Any) -> None:
        self.infos.append({"event": event, "args": args, "kwargs": kwargs})

    def warning(self, event: str, *args: Any, **kwargs: Any) -> None:
        self.warnings.append({"event": event, "args": args, "kwargs": kwargs})

    def debug(self, event: str, *args: Any, **kwargs: Any) -> None:
        self.debugs.append({"event": event, "args": args, "kwargs": kwargs})


class _DummyElement:
    """
    Purpose
    -------
    Lightweight XML element stand-in used by tests for `find_element(...)`
    consumers.

    Key behaviors
    -------------
    - Exposes `.text` and `.get(...)` attributes used by search-condition
      helpers.
    - Can be constructed with arbitrary text and attributes.

    Parameters
    ----------
    text : str | None
        Text content returned by the `.text` attribute.
    attrs : dict[str, Any] | None
        Mapping of attribute names to values used by `.get(...)`.

    Attributes
    ----------
    text : str | None
        Stored text content.
    _attrs : dict[str, Any]
        Internal attribute dictionary used for `.get(...)` lookups.

    Notes
    -----
    - Only the minimal API required by the tests is implemented; it is not
      a full XML element implementation.
    """

    def __init__(self, text: str | None = None, attrs: Dict[str, Any] | None = None) -> None:
        self.text = text
        self._attrs = attrs or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._attrs.get(key, default)


def test_evaluate_updated_timestamp_returns_ts_and_no_break(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that `evaluate_updated_timestamp` returns the parsed timestamp
    when `<updated>` is within the validity window and does not log a
    loop-break condition.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `find_element(...)` so that a synthetic `<updated>`
        element is returned.

    Returns
    -------
    None
        The test passes if the function returns the expected timestamp and
        the logger records no `loop_break` events.

    Raises
    ------
    AssertionError
        If the returned timestamp is incorrect or if any info-level
        loop-break log is emitted.
    """

    start = pd.Timestamp("2020-01-01", tz="UTC")
    end = pd.Timestamp("2020-02-01", tz="UTC")
    window: ValidityWindow = (start, end)

    updated_str = "2020-01-15T00:00:00Z"
    expected_ts = pd.to_datetime(updated_str, utc=True)

    def fake_find_element(entry: Any, tag: str, ns: NameSpaceBindings | None) -> Any:
        if tag == "updated":
            return _DummyElement(text=updated_str)
        return None

    monkeypatch.setattr(esc, "find_element", fake_find_element)

    logger = cast(InfraLogger, _DummyLogger())
    run_data: RunData = {
        "ticker": "AAA",
        "validity_window": window,
        "candidate": "AAA",
        "name_space_bindings": None,
        "entry": object(),
        "logger": logger,
        "oldest_filing_date": pd.Timestamp("2100-01-10", tz="UTC"),
    }

    result = esc.evaluate_updated_timestamp(run_data)
    assert result == expected_ts
    # No loop_break should be logged.
    assert not any(info["event"] == "loop_break" for info in cast(_DummyLogger, logger).infos)


def test_evaluate_updated_timestamp_logs_and_returns_none_for_older_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure `evaluate_updated_timestamp` logs a loop-break and returns None
    when `<updated>` is older than the validity window start.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `find_element(...)` so that `<updated>` is earlier
        than the window start.

    Returns
    -------
    None
        The test passes if:
            - the function returns None, and
            - an info-level `loop_break` is logged with a
              `"older than window"` message and the expected context.

    Raises
    ------
    AssertionError
        If the function does not return None or if the logged event does
        not match the expected payload.
    """

    start = pd.Timestamp("2020-01-01", tz="UTC")
    end = pd.Timestamp("2020-02-01", tz="UTC")
    window: ValidityWindow = (start, end)

    older_str = "2019-12-31T23:59:59Z"

    def fake_find_element(entry: Any, tag: str, ns: NameSpaceBindings | None) -> Any:
        if tag == "updated":
            return _DummyElement(text=older_str)
        return None

    monkeypatch.setattr(esc, "find_element", fake_find_element)

    logger = cast(InfraLogger, _DummyLogger())
    run_data: RunData = {
        "ticker": "AAA",
        "validity_window": window,
        "candidate": "AAA",
        "name_space_bindings": None,
        "entry": object(),
        "logger": logger,
        "oldest_filing_date": pd.Timestamp("2100-01-10", tz="UTC"),
    }

    result = esc.evaluate_updated_timestamp(run_data)
    assert result is None

    dummy_logger = cast(_DummyLogger, logger)
    assert dummy_logger.infos, "Expected a loop_break info log"
    event = dummy_logger.infos[0]
    assert event["event"] == "loop_break"
    assert event["kwargs"].get("msg") == "older than window"
    ctx = event["kwargs"].get("context", {})
    assert ctx.get("stage") == "time_stamp_evaluation"
    assert ctx.get("ticker") == "AAA"
    assert ctx.get("window_start") == window[0].isoformat()
    assert ctx.get("updated_ts") == pd.to_datetime(older_str, utc=True).isoformat()


def test_evaluate_updated_timestamp_returns_none_when_missing_updated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Confirm that `evaluate_updated_timestamp` returns None and does not log
    a loop-break when the `<updated>` element is missing.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `find_element(...)` to always return None.

    Returns
    -------
    None
        The test passes if the function returns None and no info-level
        events are recorded.

    Raises
    ------
    AssertionError
        If a timestamp is returned or if any info log is emitted.
    """

    def fake_find_element(entry: Any, tag: str, ns: NameSpaceBindings | None) -> Any:
        return None

    monkeypatch.setattr(esc, "find_element", fake_find_element)

    logger = cast(InfraLogger, _DummyLogger())
    window = (
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2020-02-01", tz="UTC"),
    )
    run_data: RunData = {
        "ticker": "AAA",
        "validity_window": window,
        "candidate": "AAA",
        "name_space_bindings": None,
        "entry": object(),
        "logger": logger,
        "oldest_filing_date": pd.Timestamp("2100-01-10", tz="UTC"),
    }

    result = esc.evaluate_updated_timestamp(run_data)
    assert result is None
    assert not cast(_DummyLogger, logger).infos


@pytest.mark.parametrize(
    "category_term,title_text,expected",
    [
        ("10-K", None, True),
        ("S-3", None, False),
        (None, "8-K Some filing title", True),
        (None, "FOO Something", False),
        (None, None, False),
    ],
)
def test_check_entry_form_type_eligibility_resolves_from_category_or_title(
    monkeypatch: pytest.MonkeyPatch,
    category_term: str | None,
    title_text: str | None,
    expected: bool,
) -> None:
    """
    Validate that `check_entry_form_type_eligibility` resolves form type
    from `category@term` first, then from `title`, and checks membership
    in `ELIGIBLE_FORM_TYPES`.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `find_element(...)` to return synthetic category and
        title elements.
    category_term : str | None
        Value to return from the category element's `term` attribute; None
        simulates a missing category.
    title_text : str | None
        Value to return as the title element's `.text`; None simulates a
        missing title.
    expected : bool
        Expected eligibility outcome.

    Returns
    -------
    None
        The test passes if the function returns `expected` for the given
        combination of category and title.

    Raises
    ------
    AssertionError
        If the resolved eligibility does not match `expected`.
    """

    def fake_find_element(entry: Any, tag: str, ns: NameSpaceBindings | None) -> Any:
        if tag == "category":
            if category_term is None:
                return None
            return _DummyElement(text=None, attrs={"term": category_term})
        if tag == "title":
            if title_text is None:
                return None
            return _DummyElement(text=title_text)
        return None

    monkeypatch.setattr(esc, "find_element", fake_find_element)

    entry = object()
    result = esc.check_entry_form_type_eligibility(entry, None)
    assert result == expected


@pytest.mark.parametrize(
    "date_str,expected",
    [
        ("2020-01-05T00:00:00Z", True),  # strictly inside
        ("2020-01-01T00:00:00Z", True),  # exactly at start
        ("2020-01-10T00:00:00Z", False),  # exactly at end (exclusive)
        ("2019-12-31T23:59:59Z", False),  # just before start
        ("2020-01-11T00:00:00Z", False),  # after end
    ],
)
def test_within_validity_window_half_open_semantics(date_str: str, expected: bool) -> None:
    """
    Check that `within_validity_window` enforces half-open semantics
    `start <= ts < end` on UTC timestamps.

    Parameters
    ----------
    date_str : str
        ISO-like timestamp string used to construct the test timestamp.
    expected : bool
        Expected membership result for the given date.

    Returns
    -------
    None
        The test passes if `within_validity_window` returns `expected`.

    Raises
    ------
    AssertionError
        If the membership result does not match `expected`.
    """

    start = pd.Timestamp("2020-01-01T00:00:00Z")
    end = pd.Timestamp("2020-01-10T00:00:00Z")
    window: ValidityWindow = (start, end)
    ts = pd.to_datetime(date_str, utc=True)

    result = esc.within_validity_window(ts, window)
    assert result == expected


def test_evaluate_page_break_conditions_breaks_on_no_entries() -> None:
    """
    Ensure `evaluate_page_break_conditions` flags a page with no entries
    as terminal and logs a `loop_break` with a "no entries" message.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if:
            - the function returns True when `entries` is empty, and
            - an info-level `loop_break` with `msg="no entries"` is logged.

    Raises
    ------
    AssertionError
        If paging is not terminated or the expected log is missing.
    """

    logger = cast(InfraLogger, _DummyLogger())
    window: ValidityWindow = (
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2020-02-01", tz="UTC"),
    )
    run_data: RunData = {
        "ticker": "AAA",
        "validity_window": window,
        "candidate": "AAA",
        "name_space_bindings": None,
        "entry": object(),
        "logger": logger,
        "oldest_filing_date": pd.Timestamp("2100-01-10", tz="UTC"),
    }

    entries: List[Any] = []
    result = esc.evaluate_page_break_conditions(entries, run_data)
    assert result is True

    dummy_logger = cast(_DummyLogger, logger)
    assert dummy_logger.infos, "Expected a loop_break info log for no entries"
    event = dummy_logger.infos[0]
    assert event["event"] == "loop_break"
    assert event["kwargs"].get("msg") == "no entries"
    ctx = event["kwargs"].get("context", {})
    assert ctx.get("stage") == "break_condition_evaluation"
    assert ctx.get("ticker") == "AAA"


def test_evaluate_page_break_conditions_breaks_when_oldest_before_window() -> None:
    """
    Verify that `evaluate_page_break_conditions` stops paging when the
    oldest filing date seen so far is older than the validity window
    start.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if:
            - the function returns True when `oldest_filing_date` is less
              than `window_start`, and
            - an info-level `loop_break` is logged with the expected
              context fields.

    Raises
    ------
    AssertionError
        If the page is not flagged as terminal or the log payload does
        not match expectations.
    """

    logger = cast(InfraLogger, _DummyLogger())
    start = pd.Timestamp("2020-01-10", tz="UTC")
    end = pd.Timestamp("2020-02-01", tz="UTC")
    window: ValidityWindow = (start, end)
    oldest = pd.Timestamp("2020-01-01", tz="UTC")

    run_data: RunData = {
        "ticker": "BBB",
        "validity_window": window,
        "candidate": "BBB",
        "name_space_bindings": None,
        "entry": object(),
        "logger": logger,
        "oldest_filing_date": oldest,
    }
    entries: List[Any] = [object(), object()]

    result = esc.evaluate_page_break_conditions(entries, run_data)
    assert result is True

    dummy_logger = cast(_DummyLogger, logger)
    assert (
        dummy_logger.infos
    ), "Expected a loop_break info log for\
          oldest_filing_date < window_start"
    event = dummy_logger.infos[0]
    assert event["event"] == "loop_break"
    assert event["kwargs"].get("msg") == "oldest filing date older than window start"
    ctx = event["kwargs"].get("context", {})
    assert ctx.get("stage") == "break_condition_evaluation"
    assert ctx.get("ticker") == "BBB"
    assert ctx.get("oldest_filing_date") == oldest.isoformat()
    assert ctx.get("window_start") == start.isoformat()


def test_evaluate_page_break_conditions_continues_when_entries_and_within_window() -> None:
    """
    Confirm that `evaluate_page_break_conditions` allows paging to
    continue when there are entries and the oldest filing date is still
    within the validity window.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if:
            - the function returns False when entries exist and
              `oldest_filing_date >= window_start`, and
            - no `loop_break` info events are logged.

    Raises
    ------
    AssertionError
        If the page is incorrectly treated as terminal or if a loop-break
        event is logged.
    """

    logger = cast(InfraLogger, _DummyLogger())
    start = pd.Timestamp("2020-01-01", tz="UTC")
    end = pd.Timestamp("2020-02-01", tz="UTC")
    window: ValidityWindow = (start, end)
    oldest = pd.Timestamp("2020-01-15", tz="UTC")

    run_data: RunData = {
        "ticker": "CCC",
        "validity_window": window,
        "candidate": "CCC",
        "name_space_bindings": None,
        "entry": object(),
        "logger": logger,
        "oldest_filing_date": oldest,
    }
    entries: List[Any] = [object()]

    result = esc.evaluate_page_break_conditions(entries, run_data)
    assert result is False
    assert not cast(_DummyLogger, logger).infos
