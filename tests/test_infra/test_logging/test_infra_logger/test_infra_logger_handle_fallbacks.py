"""
Purpose
-------
Validate `handle_fallbacks` dispatch: emit exactly one WARNING per fallback flag
(level / format / dest) and in deterministic order.

Key behaviors
-------------
- Parametrizes all 8 combinations of fallback flags.
- Builds the expected `event` sequence (level → log_format → log_dest) and
  asserts exact equality with the actual calls.
- Asserts call-count equals the number of True flags.

Conventions
-----------
- `logger.emit` is patched on the instance with `autospec=True`.
- Assertions read `event` from keyword args (`call.kwargs["event"]`).
- No environment access is required; the test focuses purely on call routing.

Downstream usage
----------------
- Run locally with PyTest: `python -m pytest -q`
- CI will pick it up via the repo's standard `pytest` step.
"""

from typing import List
from unittest.mock import MagicMock
import pytest
from pytest_mock import MockerFixture
from infra.logging.infra_logger import InfraLogger, handle_fallbacks
from tests.test_infra.test_logging.test_infra_logger.infra_logger_testing_utils import init_logger_for_test


TEST_TUPLES: List[tuple[bool, str | None, bool, str | None, bool, str | None]] = [
    (False, None, False, None, False, None),
    (True, "FALLBACK_LOG_LEVEL", False, None, False, None),
    (False, None, True, "FALLBACK_LOG_FORMAT", False, None),
    (False, None, False, None, True, "FALLBACK_LOG_DEST"),
    (True, "FALLBACK_LOG_LEVEL", True, "FALLBACK_LOG_FORMAT", False, None),
    (True, "FALLBACK_LOG_LEVEL", False, None, True, "FALLBACK_LOG_DEST"),
    (False, None, True, "FALLBACK_LOG_FORMAT", True, "FALLBACK_LOG_DEST"),
    (True, "FALLBACK_LOG_LEVEL", True, "FALLBACK_LOG_FORMAT", True, "FALLBACK_LOG_DEST"),
]

@pytest.mark.parametrize(
    "fallback_level, expected_emit_level, fallback_format," \
    "expected_emit_format, fallback_dest, expected_emit_dest",
    TEST_TUPLES,
)
def test_handle_fallbacks(
        mocker: MockerFixture,
        fallback_level: bool,
        expected_level_event: str | None,
        fallback_format: bool,
        expected_format_event: str | None,
        fallback_dest: bool,
        expected_dest_event: str | None,
    ):
    """
    Ensure `handle_fallbacks` emits the correct events in order for a given flag set.

    Parameters
    ----------
    mocker : MockerFixture
        Used to patch `logger.emit` on the instance under test.
    fallback_level : bool
        Whether the level fallback flag is set (expect "FALLBACK_LOG_LEVEL" if True).
    expected_level_event : str | None
        Expected event name for the level fallback ("FALLBACK_LOG_LEVEL") or None.
    fallback_format : bool
        Whether the format fallback flag is set (expect "FALLBACK_LOG_FORMAT" if True).
    expected_format_event : str | None
        Expected event name for the format fallback ("FALLBACK_LOG_FORMAT") or None.
    fallback_dest : bool
        Whether the dest fallback flag is set (expect "FALLBACK_LOG_DEST" if True).
    expected_dest_event : str | None
        Expected event name for the dest fallback ("FALLBACK_LOG_DEST") or None.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the number of `emit` calls does not match the number of True flags
        or if the ordered list of emitted `event` values differs from expectation.

    Notes
    -----
    - Only the `event` keyword is asserted here as an identifier of the emit call.
    """

    logger: InfraLogger = init_logger_for_test()
    mock_emit: MagicMock = mocker.patch.object(logger, "emit", autospec=True)
    fallbacks = {
        "level": fallback_level,
        "log_format": fallback_format,
        "log_dest": fallback_dest
    }
    handle_fallbacks(logger, fallbacks)
    expected_events = build_expected_events_list(
        fallback_level,
        expected_level_event,
        fallback_format,
        expected_format_event,
        fallback_dest,
        expected_dest_event,
    )
    actual_events = [call.kwargs["event"] for call in mock_emit.call_args_list]
    assert mock_emit.call_count == len(expected_events)
    assert actual_events == expected_events


def build_expected_events_list(
        fallback_level: bool,
        expected_level_event: str | None,
        fallback_format: bool,
        expected_format_event: str | None,
        fallback_dest: bool,
        expected_dest_event: str | None,
    ) -> List[str]:
    """
    Construct the ordered list of expected event names for the given flags.

    Parameters
    ----------
    fallback_level : bool
        Include the level fallback event if True.
    expected_level_event : str | None
        The event name to include for level fallback ("FALLBACK_LOG_LEVEL").
    fallback_format : bool
        Include the format fallback event if True.
    expected_format_event : str | None
        The event name to include for format fallback ("FALLBACK_LOG_FORMAT").
    fallback_dest : bool
        Include the dest fallback event if True.
    expected_dest_event : str | None
        The event name to include for dest fallback ("FALLBACK_LOG_DEST").

    Returns
    -------
    List[str]
        The expected events in deterministic order: level → log_format → log_dest.

    Raises
    ------
    AssertionError
        (Not raised by this helper.) Test code should ensure that when a flag is True,
        the corresponding expected event is non-None.

    Notes
    -----
    - This helper centralizes expectation-building to keep the test body terse and clear.
    """

    expected_events: List = []
    if fallback_level:
        expected_events.append(expected_level_event)
    if fallback_format:
        expected_events.append(expected_format_event)
    if fallback_dest:
        expected_events.append(expected_dest_event)
    return expected_events
