"""
Purpose
-------
Provide small, reusable helpers for InfraLogger tests: a preconfigured test logger
and a deterministic UTC time patch.

Key behaviors
-------------
- Builds an `InfraLogger` with stable test metadata and explicit level/format/dest.
- Patches `infra.logging.infra_logger.dt.datetime.now` to a fixed UTC-aware instant.

Conventions
-----------
- Helpers are test-only; they perform no real I/O and avoid environment dependence.
- Time is always UTC; callers assert `datetime.now` is invoked with `timezone.utc`.
- Constants (e.g., TEST_COMPONENT_NAME / TEST_RUN_ID) are stable across tests.

Downstream usage
----------------
Import directly from test modules, e.g.:
`from tests.test_infra.test_logging.test_infra_logger.infra_logger_testing_utils
import init_logger_for_test, mock_datetime_now`
Use `init_logger_for_test()` to construct the subject, and `mock_datetime_now(mocker)`
to patch time before invoking `emit`.
"""

import datetime as dt
from typing import TypeAlias
from unittest.mock import MagicMock

from pytest_mock import MockerFixture

from infra.logging.infra_logger import InfraLogger, LogEntry

Context: TypeAlias = dict[str, str]

TEST_LOGGER_LEVEL: str = "DEBUG"
TEST_FORMAT: str = "json"
TEST_DEST: str = "stderr"
TEST_COMPONENT_NAME: str = "test_component"
TEST_RUN_ID: str = "test_run_id"
TEST_RUN_META: dict[str, str] = {"meta_key": "meta_value"}
TEST_NOW = dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
TEST_EVENT: str = "test_event"
TEST_MESSAGE: str = "test_message"
TEST_CONTEXT: Context = {"context_key": "context_value"}
TEST_FORMATTED_ENTRY: str = '{"formatted": "entry"}'
TEST_FORMATTED_TEXT: str = (
    f"{TEST_NOW.isoformat().replace('+00:00', 'Z')} "
    f"[DEBUG] {TEST_COMPONENT_NAME} {TEST_EVENT} - {TEST_MESSAGE} "
    + " ".join(f"{k}={v}" for k, v in TEST_CONTEXT.items())
)


def init_logger_for_test(
    log_level: str = "DEBUG",
    log_format: str = "json",
    log_dest: str = "stderr",
) -> InfraLogger:
    """
    Construct a minimal `InfraLogger` instance for unit tests.

    Parameters
    ----------
    log_level : str, default="DEBUG"
        Effective minimum level; DEBUG ensures all events pass threshold unless
        a test overrides it explicitly.
    log_format : str, default="json"
        Output format; not exercised here since formatting is stubbed.
    log_dest : str, default="stderr"
        Destination; not exercised here since writing is stubbed.

    Returns
    -------
    InfraLogger
        A logger configured with test constants for component, run_id, and run_meta.

    Raises
    ------
    None

    Notes
    -----
    - This helper avoids environment dependence and makes level/format/dest explicit.
    """

    return InfraLogger(
        component_name=TEST_COMPONENT_NAME,
        run_id=TEST_RUN_ID,
        run_meta=TEST_RUN_META,
        log_level=log_level,
        log_format=log_format,
        log_dest=log_dest,
    )


def mock_datetime_now(mocker: MockerFixture) -> MagicMock:
    """
    Patch the logger module's datetime to a fixed UTC-aware instant.

    Parameters
    ----------
    mocker : MockerFixture
        pytest-mock fixture used to patch `infra.logging.infra_logger.dt.datetime`.

    Returns
    -------
    MagicMock
        The patched `datetime` object whose `now` returns the fixed test instant.

    Raises
    ------
    None

    Notes
    -----
    - The test asserts `datetime.now` is called with `timezone.utc`, matching the
    implementation in `emit`.
    """

    mock_dt: MagicMock = mocker.patch("infra.logging.infra_logger.dt.datetime")
    mock_dt.now.return_value = TEST_NOW
    return mock_dt


def create_test_entry() -> LogEntry:
    return {
        "timestamp": TEST_NOW.isoformat().replace("+00:00", "Z"),
        "level": "DEBUG",
        "run_id": TEST_RUN_ID,
        "component": TEST_COMPONENT_NAME,
        "event": TEST_EVENT,
        "message": TEST_MESSAGE,
        "run_meta": TEST_RUN_META,
        "context": TEST_CONTEXT,
    }
