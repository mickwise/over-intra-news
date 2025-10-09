"""
Purpose
-------
Validate `InfraLogger.write_entry` behavior for both stderr and file destinations.

Key behaviors
-------------
- Confirms stderr writes include a trailing newline and go to stderr (not stdout).
- Confirms file writes are appended (`"a"` mode) with UTF-8 encoding and newline termination.

Conventions
-----------
- Uses pytest's `capsys` fixture to capture stderr during the test.
- Uses pytest's `tmp_path` fixture to write to an ephemeral file per test.
- Normalizes newlines to `\n` when asserting file contents for cross-platform stability.

Downstream usage
----------------
Run with:
    `pytest -q tests/test_infra/test_logging/test_infra_logger/test_infra_logger_write_entry.py`
"""

import pathlib

from pytest import CaptureFixture

from infra.logging.infra_logger import InfraLogger
from tests.test_infra.test_logging.test_infra_logger.infra_logger_testing_utils import (
    TEST_FORMATTED_TEXT,
    init_logger_for_test,
)


def test_infra_logger_write_entry_stderr(capsys: CaptureFixture) -> None:
    """
    Verify that a single entry is printed to **stderr** with a trailing newline.

    Parameters
    ----------
    capsys : CaptureFixture
        Pytest capture fixture used to read the process's stderr after the call.

    Returns
    -------
    None
        The test asserts on captured stderr and has no return value.

    Raises
    ------
    AssertionError
        If the captured stderr does not equal the expected line plus newline.

    Notes
    -----
    - This validates the `"stderr"` destination branch of `write_entry`.
    """

    logger: InfraLogger = init_logger_for_test()
    logger.write_entry(TEST_FORMATTED_TEXT)
    captured_err: str = capsys.readouterr().err
    assert captured_err == f"{TEST_FORMATTED_TEXT}\n"


def test_infra_logger_write_entry_file(capsys: CaptureFixture, tmp_path: pathlib.Path) -> None:
    """
    Verify that entries are appended to a file in UTF-8 with newline termination.

    Parameters
    ----------
    capsys : CaptureFixture
        Pytest capture fixture, used here only to confirm nothing was written to stderr.
    tmp_path : pathlib.Path
        Temporary directory provided by pytest; the test writes a log file within it.

    Returns
    -------
    None
        The test asserts on file contents and captured stderr; no return value.

    Raises
    ------
    AssertionError
        If stderr is non-empty or if the file contents do not match the expected
        `"A\\nB\\n"` sequence (after normalizing newlines).

    Notes
    -----
    - Exercises the file branch of `write_entry` by passing a concrete file path.
    - Two sequential writes verify append semantics (`"a"` mode) rather than overwrite.
    - File text is normalized to `\\n` to avoid OS-specific newline differences.
    """

    test_file_path: pathlib.Path = tmp_path / "test_log.txt"
    logger: InfraLogger = init_logger_for_test(log_dest=str(test_file_path))
    logger.write_entry("A")
    logger.write_entry("B")
    capture_err: str = capsys.readouterr().err
    assert capture_err == ""
    assert test_file_path.read_text(encoding="utf-8").replace("\r\n", "\n") == "A\nB\n"
