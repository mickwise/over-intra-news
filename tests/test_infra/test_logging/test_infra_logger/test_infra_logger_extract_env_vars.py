"""
Purpose
-------
Validate `extract_env_vars` normalization and fallback behavior driven by environment
variables. Ensures level/format/destination are coerced as specified and that the
`fall_backs` flags are set only when defaults are applied.

Key behaviors
-------------
- Returns normalized triple: (LEVEL uppercased, format lowercased, dest unchanged).
- Applies defaults when env vars are missing or invalid and flips corresponding flags.
- Treats destination strictly: only the literal "stderr" is special; otherwise a file
  writability probe is attempted and may fall back to "stderr".

Conventions
-----------
- Tests isolate environment per-case using `monkeypatch.delenv`/`setenv`.
- File destination tests use `tmp_path` for safe, writable paths.
- The file-dest error path patches `open` at the module import path and asserts a single call.

Downstream usage
----------------
- Run locally via: `pytest -q tests`
- In CI, these tests run as part of the standard `pytest -q` step.
"""

import pathlib
from unittest.mock import MagicMock

from pytest import MonkeyPatch
from pytest_mock import MockerFixture

from infra.logging.infra_logger import extract_env_vars
from tests.test_infra.test_logging.test_infra_logger.infra_logger_testing_utils import (
    TEST_DEST,
    TEST_FORMAT,
    TEST_LOGGER_LEVEL,
)


def test_infra_logger_extract_env_vars_happy(monkeypatch: MonkeyPatch) -> None:
    """
    Validate that valid env vars are returned unchanged and no fallbacks are set.

    Parameters
    ----------
    monkeypatch : MonkeyPatch
        Used to set all three environment variables to valid values.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If any returned value deviates from the provided env vars, or if
        any fallback flag is unexpectedly True.

    Notes
    -----
    - Exercises the "everything valid" path: no probing, no coercion beyond
      the documented normalization (upper/lower on level/format).
    """

    set_up_env_vars(monkeypatch)
    fallbacks = {
        "level": False,
        "log_format": False,
        "log_dest": False,
    }
    log_level, log_format, log_dest = extract_env_vars(fallbacks)
    general_assertions(
        log_level=log_level,
        log_format=log_format,
        log_dest=log_dest,
        fallbacks=fallbacks,
    )


def test_infra_logger_extract_env_vars_defaults(monkeypatch: MonkeyPatch) -> None:
    """
    When env vars are absent, defaults are applied without marking fallbacks.

    Parameters
    ----------
    monkeypatch : MonkeyPatch
        Used to delete LOG_LEVEL, LOG_FORMAT, and LOG_DEST.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If defaults ("INFO", "json", "stderr") are not returned or fallback
        flags are set.

    Notes
    -----
    - Confirms the implicit defaults path works without treating it as a "fallback"
      scenario; flags remain False.
    """

    fallbacks = {
        "level": False,
        "log_format": False,
        "log_dest": False,
    }
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    monkeypatch.delenv("LOG_DEST", raising=False)
    log_level, log_format, log_dest = extract_env_vars(fallbacks)
    general_assertions(
        log_level=log_level,
        log_format=log_format,
        log_dest=log_dest,
        fallbacks=fallbacks,
        expected_level="INFO",
    )


def test_infra_logger_extract_env_vars_missing_level(monkeypatch: MonkeyPatch) -> None:
    """
    Invalid level coerces to "INFO" and flips the level fallback flag.

    Parameters
    ----------
    monkeypatch : MonkeyPatch
        Used to set LOG_LEVEL to an invalid value (empty string).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If level is not "INFO" or the level fallback flag is not True.

    Notes
    -----
    - Format and destination remain as provided and should not flip their flags.
    """

    fallbacks = {
        "level": False,
        "log_format": False,
        "log_dest": False,
    }
    set_up_env_vars(monkeypatch, level="")
    log_level, log_format, log_dest = extract_env_vars(fallbacks)
    general_assertions(
        log_level=log_level,
        log_format=log_format,
        log_dest=log_dest,
        fallbacks=fallbacks,
        expected_level="INFO",
        fallback_level=True,
    )


def test_infra_logger_extract_env_vars_missing_format(monkeypatch: MonkeyPatch) -> None:
    """
    Invalid format coerces to "json" and flips the format fallback flag.

    Parameters
    ----------
    monkeypatch : MonkeyPatch
        Used to set LOG_FORMAT to an invalid value (empty string).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If format is not "json" or the format fallback flag is not True.

    Notes
    -----
    - Level and destination remain as provided and should not flip their flags.
    """

    fallbacks = {
        "level": False,
        "log_format": False,
        "log_dest": False,
    }
    set_up_env_vars(monkeypatch, log_format="")
    log_level, log_format, log_dest = extract_env_vars(fallbacks)
    general_assertions(
        log_level=log_level,
        log_format=log_format,
        log_dest=log_dest,
        fallbacks=fallbacks,
        expected_format="json",
        fallback_format=True,
    )


def test_infra_logger_extract_env_vars_file_dest(
    tmp_path: pathlib.Path, monkeypatch: MonkeyPatch
) -> None:
    """
    Writable file destination is accepted as-is with no fallbacks.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Per-test temporary directory for creating a writable file path.
    monkeypatch : MonkeyPatch
        Used to set LOG_DEST to the temp file path.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the returned destination does not match the provided file path
        or any fallback flag is set.

    Notes
    -----
    - This exercises the positive file-probe path (no exception).
    """

    fallbacks = {
        "level": False,
        "log_format": False,
        "log_dest": False,
    }
    test_file_path: pathlib.Path = tmp_path / "test_log.txt"
    set_up_env_vars(monkeypatch, log_dest=str(test_file_path))
    log_level, log_format, log_dest = extract_env_vars(fallbacks)
    general_assertions(
        log_level=log_level,
        log_format=log_format,
        log_dest=log_dest,
        fallbacks=fallbacks,
        expected_dest=str(test_file_path),
    )


def test_infra_logger_extract_env_vars_file_dest_err(
    mocker: MockerFixture, monkeypatch: MonkeyPatch
) -> None:
    """
    Unwritable file destination falls back to "stderr" and flips the dest flag.

    Parameters
    ----------
    mocker : MockerFixture
        Used to patch `open` at the module import path to raise OSError.
    monkeypatch : MonkeyPatch
        Used to set LOG_DEST to an invalid path (or empty string).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If destination does not fall back to "stderr", the dest fallback flag
        is not True, or `open` is not called exactly once.

    Notes
    -----
    - Verifies the error branch of the file-probe logic.
    """

    fallbacks = {
        "level": False,
        "log_format": False,
        "log_dest": False,
    }
    set_up_env_vars(monkeypatch, log_dest="")
    mock_open: MagicMock = mocker.patch("infra.logging.infra_logger.open")
    mock_open.side_effect = OSError("boom")
    log_level, log_format, log_dest = extract_env_vars(fallbacks)
    general_assertions(
        log_level=log_level,
        log_format=log_format,
        log_dest=log_dest,
        fallbacks=fallbacks,
        expected_dest="stderr",
        fallback_dest=True,
    )
    mock_open.assert_called_once()


def general_assertions(
    log_level: str,
    log_format: str,
    log_dest: str,
    fallbacks: dict[str, bool],
    expected_level: str = TEST_LOGGER_LEVEL,
    expected_format: str = TEST_FORMAT,
    expected_dest: str = TEST_DEST,
    fallback_level: bool = False,
    fallback_format: bool = False,
    fallback_dest: bool = False,
) -> None:
    """
    Shared assertion helper for validating returned triple and fallback flags.

    Parameters
    ----------
    log_level : str
        Actual level returned by `extract_env_vars`.
    log_format : str
        Actual format returned by `extract_env_vars`.
    log_dest : str
        Actual destination returned by `extract_env_vars`.
    fallbacks : dict[str, bool]
        The mutable flag dict after the call.
    expected_level : str, default=TEST_LOGGER_LEVEL
        Expected normalized level.
    expected_format : str, default=TEST_FORMAT
        Expected normalized format.
    expected_dest : str, default=TEST_DEST
        Expected destination (either "stderr" or a file path).
    fallback_level : bool, default=False
        Expected state of the level fallback flag.
    fallback_format : bool, default=False
        Expected state of the format fallback flag.
    fallback_dest : bool, default=False
        Expected state of the dest fallback flag.

    Returns
    -------
    None

    Notes
    -----
    - Centralizes repetitive assertions and keeps case bodies minimal.
    """

    assert log_level == expected_level
    assert log_format == expected_format
    assert log_dest == expected_dest
    assert fallbacks == {
        "level": fallback_level,
        "log_format": fallback_format,
        "log_dest": fallback_dest,
    }


def set_up_env_vars(
    monkeypatch: MonkeyPatch,
    level: str = TEST_LOGGER_LEVEL,
    log_format: str = TEST_FORMAT,
    log_dest: str = TEST_DEST,
) -> None:
    """
    Reset and set the three logging env vars to the provided values.

    Parameters
    ----------
    monkeypatch : MonkeyPatch
        Used to delete and then set LOG_LEVEL, LOG_FORMAT, and LOG_DEST.
    level : str, default=TEST_LOGGER_LEVEL
        Level to be exported (should be uppercased by the function under test).
    log_format : str, default=TEST_FORMAT
        Format to be exported (should be lowercased by the function under test).
    log_dest : str, default=TEST_DEST
        Destination to be exported ("stderr" or a file path).

    Returns
    -------
    None

    Notes
    -----
    - Ensures a clean env state per test and avoids cross-test contamination.
    """

    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    monkeypatch.delenv("LOG_DEST", raising=False)
    monkeypatch.setenv("LOG_LEVEL", level)
    monkeypatch.setenv("LOG_FORMAT", log_format)
    monkeypatch.setenv("LOG_DEST", log_dest)
