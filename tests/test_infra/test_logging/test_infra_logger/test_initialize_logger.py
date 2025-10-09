"""
Purpose
-------
Validate the `initialize_logger` factory wiring in isolation: environment extraction,
optional run_id generation, defaulting of `run_meta`, and handoff to `handle_fallbacks`.

Key behaviors
-------------
- Calls `extract_env_vars` once and uses its (format, dest) to build the logger.
- Generates a run_id only when `run_id` is None; otherwise uses the provided value.
- Defaults `run_meta` to `{}` when None.
- Invokes `handle_fallbacks` exactly once on the constructed logger.

Conventions
-----------
- Collaborators are patched at `infra.logging.infra_logger.*` (unit under test is the module).
- Tests use stable constants from `infra_logger_testing_utils`; no I/O or env access.

Downstream usage
----------------
- Run locally with PyTest: `pytest -q tests/test_infra/test_logging/test_infra_logger`
- CI runs the same path as part of the project test suite.
"""

from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from infra.logging.infra_logger import InfraLogger, initialize_logger
from tests.test_infra.test_logging.test_infra_logger.infra_logger_testing_utils import (
    TEST_COMPONENT_NAME,
    TEST_DEST,
    TEST_FORMAT,
    TEST_LOGGER_LEVEL,
    TEST_RUN_ID,
    TEST_RUN_META,
)

TEST_RUN_ID_GENERATED: str = "generated_run_id"
MISSING_RUN_ID_META_TUPLES = [
    (None, TEST_RUN_ID_GENERATED, None, {}),
    (TEST_RUN_ID, TEST_RUN_ID, None, {}),
    (None, TEST_RUN_ID_GENERATED, TEST_RUN_META, TEST_RUN_META),
    (TEST_RUN_ID, TEST_RUN_ID, TEST_RUN_META, TEST_RUN_META),
]


def test_infra_logger_initialize_logger_happy(mocker: MockerFixture) -> None:
    """
    Ensure `initialize_logger` wires collaborators in the "all provided" case.

    Parameters
    ----------
    mocker : MockerFixture
        Used to patch `extract_env_vars`, `generate_run_id`, and `handle_fallbacks`.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If `extract_env_vars` is not called, if `generate_run_id` is called
        despite a provided run_id, or if `handle_fallbacks` is not invoked.

    Notes
    -----
    - Asserts that no run_id generation occurs when `run_id` is supplied.
    """

    mock_extract_env_vars, mock_generate_run_id, mock_handle_fallbacks = mock_infra_logger_helpers(
        mocker
    )
    initialize_logger(TEST_COMPONENT_NAME, TEST_LOGGER_LEVEL, TEST_RUN_ID, TEST_RUN_META)
    mock_extract_env_vars.assert_called_once()
    mock_generate_run_id.assert_not_called()
    mock_handle_fallbacks.assert_called_once()


@pytest.mark.parametrize(
    "input_run_id, expected_run_id, input_run_meta, expected_run_meta",
    MISSING_RUN_ID_META_TUPLES,
)
def test_infra_logger_initialize_logger_missing_variables(
    mocker: MockerFixture,
    input_run_id: str | None,
    expected_run_id: str,
    input_run_meta: dict[str, str] | None,
    expected_run_meta: dict[str, str],
) -> None:
    """
    Validate defaulting behavior when `run_id` and/or `run_meta` are omitted.

    Parameters
    ----------
    mocker : MockerFixture
        Used to patch `extract_env_vars`, `generate_run_id`, and `handle_fallbacks`.
    input_run_id : str | None
        The provided run_id or None to trigger generation.
    expected_run_id : str
        The expected run_id after initialization (provided or generated).
    input_run_meta : dict[str, str] | None
        The provided run metadata or None to trigger defaulting.
    expected_run_meta : dict[str, str]
        The expected run metadata after initialization.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If generation is not called when `run_id` is None, if `handle_fallbacks`
        is not invoked, or if the resulting `run_id`/`run_meta` differ from expectations.

    Notes
    -----
    - Also asserts that `generate_run_id` receives the `component_name`
      when generation is required.
    """

    mock_extract_env_vars, mock_generate_run_id, mock_handle_fallbacks = mock_infra_logger_helpers(
        mocker
    )
    logger: InfraLogger = initialize_logger(
        TEST_COMPONENT_NAME, TEST_LOGGER_LEVEL, input_run_id, input_run_meta
    )
    mock_extract_env_vars.assert_called_once()
    if input_run_id is None:
        mock_generate_run_id.assert_called_once_with(TEST_COMPONENT_NAME)
    else:
        mock_generate_run_id.assert_not_called()
    mock_handle_fallbacks.assert_called_once()
    assert logger.run_id == expected_run_id
    assert logger.run_meta == expected_run_meta


def mock_infra_logger_helpers(mocker: MockerFixture) -> tuple[MagicMock, MagicMock, MagicMock]:
    """
    Patch `initialize_logger` collaborators with safe defaults for tests.

    Parameters
    ----------
    mocker : MockerFixture
        Used to patch `extract_env_vars`, `generate_run_id`, and `handle_fallbacks`.

    Returns
    -------
    tuple[MagicMock, MagicMock, MagicMock]
        The mocks for (`extract_env_vars`, `generate_run_id`, `handle_fallbacks`), in that order.

    Raises
    ------
    None

    Notes
    -----
    - `extract_env_vars` returns a valid (level, format, dest) triple.
    - `generate_run_id` returns a deterministic sentinel.
    - `handle_fallbacks` is patched without side effects for simple call-count checks.
    """

    mock_extract_env_vars: MagicMock = mocker.patch(
        "infra.logging.infra_logger.extract_env_vars",
        return_value=(TEST_FORMAT, TEST_DEST),
    )
    mock_generate_run_id: MagicMock = mocker.patch(
        "infra.logging.infra_logger.generate_run_id",
        return_value=TEST_RUN_ID_GENERATED,
    )
    mock_handle_fallbacks: MagicMock = mocker.patch("infra.logging.infra_logger.handle_fallbacks")
    return mock_extract_env_vars, mock_generate_run_id, mock_handle_fallbacks
