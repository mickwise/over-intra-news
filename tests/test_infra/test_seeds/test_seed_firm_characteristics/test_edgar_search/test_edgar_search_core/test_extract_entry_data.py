"""
Purpose
-------
Unit tests for the orchestration function that gates by timestamps, resolves
filing metadata via delegate, and constructs evidence.

Key behaviors
-------------
- Early-exit when `updated_ts` is outside the half-open validity window.
- Early-exit when filing `filed_at` is outside the half-open validity window.
- Happy path builds `MappingEvidence`; the `company_name` is passed through to
  `MappingEvidence.name`.

Conventions
-----------
- Delegates (`extract_data_from_links`, `build_mapping_evidence`) are patched so
  assertions can focus on control flow and wiring.
- Positional arguments to `build_mapping_evidence` are asserted to match the
  production signature ordering.

Downstream usage
----------------
These tests protect the window semantics and evidence wiring without performing
any network calls or parsing.
"""

from typing import TypeAlias
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pytest_mock import MockerFixture

import infra.seeds.seed_firm_characteristics.seed_evidence.edgar_search.edgar_search_core as core
from infra.seeds.seed_firm_characteristics.records.table_records import MappingEvidence
from infra.seeds.seed_firm_characteristics.seed_evidence.edgar_search.edgar_filing_parse import (
    FilledLinkData,
)
from infra.seeds.seed_firm_characteristics.seed_evidence.edgar_search.edgar_search_utils import (
    RunData,
)

# fmt: off
from tests.test_infra.test_seeds.test_seed_firm_characteristics.test_edgar_search.\
    edgar_search_testing_utils import (
    TEST_COMPANY_NAME,
    TEST_TICKER,
    TEST_VALIDITY_WINDOW,
)
from tests.test_infra.test_seeds.test_seed_firm_characteristics.test_edgar_search.\
    test_edgar_search_core.edgar_search_core_testing_utils import (
    TEST_RAW_RECORD,
    UPDATED_WITHIN_WINDOW,
    make_filled_link_data,
    make_run_data,
    mock_helpers_extract_entry_data,
)

# fmt: on

ExpectedReturnType: TypeAlias = MappingEvidence | None


@pytest.mark.parametrize(
    "updated_timestamp,updated_ok",
    [
        (pd.Timestamp("2024-01-01T00:00:00Z"), False),  # Before window
        (pd.Timestamp("2024-01-02T00:00:00Z"), True),  # Window start
        (UPDATED_WITHIN_WINDOW, True),  # Within window
    ],
)
def test_extract_entry_data_returns_none(
    mocker: MockerFixture, updated_timestamp: pd.Timestamp, updated_ok: bool
) -> None:
    """
    Return None when `updated_ts` is outside the window and verify delegate call policy.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Used to patch `extract_data_from_links`.
    updated_timestamp : pandas.Timestamp
        The entry-level updated timestamp under test.
    updated_ok : bool
        True when the timestamp should pass the first gate; otherwise False.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the function returns a non-None value or the delegate call policy is violated.

    Notes
    -----
    - When `updated_ts < window.start`, the function must exit before calling
      `extract_data_from_links`.
    - When `updated_ts >= window.start`, the delegate is called but returns None
      (patched), causing the function to return None.
    """

    run_data: RunData = make_run_data(TEST_TICKER, TEST_VALIDITY_WINDOW)

    mock_extract: MagicMock = mocker.patch.object(
        core,
        "extract_data_from_links",
        return_value=None,
    )

    result: ExpectedReturnType = core.extract_entry_data(run_data, updated_timestamp, MagicMock())
    assert result is None
    if updated_ok:
        mock_extract.assert_called_once()
    else:
        mock_extract.assert_not_called()  # early exit policy


@pytest.mark.parametrize(
    "filed_at",
    [
        pd.Timestamp("2024-01-01T00:00:00Z"),  # Before window
        pd.Timestamp("2024-01-10T00:00:00Z"),  # Window end
    ],
)
def test_extract_entry_data_filed_at_outside_window(
    mocker: MockerFixture, filed_at: pd.Timestamp
) -> None:
    """
    Return None when filing `filed_at` is outside the half-open validity window.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Used to patch `extract_data_from_links` to return a `FilledLinkData`.
    filed_at : pandas.Timestamp
        Filing timestamp on the boundary or before the window start.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the function does not return None for out-of-window `filed_at`.

    Notes
    -----
    - Validates both the "before start" and "at end" (right-open) boundary conditions.
    """

    run_data: RunData = make_run_data(TEST_TICKER, TEST_VALIDITY_WINDOW)
    link_data: FilledLinkData = make_filled_link_data(
        accession_num="0000000000",
        cik="0000123456",
        form_type="8-K",
        filed_at=filed_at,
        company_name="Any Co",
    )
    mocker.patch.object(
        core,
        "extract_data_from_links",
        return_value=link_data,
    )

    result = core.extract_entry_data(run_data, UPDATED_WITHIN_WINDOW, MagicMock())
    assert result is None


def test_extract_entry_data_happy_path(mocker: MockerFixture) -> None:
    """
    Builds evidence for the happy path and verifies that company name and other fields are
    wired into `build_mapping_evidence`, and that
    `core.extract_entry_data` returns the mocked evidence object.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Fixture to patch collaborators invoked by the function under test and to capture
        call arguments.

    Returns
    -------
    None
        The test passes if the function returns the mocked evidence and collaborators are
        called with the expected arguments.

    Raises
    ------
    AssertionError
        If the function does not return the mocked evidence object, or if
        `build_mapping_evidence` is not called exactly once with the expected signature.

    Notes
    -----
    - Patches `extract_data_from_links` to return a `FilledLinkData` whose `filed_at` lies
      inside the window.
    - Patches `generate_initial_raw_record` to supply a stable `raw_record`.
    - Asserts `build_mapping_evidence` is called with:
      (ticker, cik, filed_at, validity_window, source, raw_record, form_type,
      accession_num, company_name).
    - Confirms end-to-end wiring without any network or XML parsing.
    """

    filed_at_inside = pd.Timestamp("2024-01-05T00:00:00Z")
    run_data: RunData = make_run_data(TEST_TICKER, TEST_VALIDITY_WINDOW)

    link_data = make_filled_link_data(
        accession_num="0000000000",
        cik="0000123456",
        form_type="10-K",
        filed_at=filed_at_inside,
        company_name=TEST_COMPANY_NAME,
    )
    evidence_obj, mock_build_evidence = mock_helpers_extract_entry_data(mocker, link_data=link_data)

    result = core.extract_entry_data(run_data, UPDATED_WITHIN_WINDOW, MagicMock())
    assert result is evidence_obj

    # Sanity: evidence wiring uses the parsed fields and window
    mock_build_evidence.assert_called_once_with(
        TEST_TICKER,
        "0000123456",
        filed_at_inside,
        TEST_VALIDITY_WINDOW,
        core.SOURCE,
        TEST_RAW_RECORD,
        "10-K",
        link_data["accession_num"],
        link_data["company_name"],
    )
