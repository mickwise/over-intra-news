"""
Purpose
-------
Unit tests for the link-scanning helper that chooses an Atom `rel="alternate"` link,
persists Atom metadata into the raw record, and delegates to TXT/index parsing.

Key behaviors
-------------
- Returns delegate result when a valid `alternate` link with `href` exists.
- Skips non-alternate links; ignores alternates without `href`.
- Calls `set_atom_entry(...)` once with the chosen href and `handle_alternate_link(...)`
  with the same href and session.
- Returns None when no usable `alternate` href is present.

Conventions
-----------
- Links are simple MagicMocks exposing `.get("rel")` and `.get("href")`.
- All collaborator functions are patched and validated via call assertions.

Downstream usage
----------------
These tests ensure the “follow the alternate link” traversal remains deterministic
and side-effect free beyond the asserted delegate calls.
"""

from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from infra.seeds.seed_firm_characteristics.seed_evidence.edgar_search.edgar_filing_parse import (
    FilledLinkData,
)
from infra.seeds.seed_firm_characteristics.seed_evidence.edgar_search.edgar_search_core import (
    extract_data_from_links,
)
from infra.seeds.seed_firm_characteristics.seed_evidence.edgar_search.edgar_search_utils import (
    RunData,
)

# fmt: off
from tests.test_infra.test_seeds.test_seed_firm_characteristics.test_edgar_search.\
    edgar_search_testing_utils import (
    TEST_TICKER,
    TEST_VALIDITY_WINDOW,
)
from tests.test_infra.test_seeds.test_seed_firm_characteristics.test_edgar_search.\
    test_edgar_search_core.edgar_search_core_testing_utils import (
    TEST_RAW_RECORD,
    UPDATED_WITHIN_WINDOW,
    make_link,
    make_run_data,
    mock_helpers_extract_data_from_links,
)

# fmt: on


@pytest.mark.parametrize(
    "links,expected_href",
    [
        # 1) Single valid alternate
        ([make_link("alternate", "URL1")], "URL1"),
        # 2) Non-alternate first, then valid alternate
        ([make_link("self", "X"), make_link("alternate", "URL2")], "URL2"),
        # 3) First alternate missing href, second has href
        ([make_link("alternate", None), make_link("alternate", "URL3")], "URL3"),
        # 4) Multiple valid alternates -> should pick the first and stop
        ([make_link("alternate", "FIRST"), make_link("alternate", "SECOND")], "FIRST"),
    ],
)
def test_extract_data_from_links_alternate_href_found(
    mocker: MockerFixture,
    links: list[MagicMock],
    expected_href: str,
) -> None:
    """
    Verify that the first usable `rel="alternate"` link is chosen and delegated.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Used to patch `find_all_elements`, `set_atom_entry`, and `handle_alternate_link`.
    links : list[MagicMock]
        Candidate link elements to be returned by `find_all_elements`.
    expected_href : str
        The href value that should be selected and passed to delegates.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If selection, recorded metadata, or delegation calls deviate from expectations.

    Notes
    -----
    - The test asserts `set_atom_entry(...)` and `handle_alternate_link(...)` each run once
      with the chosen href and appropriate context objects.
    - The function’s return value must be exactly the value returned by `handle_alternate_link`.
    """

    run_data: RunData = make_run_data(TEST_TICKER, TEST_VALIDITY_WINDOW)
    session: MagicMock = MagicMock()
    handle_return_val = MagicMock()
    mock_find_all, mock_set_atom_entry, mock_handle_alt = mock_helpers_extract_data_from_links(
        mocker,
        links,
        handle_return_val=handle_return_val,
    )
    result: FilledLinkData | None = extract_data_from_links(
        run_data, TEST_RAW_RECORD, UPDATED_WITHIN_WINDOW, session
    )

    assert result is handle_return_val
    mock_find_all.assert_called_once_with(
        run_data["entry"], "link", run_data["name_space_bindings"]
    )

    # set_atom_entry called exactly once with the chosen href
    mock_set_atom_entry.assert_called_once_with(
        run_data["entry"],
        expected_href,
        UPDATED_WITHIN_WINDOW,
        TEST_RAW_RECORD,
        run_data["name_space_bindings"],
    )

    # handle_alternate_link called exactly once with the same href + logger + session
    mock_handle_alt.assert_called_once_with(
        expected_href,
        TEST_RAW_RECORD,
        run_data["logger"],
        session,
    )


@pytest.mark.parametrize(
    "links",
    [
        # 1) No links at all
        ([]),
        # 2) Only non-alternate links
        ([make_link("self", "X"), make_link("next", "Y")]),
        # 3) Alternates present but all missing href
        ([make_link("alternate", None)]),
    ],
)
def test_extract_data_from_links_no_valid_alternate(
    mocker: MockerFixture,
    links: list[MagicMock],
) -> None:
    """
    Ensure None is returned when no valid `rel="alternate"` href can be found.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Used to patch collaborators.
    links : list[MagicMock]
        Candidate links lacking a usable alternate-href combination.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If a delegate is called despite the absence of a usable alternate href.

    Notes
    -----
    - Asserts that neither `set_atom_entry` nor `handle_alternate_link` is invoked.
    """

    run_data = make_run_data("CRL", TEST_VALIDITY_WINDOW)
    session: MagicMock = MagicMock()
    _, mock_set_atom_entry, mock_handle_alt = mock_helpers_extract_data_from_links(
        mocker,
        links,
    )

    result = extract_data_from_links(run_data, TEST_RAW_RECORD, UPDATED_WITHIN_WINDOW, session)

    assert result is None
    mock_set_atom_entry.assert_not_called()
    mock_handle_alt.assert_not_called()
