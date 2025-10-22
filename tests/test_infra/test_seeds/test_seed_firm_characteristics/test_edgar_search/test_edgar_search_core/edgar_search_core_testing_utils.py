"""
Purpose
-------
Test helper utilities for EDGAR search unit tests. Provides small factory functions
to build realistic MagicMock objects, TypedDict payloads and patching helpers
that isolate lxml parsing and downstream delegates during tests.

Key behaviors
-------------
- Creates synthetic Atom sub-elements (category/title) with the minimal interface
  used by production code.
- Patches `lxml.etree.fromstring` and `lxml.etree.QName` to control XML root shape,
  namespacing, and returned entries without parsing real XML.
- Constructs canonical `RunData` and `FilledLinkData` dictionaries used by the module
  under test.
- Provides patch bundles that stub out delegate functions (e.g., `extract_data_from_links`,
  `build_mapping_evidence`, `find_all_elements`, `set_atom_entry`, `handle_alternate_link`)
  with predictable fakes.

Conventions
-----------
- `TEST_VALIDITY_WINDOW` reflects left-closed, right-open semantics and is reused
  by multiple tests for consistency.
- Patching helpers return the mocks they create so tests can assert on calls and
  arguments directly.

Downstream usage
----------------
- Run with: `python -m pytest -q tests`
- Part of CI/CD pipelines to ensure seed logic correctness.
"""

from unittest.mock import MagicMock

import pandas as pd
from pytest_mock import MockerFixture

import infra.seeds.seed_firm_characteristics.seed_evidence.edgar_search.edgar_search_core as core
from infra.seeds.seed_firm_characteristics.records.table_records import ValidityWindow
from infra.seeds.seed_firm_characteristics.seed_evidence.edgar_search.edgar_filing_parse import (
    FilledLinkData,
)
from infra.seeds.seed_firm_characteristics.seed_evidence.edgar_search.edgar_search_utils import (
    NameSpaceBindings,
    RunData,
)

# fmt: off
from tests.test_infra.test_seeds.test_seed_firm_characteristics.test_edgar_search\
    .edgar_search_testing_utils import (
    TEST_RAW_RECORD,
)

# fmt: on

UPDATED_WITHIN_WINDOW: pd.Timestamp = pd.Timestamp("2024-01-05T00:00:00Z")


def make_category_elem(term: str) -> MagicMock:
    """
    Create a MagicMock that behaves like an Atom `<category>` element.

    Parameters
    ----------
    term : str
        The value that should be returned when `.get("term")` is called.

    Returns
    -------
    MagicMock
        An object exposing `.get("term") -> term`, suitable for use in tests.

    Raises
    ------
    None

    Notes
    -----
    - Only the `.get("term")` accessor is implemented because the production code
      only reads that attribute when validating form eligibility.
    """

    category_element = MagicMock()
    category_element.get.return_value = term
    return category_element


def make_title_elem(text: str) -> MagicMock:
    """
    Create a MagicMock that behaves like an Atom `<title>` element.

    Parameters
    ----------
    text : str
        The text content to expose via the `.text` attribute.

    Returns
    -------
    MagicMock
        An object exposing `.text`, suitable for use in tests.

    Raises
    ------
    None

    Notes
    -----
    - The production code reads only `.text` and then splits on whitespace to find
      the first token (e.g., "10-K").
    """

    title_element = MagicMock()
    title_element.text = text
    return title_element


def patch_root_object(
    mocker: MockerFixture,
    root_local: str,
    entries: core.Entries,
    nsmap_return: NameSpaceBindings | None,
) -> None:
    """
    Patch lxml entry points.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Pytest-mock fixture used to apply patches.
    root_local : str
        The local name to set on the mocked `QName(...).localname` (e.g., "feed").
    entries : Entries
        The list of entry nodes that `.findall(...)` or `.xpath(...)` should return.
    nsmap_return : NameSpaceBindings | None
        The mapping to expose via `root.nsmap` (e.g., {None: "http://www.w3.org/2005/Atom"}).

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    - Patches `etree.fromstring` to return a MagicMock `root`.
    - Patches `etree.QName` so `QName(root).localname` equals `root_local`.
    - Configures `root.findall(...)`, `root.xpath(...)`, and `root.nsmap` directly,
      allowing tests to exercise both namespaced and non-namespaced paths without
      parsing real XML.
    """

    mock_root: MagicMock = MagicMock()
    mock_qname: MagicMock = MagicMock()
    mocker.patch.object(core.etree, "fromstring", return_value=mock_root)
    mocker.patch.object(core.etree, "QName", return_value=mock_qname)
    mock_qname.localname = root_local
    mock_root.findall.return_value = entries
    mock_root.xpath.return_value = entries
    mock_root.nsmap = nsmap_return


def make_run_data(ticker: str, validity_window: ValidityWindow) -> RunData:
    """
    Build a minimal `RunData` mapping for use in EDGAR search unit tests.

    Parameters
    ----------
    ticker : str
        Ticker symbol to set in the run context.
    validity_window : ValidityWindow
        Half-open window `(start_utc, end_utc)` used by filtering logic.

    Returns
    -------
    RunData
        TypedDict containing ticker, validity_window, name_space_bindings=None,
        a MagicMock logger, and a MagicMock entry.

    Raises
    ------
    None

    Notes
    -----
    - `name_space_bindings` is set to `None` because most tests do not depend on
      namespace-aware XPath.
    - The logger and entry are MagicMocks so tests can patch or assert on them if needed.
    """

    return {
        "ticker": ticker,
        "validity_window": validity_window,
        "name_space_bindings": None,
        "logger": MagicMock(),
        "entry": MagicMock(),
    }


def make_filled_link_data(
    accession_num: str,
    cik: str,
    form_type: str,
    filed_at: pd.Timestamp,
    company_name: str,
) -> FilledLinkData:
    """
    Construct a complete `FilledLinkData` dictionary for the happy path.

    Parameters
    ----------
    accession_num : str
        Canonical accession number (digits only).
    cik : str
        Zero-padded 10-character CIK.
    form_type : str
        Filing form type (e.g., "10-K"). Must be considered eligible by the caller.
    filed_at : pandas.Timestamp
        Filing date/time; typically UTC.
    company_name : str
        Parsed company name.

    Returns
    -------
    FilledLinkData
        Fully filled structure expected by evidence-building code.

    Raises
    ------
    None

    Notes
    -----
    - Mirrors the shape returned by the production parsing path so unit tests can
      skip network and lxml parsing entirely.
    """

    return {
        "accession_num": accession_num,
        "cik": cik,
        "form_type": form_type,
        "filed_at": filed_at,
        "company_name": company_name,
    }


def mock_helpers_extract_entry_data(
    mocker: MockerFixture,
    link_data: FilledLinkData,
) -> tuple[MagicMock, MagicMock]:
    """
    Patch delegates used by `extract_entry_data` and return their mocks.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Pytest-mock fixture used to apply patches.
    link_data : FilledLinkData
        The value to be returned by `extract_data_from_links(...)`.

    Returns
    -------
    tuple[MagicMock, MagicMock]
        (evidence_obj, mock_build_evidence)
        where:
            - `evidence_obj` is the MagicMock returned by `build_mapping_evidence`.
            - `mock_build_evidence` is the patched function object.

    Notes
    -----
    - Also patches `extract_data_from_links` to return `link_data`, enabling tests
      to focus solely on `extract_entry_data` control flow.
    """

    mocker.patch.object(core, "extract_data_from_links", return_value=link_data)
    mocker.patch.object(core, "generate_initial_raw_record", return_value=TEST_RAW_RECORD)

    evidence_obj: MagicMock = MagicMock()
    mock_build_evidence: MagicMock = mocker.patch.object(
        core, "build_mapping_evidence", return_value=evidence_obj
    )

    return evidence_obj, mock_build_evidence


def mock_helpers_extract_data_from_links(
    mocker: MockerFixture,
    links: list[MagicMock],
    handle_return_val: MagicMock | None = None,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    """
    Patch collaborators used by `extract_data_from_links` and return their mocks.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Pytest-mock fixture used to apply patches.
    links : list[MagicMock]
        The list of link objects to be returned by `find_all_elements(...)`.
    handle_return_val : MagicMock | None
        The value to return from `handle_alternate_link(...)` when a valid href is chosen.

    Returns
    -------
    tuple[MagicMock, MagicMock, MagicMock]
        (mock_find_all, mock_set_atom_entry, mock_handle_alternate_link)

    Raises
    ------
    None

    Notes
    -----
    - `find_all_elements` is patched to return `links` exactly as provided.
    - `set_atom_entry` is patched but returns None; tests assert it is called with
      the chosen href only when appropriate.
    - `handle_alternate_link` is patched to return `handle_return_val`, allowing
      tests to assert that the same object is returned by `extract_data_from_links`.
    """

    mock_find_all: MagicMock = mocker.patch.object(core, "find_all_elements", return_value=links)
    mock_set_atom_entry: MagicMock = mocker.patch.object(core, "set_atom_entry")
    mock_handle_alt: MagicMock = mocker.patch.object(
        core, "handle_alternate_link", return_value=handle_return_val
    )
    return mock_find_all, mock_set_atom_entry, mock_handle_alt


def make_link(rel: str | None, href: str | None) -> MagicMock:
    """
    Create a MagicMock that behaves like an Atom `<link>` element with `.get(...)`.

    Parameters
    ----------
    rel : str | None
        The value to return for `.get("rel")`.
    href : str | None
        The value to return for `.get("href")`.

    Returns
    -------
    unittest.mock.MagicMock
        An object whose `.get(key, default)` returns `rel` for "rel", `href` for "href",
        and `default` otherwise.

    Raises
    ------
    None

    Notes
    -----
    - This mirrors the minimal API consumed by the production code during link scanning.
    """

    link: MagicMock = MagicMock()

    def get_side_effect(key: str, default: None) -> str | None:
        if key == "rel":
            return rel
        if key == "href":
            return href
        return default

    link.get.side_effect = get_side_effect
    return link
