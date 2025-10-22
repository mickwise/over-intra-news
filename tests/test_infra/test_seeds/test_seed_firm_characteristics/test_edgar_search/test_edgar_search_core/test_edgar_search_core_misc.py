"""
Purpose
-------
Unit tests for EDGAR search core helpers that parse form eligibility and XML feeds.

Key behaviors
-------------
- Validate category/title resolution and normalization in `check_entry_form_type_eligibility`.
- Validate XML root detection and namespace-aware vs. namespace-free entry extraction
  in `extract_entries_and_namespace`.

Conventions
-----------
- Tests use small MagicMocks and patching instead of real XML payloads or network I/O.
- Parametrized cases capture the minimal set of behaviorally distinct branches.

Downstream usage
----------------
These tests serve as guardrails against regressions in parsing behavior and
namespace handling within the EDGAR search core module.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from infra.seeds.seed_firm_characteristics.seed_evidence.edgar_search.edgar_search_core import (
    Entries,
    check_entry_form_type_eligibility,
    extract_entries_and_namespace,
)
from infra.seeds.seed_firm_characteristics.seed_evidence.edgar_search.edgar_search_utils import (
    NameSpaceBindings,
)

# fmt: off
from tests.test_infra.test_seeds.test_seed_firm_characteristics.test_edgar_search.\
    test_edgar_search_core.edgar_search_core_testing_utils import (
    make_category_elem,
    make_title_elem,
    patch_root_object,
)

# fmt: on


@pytest.mark.parametrize(
    "category_term,title_text,expected",
    [
        # 1) Category present & eligible (whitespace normalization)
        (" 8-K ", None, True),
        # 2) Category present & ineligible -> NO fallback to title in current code
        ("Not eligible", None, False),
        # 3) Category missing; title eligible by first token "10-K"
        (None, "10-K Annual Report", True),
        # 4) Neither present -> False
        (None, None, False),
    ],
)
def test_check_entry_form_type_eligibility(
    category_term: str, title_text: str, expected: bool, mocker: MockerFixture
) -> None:
    """
    Exercise category-first, title-fallback logic and normalization for form eligibility.

    Parameters
    ----------
    category_term : str
        The mocked value returned by the `<category>` element's "term" attribute, or None.
    title_text : str
        The mocked value for the `<title>` element's `.text`, or None.
    expected : bool
        Whether eligibility should be True or False for this case.
    mocker : pytest_mock.MockerFixture
        Used to patch `find_element` and inject category/title nodes.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the computed eligibility does not match `expected`.

    Notes
    -----
    - Category is authoritative when present; otherwise the first token of title is considered.
    - Whitespace and case are normalized prior to membership checks.
    """

    def side_effect(_entry: Any, tag: str, _: Any) -> MagicMock | None:
        if tag == "category":
            return None if category_term is None else make_category_elem(category_term)
        if tag == "title":
            return None if title_text is None else make_title_elem(title_text)
        return None

    mocker.patch(
        "infra.seeds.seed_firm_characteristics.seed_evidence"
        ".edgar_search.edgar_search_core.find_element",
        side_effect=side_effect,
    )
    result = check_entry_form_type_eligibility(MagicMock(), None)
    assert result is expected


@pytest.mark.parametrize(
    "root_local,expected_entries,nsmap_return,expected_namespace",
    [
        # 1) Not feed -> empty entries, empty namespace
        ("Not feed", [], {}, None),
        # 2) Atom feed with entry and namespace
        (
            "feed",
            [MagicMock()],
            {None: "http://www.w3.org/2005/Atom"},
            {"ns": "http://www.w3.org/2005/Atom"},
        ),
        # 3) Feed but no namespace
        (
            "feed",
            [MagicMock()],
            {},
            None,
        ),
    ],
)
def test_extract_entries_and_namespace(
    mocker: MockerFixture,
    root_local: str,
    expected_entries: Entries,
    nsmap_return: dict | None,
    expected_namespace: NameSpaceBindings | None,
) -> None:
    """
    Validate XML root detection and namespace handling for Atom feeds.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Used to patch `lxml.etree.fromstring` and `lxml.etree.QName`.
    root_local : str
        The local name of the root element (e.g., "feed" or something else).
    expected_entries : Entries
        The entries list that `findall`/`xpath` should return from the patched root.
    nsmap_return : dict | None
        The value to expose via `root.nsmap` (default namespace mapping or empty).
    expected_namespace : NameSpaceBindings | None
        The expected namespace dict returned by the function under test.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the returned `(entries, namespace)` does not match expectations.

    Notes
    -----
    - When the root is not `feed`, the function should return `([], None)`.
    - When a default namespace is present, entries are found via `.findall("ns:entry", ...)`
      and the function should return `({"ns": <uri>})`.
    - Without a default namespace, the XPath fallback is used and `None` is returned
      for the namespace bindings.
    """

    patch_root_object(
        mocker,
        root_local,
        expected_entries,
        nsmap_return,
    )
    assert extract_entries_and_namespace(MagicMock(), MagicMock()) == (
        expected_entries,
        expected_namespace,
    )
