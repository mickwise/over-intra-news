"""
Purpose
-------
Unit-test the EDGAR filing index and submission TXT parsing helpers by exercising
happy paths, fallback behaviors, and failure modes with synthetic HTML/TXT inputs.

Key behaviors
-------------
- Verifies index-page fast-path extraction of form type, filed date (UTC), accession,
  CIK, and company name.
- Ensures TXT link resolution prefers the canonical complete-submission anchor and
  records fallback usage when only a generic .txt link exists.
- Confirms single-line TXT header parsing and normalization across a variety of inputs.

Conventions
-----------
- All timestamps are compared as UTC-aware pandas.Timestamp objects.
- HTML snippets are minimal and constructed to toggle presence/absence of nodes.
- TypedDict key assertions use Literal-based Key and the ALL_KEYS tuple to satisfy mypy.

Downstream usage
----------------
Serve as regression tests for edgar_filing_parse.
"""

import copy
from typing import Any, Literal
from unittest.mock import MagicMock

import pandas as pd
import pytest

from infra.seeds.seed_firm_characteristics.records.raw_record import RawRecord
from infra.seeds.seed_firm_characteristics.seed_evidence.edgar_search.edgar_filing_parse import (
    FilledLinkData,
    LinkData,
    extract_filing_url,
    parse_line,
    try_extract_evidence_from_index,
)

# fmt: off
from tests.test_infra.test_seeds.test_seed_firm_characteristics.\
    test_edgar_search.edgar_search_testing_utils import TEST_RAW_RECORD

# fmt: on

BASE_INDEX_URL: str = "https://example.com/edgar/filing-index.htm"
FILE_TXT_NAME: str = "0000000000.txt"


# Allowed LinkData keys, as a literal union:
Key = Literal["accession_num", "cik", "form_type", "filed_at", "company_name"]

# Type for the expectation dict in the param cases:
Expect = dict[Key, Any]

# Canonical ordered set of keys for the "unchanged" assertions:
ALL_KEYS: tuple[Key, ...] = ("accession_num", "cik", "form_type", "filed_at", "company_name")


def make_response_with_html(html: str, url: str) -> MagicMock:
    """
    Build a MagicMock that mimics a requests.Response carrying the given HTML at a URL.

    Parameters
    ----------
    html : str
        The raw HTML or XML payload to expose via .text and .content (UTF-8 encoded).
    url : str
        The final URL to assign to the mock response's .url.

    Returns
    -------
    unittest.mock.MagicMock
        A mock with attributes: content (bytes), text (str), url (str),
        status_code (int, 200), and headers (dict, empty).

    Raises
    ------
    None

    Notes
    -----
    - Use this helper to avoid coupling tests to requests.Response internals.
    """
    response: MagicMock = MagicMock()
    response.content = html.encode("utf-8")
    response.text = html
    response.url = url
    response.status_code = 200
    response.headers = {}
    return response


def build_index_html(
    include_form: bool,
    include_filed_at: bool,
    include_accession: bool,
    include_cik: bool,
    include_company: bool,
    form_token: str = "10-K",
    filed_at_value: str = "2024-01-05",
    accession_value: str = "0001193125-22-254310",
    cik_text_value: str = "0000123456 (CIK)",
    company_name_value: str = "Charles River (Filer)\n",
) -> str:
    """
    Construct a minimal SEC filing index HTML document with feature toggles.

    Parameters
    ----------
    include_form : bool
        Include the "Form X" element when True.
    include_filed_at : bool
        Include the Filing Date block when True.
    include_accession : bool
        Include the Accession Number block when True.
    include_cik : bool
        Include the CIK link text when True.
    include_company : bool
        Include the companyName span when True.
    form_token : str
        The form token placed after "Form " (e.g., 10-K). Default "10-K".
    filed_at_value : str
        Filing Date string to embed. Default "2024-01-05".
    accession_value : str
        Accession number (with hyphens) to embed. Default "0001193125-22-254310".
    cik_text_value : str
        Link text containing the numeric CIK token. Default "0000123456 (CIK)".
    company_name_value : str
        Company name text, often including "(Filer)\\n". Default "Charles River (Filer)\\n".

    Returns
    -------
    str
        A complete HTML page string for use by index-page parsers.

    Raises
    ------
    None

    Notes
    -----
    - Each boolean flag controls inclusion of the corresponding fragment and allows
      tests to simulate missing nodes that trigger None returns or IndexError.
    """
    form_html: str = (
        f'<div id="formName"><strong>Form {form_token}</strong></div>' if include_form else ""
    )
    filed_html: str = (
        '<div class="formContent">'
        '<div class="infoHead">Filing Date</div>'
        f'<div class="info">{filed_at_value}</div>'
        "</div>"
        if include_filed_at
        else ""
    )
    accession_html: str = (
        f"<div id='secNum'><strong>Accession Number:</strong> {accession_value}</div>"
        if include_accession
        else ""
    )
    cik_html: str = (
        f'<a href="/cgi-bin/browse-edgar?CIK=0000123456">{cik_text_value}</a>'
        if include_cik
        else ""
    )
    company_html: str = (
        f'<span class="companyName">{company_name_value}</span>' if include_company else ""
    )
    company_block: str = f'<div class="companyInfo">{cik_html + company_html}</div>'

    html: str = (
        f"<html><body>{form_html + filed_html + accession_html + company_block}{""}"
        "</body></html>"
    )
    return html


def build_filing_table_html(include_complete: bool, include_any: bool, txt_name: str) -> str:
    """
    Build a tableFile HTML snippet containing submission .txt link rows.

    Parameters
    ----------
    include_complete : bool
        When True, include the "Complete submission text file" row (canonical anchor).
    include_any : bool
        When True, include a generic .txt row to exercise fallback behavior.
    txt_name : str
        Relative href and link text to place in the table cells.

    Returns
    -------
    str
        An HTML page string wrapping the constructed table.

    Raises
    ------
    None

    Notes
    -----
    - Used to verify that extract_filing_url prefers the canonical anchor and,
      when absent, falls back to any .txt link while flagging the fallback.
    """
    rows: list[str] = []
    if include_complete:
        rows.append(
            "<tr>"
            "<td>Complete submission text file</td>"
            f'<td><a href="{txt_name}">{txt_name}</a></td>'
            "</tr>"
        )
    if include_any:
        rows.append("<tr>" "<td>Other</td>" f'<td><a href="{txt_name}">{txt_name}</a></td>' "</tr>")
    table_html = f'<table class="tableFile">{"".join(rows)}</table>'
    return f"<html><body>{table_html}</body></html>"


@pytest.mark.parametrize(
    "form_on,filed_on,accession_on,cik_on,company_on,expect_ok",
    [
        # 1) All fields present -> returns FilledLinkData
        (True, True, True, True, True, True),
        # 2) Missing accession -> safe None (normalize-space over empty returns '')
        (True, True, False, True, True, False),
        # 3) Missing CIK -> IndexError due to [0]
        (True, True, True, False, True, False),
        # 4) Missing company name -> IndexError due to [0]
        (True, True, True, True, False, False),
    ],
)
def test_try_extract_evidence_from_index(
    form_on: bool,
    filed_on: bool,
    accession_on: bool,
    cik_on: bool,
    company_on: bool,
    expect_ok: bool,
) -> None:
    """
    Validate index-page fast-path extraction across presence/absence combinations.

    Parameters
    ----------
    form_on : bool
        Toggle inclusion of the "Form" element.
    filed_on : bool
        Toggle inclusion of the "Filing Date" block.
    accession_on : bool
        Toggle inclusion of the "Accession Number" block.
    cik_on : bool
        Toggle inclusion of the CIK link text.
    company_on : bool
        Toggle inclusion of the company name span.
    expect_ok : bool | None
        Expected outcome: True for fully populated result, None when accession is
        intentionally missing (should return None), False when required nodes are
        missing and an IndexError is expected.

    Returns
    -------
    None
        Asserts on the returned FilledLinkData or raised error.

    Raises
    ------
    AssertionError
        If the function under test does not match the expected behavior.
    IndexError
        Intentionally provoked when required nodes are omitted and the implementation
        indexes with [0].

    Notes
    -----
    - This test encodes the contract that missing accession yields a clean None
      (to allow TXT fallback), while other missing fields surface IndexError.
    """
    html: str = build_index_html(
        include_form=form_on,
        include_filed_at=filed_on,
        include_accession=accession_on,
        include_cik=cik_on,
        include_company=company_on,
    )
    response: MagicMock = make_response_with_html(html, BASE_INDEX_URL)

    if expect_ok is True:
        result: FilledLinkData | None = try_extract_evidence_from_index(response)
        assert result is not None
        assert result["form_type"] == "10-K"
        assert result["filed_at"] == pd.Timestamp("2024-01-05", tz="UTC")
        assert result["accession_num"] == "000119312522254310"
        assert result["cik"] == "0000123456"
        assert result["company_name"] == "Charles River"
    else:
        result_none: FilledLinkData | None = try_extract_evidence_from_index(response)
        assert result_none is None


@pytest.mark.parametrize(
    "include_complete,include_any,expected_anchor,expected_fallback,expect_none",
    [
        # 1) Preferred "Complete submission" row exists
        (True, False, "complete-submission", False, False),
        # 2) Fallback to any .txt link
        (False, True, "any-txt-file", True, False),
        # 3) No .txt at all -> None
        (False, False, "", False, True),
    ],
)
def test_extract_filing_url(
    include_complete: bool,
    include_any: bool,
    expected_anchor: str,
    expected_fallback: bool,
    expect_none: bool,
) -> None:
    """
    Ensure TXT URL resolution and raw_record enrichment behave as specified.

    Parameters
    ----------
    include_complete : bool
        Include the canonical complete-submission row when True.
    include_any : bool
        Include a generic .txt link when True.
    expected_anchor : str
        Expected anchor_used to be recorded in raw_record
        (e.g., complete-submission or any-txt-file).
    expected_fallback : bool
        Whether used_fallback should be True or False.
    expect_none : bool
        When True, expect no .txt link to be found and result to be None.

    Returns
    -------
    None
        Asserts on the resolved URL suffix and raw_record['filing_page'] fields.

    Raises
    ------
    AssertionError
        If the extracted URL or raw_record annotations do not match expectations.

    Notes
    -----
    - When no .txt link exists, the test also asserts that no filing_page block was attached.
    """
    html: str = build_filing_table_html(include_complete, include_any, FILE_TXT_NAME)
    response: MagicMock = make_response_with_html(html, BASE_INDEX_URL)
    raw_record: RawRecord = copy.deepcopy(TEST_RAW_RECORD)
    logger: MagicMock = MagicMock()

    result: str | None = extract_filing_url(response, logger, raw_record)
    if expect_none:
        assert result is None
        assert "filing_page" not in raw_record
    else:
        assert result is not None
        assert result.endswith("/" + FILE_TXT_NAME)
        assert raw_record["filing_page"]["page_url"] == BASE_INDEX_URL
        assert raw_record["filing_page"]["anchor_used"] == expected_anchor
        assert raw_record["filing_page"]["used_fallback"] is expected_fallback


@pytest.mark.parametrize(
    "line,expect",
    [
        # 1) Accession (strip hyphens)
        ("ACCESSION NUMBER: 0001193125-22-254310", {"accession_num": "000119312522254310"}),
        # 2) CIK (zero-pad)
        ("CENTRAL INDEX KEY: 123456", {"cik": "0000123456"}),
        # 3) Allowed form type
        ("CONFORMED SUBMISSION TYPE: 10-K", {"form_type": "10-K"}),
        # 4) Disallowed form type -> ignored (no change)
        ("CONFORMED SUBMISSION TYPE: FOO", {}),
        # 5) Filed date to UTC
        ("FILED AS OF DATE: 2024-01-05", {"filed_at": pd.Timestamp("2024-01-05", tz="UTC")}),
        # 6) Company name with weird spaces -> normalized by parse_line via normalize_line
        ("COMPANY CONFORMED NAME:\tCharles\xa0River", {"company_name": "Charles River"}),
        # 7) XML-ish line ignored
        ("<DOCUMENT>", {}),
        # 8) No colon -> ignored
        ("THIS IS NOT A KEY VALUE", {}),
    ],
)
def test_parse_line(
    line: str,
    expect: Expect,
) -> None:
    """
    Check single-line TXT header parsing and normalization into LinkData.

    Parameters
    ----------
    line : str
        A raw header line (e.g., ACCESSION NUMBER: ..., CENTRAL INDEX KEY: ..., etc.).
    expect : dict[Literal["accession_num","cik","form_type","filed_at","company_name"], Any]
        Mapping of the field(s) expected to be updated by parse_line for this case.

    Returns
    -------
    None
        Asserts that updated fields match 'expect' and all other fields remain None.

    Raises
    ------
    AssertionError
        If LinkData does not reflect the expected mutation after parsing the line.

    Notes
    -----
    - Lines starting with '<' or missing a colon are ignored.
    - Disallowed form types leave LinkData unchanged.
    - Tabs and non-breaking spaces are normalized before parsing, and dates are
      parsed into UTC-aware pandas.Timestamp objects.
    """
    link_data: LinkData = {
        "accession_num": None,
        "cik": None,
        "form_type": None,
        "filed_at": None,
        "company_name": None,
    }

    parse_line(line, link_data)

    for key, exp in expect.items():
        assert link_data[key] == exp

    for key in ALL_KEYS:
        if key not in expect:
            assert link_data[key] is None
