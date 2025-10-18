"""
Purpose
-------
Resolve an EDGAR Atom entry’s filing index page to the corresponding “complete submission”
TXT, parse a minimal header subset, and return lightweight metadata for evidence-building.
Also enriches the provided `RawRecord` with filing-page/HTTP provenance and emits
structured logs for observability.

Key behaviors
-------------
- Locates the “Complete submission text file” link on an SEC filing index page; if absent,
  falls back to any `.txt` link and logs a WARNING.
- Fetches the TXT, records request/response metadata into `raw_record['raw_http']`,
  and parses a small set of header lines.
- Filters forms against `ELIGIBLE_FORM_TYPES` before returning parsed results.
- Emits DEBUG/INFO/WARNING/ERROR events via the supplied `InfraLogger`.

Conventions
-----------
- Timezone: all parsed dates (e.g., "FILED AS OF DATE") are converted to UTC.
- Header parsing: lines are normalized (tabs/non-breaking spaces collapsed, whitespace squashed).
- Key matching is case-insensitive; values are trimmed; accession hyphens are removed.
- `ELIGIBLE_FORM_TYPES` is a `frozenset[str]` and should be treated as immutable.

Downstream usage
----------------
- Call `handle_alternate_link(link_url, raw_record, logger)` with the Atom entry’s
  `rel="alternate"` URL to obtain `LinkData` or `None`.
- Use `extract_filing_url(...)` when you already hold an index page `Response` and need only
  the TXT URL resolution (with logging and `raw_record['filing_page']` enrichment).
- The caller is expected to integrate the returned `LinkData` with window gating
  and evidence construction.
"""

import urllib
from typing import Any, List, TypedDict

import pandas as pd
import requests
from lxml import etree

from infra.logging.infra_logger import InfraLogger
from infra.seeds.seed_firm_characteristics.edgar_search.edgar_search_utils import (
    ELIGIBLE_FORM_TYPES,
    REQUEST_DELAY_SECONDS,
)
from infra.seeds.seed_firm_characteristics.records.raw_record import RawRecord
from infra.utils.requests_utils import make_request

XPATH_FORM_TYPE: str = '//div[@id="formName"]/strong/text()'
XPATH_FILED_AT: str = (
    '//div[@class="formContent"]//div[@class="infoHead" and '
    'normalize-space()="Filing Date"]/following-sibling::div[contains(@class,"info")][1]/text()'
)
XPATH_ACCESSION_NUM: str = (
    "normalize-space(//div[@id='secNum']/strong/following-sibling::text()[1])"
)
XPATH_CIK: str = '//div[contains(@class,"companyInfo")]//a[contains(@href,"CIK=")]/text()'
XPATH_COMPANY_NAME: str = (
    '//div[contains(@class,"companyInfo")]//span[contains(@class,"companyName")]/text()[1]'
)


class LinkData(TypedDict):
    """
    Purpose
    -------
    Minimal metadata parsed from a submission TXT header used to build `MappingEvidence`.

    Fields
    ------
    accession_num : str | None
        Accession number with hyphens removed (e.g., '000119312522254310').
    cik : str | None
        Zero-padded 10-character CIK string.
    form_type : str | None
        Parsed 'CONFORMED SUBMISSION TYPE'; must be in `ELIGIBLE_FORM_TYPES`.
    filed_at : pandas.Timestamp | None
        Parsed 'FILED AS OF DATE' converted to UTC.
    company_name : str | None
        Parsed 'COMPANY CONFORMED NAME' as-is (post normalization).

    Notes
    -----
    - A `LinkData` is considered complete when all fields except `company_name` are non-None.
    """

    accession_num: str | None
    cik: str | None
    form_type: str | None
    filed_at: pd.Timestamp | None
    company_name: str | None


class FilledLinkData(TypedDict):
    """
    Purpose
    -------
    A `LinkData` that has been verified to be complete.

    Fields
    ------
    accession_num : str
        Accession number with hyphens removed (e.g., '000119312522254310').
    cik : str
        Zero-padded 10-character CIK string.
    form_type : str
        Parsed 'CONFORMED SUBMISSION TYPE'; must be in `ELIGIBLE_FORM_TYPES`.
    filed_at : pandas.Timestamp
        Parsed 'FILED AS OF DATE' converted to UTC.
    company_name : str | None
        Parsed 'COMPANY CONFORMED NAME' as-is (post normalization).

    Notes
    -----
    - A `FilledLinkData` is guaranteed to have all fields except `company_name` non-None.
    """

    accession_num: str
    cik: str
    form_type: str
    filed_at: pd.Timestamp
    company_name: str | None


def handle_alternate_link(
    link_url: str, raw_record: RawRecord, logger: InfraLogger, session: requests.Session
) -> FilledLinkData | None:
    """
    Resolve a filing’s index page, attempt fast metadata extraction,
    and fall back to the full TXT submission when needed.

    Parameters
    ----------
    link_url : str
        The Atom entry’s `rel="alternate"` URL for the filing’s index page (`*-index.htm`).
    raw_record : RawRecord
        Mutable record updated with HTTP metadata when the TXT fallback path is used.
    logger : InfraLogger
        Logger for diagnostics (e.g., `'txt_fetched'`).
    session : requests.Session
        Session used for all HTTP requests (headers/connection reuse).

    Returns
    -------
    FilledLinkData | None
        Populated metadata when index-page extraction or TXT fallback succeeds;
        `None` if the filing cannot be resolved.

    Raises
    ------
    requests.HTTPError
        For non-retryable HTTP statuses surfaced by `make_request`.
    requests.RequestException
        For transport-level errors from `requests`.
    ValueError
        If downstream parsing (e.g., date) fails.

    Notes
    -----
    - Control flow:
        1) GET the index page via `make_request(...)`.
        2) Try `try_extract_evidence_from_index(response)` (fast path).
        3) If incomplete, derive the TXT URL via `extract_filing_url(...)`
           and GET via `make_request(...)`.
        4) Populate `raw_record["raw_http"]` with request/response metadata;
           parse via `extract_data_from_txt_file(...)`.
    - Honors global pacing (e.g., `REQUEST_DELAY_SECONDS`) and `User-Agent`.
    - The fast path avoids multi-MB TXT downloads; TXT remains a robust fallback.
    """

    with make_request(
        link_url, expect_json=False, sleep_time=REQUEST_DELAY_SECONDS, session=session
    ) as response:

        # Try to extract data directly from the filing index page first
        possible_data: FilledLinkData | None = try_extract_evidence_from_index(response)
        if possible_data is not None:
            raw_record["filing_page"] = {
                "page_url": response.url,
                "anchor_used": "index-page-extraction",
                "used_fallback": False,
            }
            logger.debug(
                "index_page_extracted",
                context={
                    "stage": "edgar_search",
                    "request_url": link_url,
                    "status": response.status_code,
                    "response_url": response.url,
                },
            )
            return possible_data

        # Fallback to fetching the complete submission TXT file
        absolute_url: str | None = extract_filing_url(response, logger, raw_record)
        if absolute_url is None:
            return None
        with make_request(
            absolute_url, expect_json=False, sleep_time=REQUEST_DELAY_SECONDS, session=session
        ) as txt_response:
            raw_record["raw_http"] = {
                "request_url": absolute_url,
                "response_url": txt_response.url,
                "status_code": txt_response.status_code,
                "resp_headers": dict(txt_response.headers),
            }
            logger.debug(
                "txt_fetched",
                context={
                    "stage": "edgar_search",
                    "request_url": absolute_url,
                    "status": txt_response.status_code,
                    "response_url": txt_response.url,
                },
            )
            return extract_data_from_txt_file(txt_response)


def try_extract_evidence_from_index(
    response: requests.Response,
) -> FilledLinkData | None:
    """
    Try extracting core filing metadata directly from the SEC filing *index page*
    HTML and build a `FilledLinkData` record.

    Parameters
    ----------
    response : requests.Response
        HTTP response for the filing index page (e.g., `*-index.htm`).
        Its `.content` is parsed with lxml.

    Returns
    -------
    FilledLinkData | None
        A populated record with `form_type`, `filed_at` (UTC), `accession_num` (digits only),
        `cik` (zero-padded to 10), and `company_name` when all fields are
        present except company name; otherwise `None`
        so the caller can fall back to the TXT path.

    Raises
    ------
    IndexError
        If any required XPath returns an empty node-set and the implementation indexes `[0]`.
    ValueError
        If `filed_at` cannot be parsed to a UTC timestamp.

    Notes
    -----
    - Sources (by XPath):
        - Form Type: `//div[@id="formName"]/strong/text()` →
          split off the leading `"Form "` to yield `"8-K"`, `"10-Q"`, etc.
        - Filing Date: `//div[@class="formContent"]//div[@class="infoHead" and
          normalize-space()='Filing Date']/following-sibling
          ::div[contains(@class,'info')][1]/text()`
        - Accession Number: `normalize-space(//div[@id='secNum']
          /strong/following-sibling::text()[1])`
        - CIK (link text): `//div[contains(@class,'companyInfo')]//a[contains(@href,'CIK=')]/text()`
        → first digit run → `zfill(10)`.
        - Company Name: `//div[contains(@class,'companyInfo')]//span[contains(@class,'companyName')]
        /text()[1]` → strip `(Filer)` and whitespace.
    - Normalization:
        - `accession_num`: remove hyphens for the canonical ID.
        - `filed_at`: convert to UTC.
        - `cik`: zero-pad to 10 characters.
    - Returns `None` on any missing field to enable a clean TXT fallback.
    """

    html_root: Any = etree.HTML(response.content)
    form_type: str | None = html_root.xpath(XPATH_FORM_TYPE)[0].split(" ")[1]
    if not form_type:
        return None
    filed_at: str | None = html_root.xpath(XPATH_FILED_AT)[0]
    if not filed_at:
        return None
    accession_num: str | None = html_root.xpath(XPATH_ACCESSION_NUM)
    if not accession_num:
        return None
    cik: str | None = html_root.xpath(XPATH_CIK)[0].split(" ")[0]
    if not cik:
        return None
    company_name: str | None = (
        html_root.xpath(XPATH_COMPANY_NAME)[0].replace("(Filer)\n", "").strip()
    )
    return create_filled_link_data(
        {
            "accession_num": accession_num.replace("-", ""),
            "cik": cik.zfill(10),
            "form_type": form_type,
            "filed_at": pd.to_datetime(filed_at, utc=True),
            "company_name": company_name,
        }
    )


def extract_filing_url(
    response: requests.Response, logger: InfraLogger, raw_record: RawRecord
) -> str | None:
    """
    Discover the submission TXT URL from a filing index page and annotate `raw_record`.

    Parameters
    ----------
    response : requests.Response
        HTTP response for the filing index page.
    logger : InfraLogger
        Logger for WARNING ('fallback_to_any_txt') and ERROR ('txt_href_missing') events.
    raw_record : RawRecord
        Mutable record; `raw_record['filing_page']` is populated with page URL,
        which anchor strategy was used, and a fallback flag.

    Returns
    -------
    str | None
        Absolute URL to the submission TXT, or None if no suitable link is found.

    Notes
    -----
    - Prefers the row labeled 'Complete submission text file'.
    - If missing, falls back to any link ending with `.txt` and logs a WARNING.
    - When neither is found, logs an ERROR and returns None.
    """

    anchor_used: str = "complete-submission"
    used_fallback: bool = False
    html_root: Any = etree.HTML(response.content)
    complete_sub_hrefs: List[str] = html_root.xpath(
        '//table[contains(@class,"tableFile")]//tr[td[normalize-space()\
            ="Complete submission text file"]]//a/@href'
    )
    if len(complete_sub_hrefs) == 0:
        logger.warning(
            "fallback_to_any_txt",
            context={"stage": "edgar_search", "filing_index_url": response.url},
        )
        anchor_used = "any-txt-file"
        complete_sub_hrefs = html_root.xpath(
            '//table[contains(@class,"tableFile")]//a[contains(@href, ".txt")]/@href'
        )
        if len(complete_sub_hrefs) == 0:
            logger.error(
                "txt_href_missing",
                context={"stage": "edgar_search", "filing_index_url": response.url},
            )
            return None
        used_fallback = True
    absolute_url: str = urllib.parse.urljoin(response.url, complete_sub_hrefs[0])
    raw_record["filing_page"] = {
        "page_url": response.url,
        "anchor_used": anchor_used,
        "used_fallback": used_fallback,
    }
    return absolute_url


def extract_data_from_txt_file(txt_response: requests.Response) -> FilledLinkData | None:
    """
    Parse a submission TXT’s header lines into `LinkData`, short-circuiting when complete.

    Parameters
    ----------
    txt_response : requests.Response
        HTTP response containing the raw submission TXT.

    Returns
    -------
    FilledLinkData | None
        A fully filled structure that is returned once all required fields
        (accession_num, cik, form_type, filed_at) are present; otherwise None.

    Notes
    -----
    - Splits on lines and feeds each through `parse_line(...)`.
    - Stops early as soon as `check_data_completeness(...)` is satisfied.
    """

    txt_lines: List[str] = txt_response.text.splitlines()
    link_data: LinkData = {
        "accession_num": None,
        "cik": None,
        "form_type": None,
        "filed_at": None,
        "company_name": None,
    }
    for line in txt_lines:
        parse_line(line, link_data)
        if check_data_completeness(link_data):
            return create_filled_link_data(link_data)
    return None


def parse_line(line: str, link_data: LinkData) -> None:
    """
    Consume one normalized header line and update `link_data` in place when recognized.

    Parameters
    ----------
    line : str
        Raw line from the TXT; will be normalized internally.
    link_data : LinkData
        Mutable structure to write recognized fields into.

    Returns
    -------
    None

    Notes
    -----
    - Ignores XML-ish lines starting with '<'.
    - Matches on the colon delimiter; keys are upper-cased before comparison.
    - Recognized keys:
    - 'ACCESSION NUMBER' → hyphens removed
    - 'CENTRAL INDEX KEY' → zero-padded to 10 digits
    - 'CONFORMED SUBMISSION TYPE' → must be in `ELIGIBLE_FORM_TYPES` or the line is ignored
    - 'FILED AS OF DATE' → parsed to UTC Timestamp
    - 'COMPANY CONFORMED NAME' → stored as-is after normalization
    """

    line = normalize_line(line)
    if line.startswith("<"):
        return None
    col_idx: int = line.find(":")
    if col_idx == -1:
        return None
    key: str = line[:col_idx].strip().upper()
    value: str = line[col_idx + 1 :].strip()
    if key == "ACCESSION NUMBER":
        link_data["accession_num"] = value.replace("-", "")
    elif key == "CENTRAL INDEX KEY":
        link_data["cik"] = value.zfill(10)
    elif key == "CONFORMED SUBMISSION TYPE":
        if value not in ELIGIBLE_FORM_TYPES:
            return None
        link_data["form_type"] = value
    elif key == "FILED AS OF DATE":
        link_data["filed_at"] = pd.to_datetime(value, utc=True)
    elif key == "COMPANY CONFORMED NAME":
        link_data["company_name"] = value
    return None


def normalize_line(line: str) -> str:
    """
    Normalize a TXT header line for robust key/value parsing.

    Parameters
    ----------
    line : str
        Raw line (may include tabs or non-breaking spaces).

    Returns
    -------
    str
        Line with tabs and non-breaking spaces converted to spaces and with
        intra-line whitespace collapsed to single spaces.

    Notes
    -----
    - Intended to run prior to colon-based key/value splitting.
    """

    line = line.replace("\t", " ").replace("\xa0", " ")
    line = " ".join(line.split())
    return line


def check_data_completeness(link_data: LinkData) -> bool:
    """
    Determine whether a `LinkData` has all required fields to proceed.

    Parameters
    ----------
    link_data : LinkData
        Working structure populated by `parse_line(...)`.

    Returns
    -------
    bool
        True if `accession_num`, `cik`, `form_type`, and `filed_at` are all non-None.

    Notes
    -----
    - `company_name` is optional and does not affect completeness.
    """

    return all(
        [
            link_data["accession_num"] is not None,
            link_data["cik"] is not None,
            link_data["form_type"] is not None,
            link_data["filed_at"] is not None,
        ]
    )


def create_filled_link_data(link_data: LinkData) -> FilledLinkData:
    """
    Convert a complete `LinkData` to a `FilledLinkData`.

    Parameters
    ----------
    link_data : LinkData
        A `LinkData` that has been verified to be complete.

    Returns
    -------
    FilledLinkData
        The same data as `link_data`, but with type guarantees.

    Raises
    ------
    None

    Notes
    -----
    - This function is a type-level assertion and does not modify the data.
    """

    assert link_data["accession_num"] is not None
    assert link_data["cik"] is not None
    assert link_data["form_type"] is not None
    assert link_data["filed_at"] is not None
    return {
        "accession_num": link_data["accession_num"],
        "cik": link_data["cik"],
        "form_type": link_data["form_type"],
        "filed_at": link_data["filed_at"],
        "company_name": link_data["company_name"],
    }
