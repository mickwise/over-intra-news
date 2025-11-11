"""
Purpose
-------
Resolve an EDGAR Atom entry’s filing index page, parse a minimal header subset
from the HTML, and return lightweight, typed metadata for evidence-building.
Also enriches `RawRecord` with filing-page/HTTP provenance.

Key behaviors
-------------
- Parses index pages for (form_type, filed_at, accession_num, cik,
  company_name).
- For 8-K / 8-K/A filings, parses the “Items” block and filters item numbers
  through `ELIGIBLE_8K_ITEMS`, carrying both item codes and descriptions.
- Normalizes accession (strip hyphens), zero-pads CIK, converts dates to UTC.
- Records basic filing-page metadata into `entry_record`.
- Emits structured logs (DEBUG/INFO/WARNING/ERROR) via `InfraLogger`.

Conventions
-----------
- Timezone: all timestamps are treated as UTC.
- Accessions are stored as 18-digit strings without hyphens.
- CIK is stored as a zero-padded 10-character string.
- `ELIGIBLE_FORM_TYPES` gates which forms are considered valid; for 8-K forms,
  `ELIGIBLE_8K_ITEMS` further restricts which items are treated as evidence.

Downstream usage
----------------
- Call `handle_alternate_link(...)` with the entry’s `rel="alternate"` URL to
  obtain a `FilledLinkData` or `None`.
- Returned `FilledLinkData` feeds the `MappingEvidence` builder upstream.
"""

from typing import Any, List, TypedDict

import pandas as pd
import requests
from lxml import etree

from infra.logging.infra_logger import InfraLogger
from infra.seeds.seed_evidence.edgar_search.edgar_config import (
    ELIGIBLE_8K_ITEMS,
    REQUEST_DELAY_SECONDS,
    XPATH_8K_ITEMS,
    XPATH_ACCESSION_NUM,
    XPATH_CIK,
    XPATH_COMPANY_NAME,
    XPATH_FILED_AT,
    XPATH_FORM_TYPE,
)
from infra.seeds.seed_evidence.edgar_search.edgar_search_utils import RunData
from infra.seeds.seed_evidence.records.raw_record import RawRecord
from infra.utils.requests_utils import make_request


class LinkData(TypedDict):
    """
    Purpose
    -------
    Container for partially parsed filing metadata from an index page, used as
    the working structure while fields are being discovered.

    Key behaviors
    -------------
    - Holds optional fields that may be filled incrementally.
    - Acts as the input to `create_filled_link_data(...)` once complete.
    - Carries optional 8-K item metadata when available.

    Fields
    ------
    accession_num : str | None
        Accession with hyphens removed (18 digits) or None if unknown.
    cik : str | None
        Zero-padded 10-character CIK, or None if unknown.
    form_type : str | None
        Filing form (e.g., 10-K/10-Q/8-K), or None.
    filed_at : pandas.Timestamp | None
        Filing date (UTC), or None.
    company_name : str | None
        Company name as normalized from source, or None.
    items_8k : list[str] | None
        Optional list of 8-K item numbers when parsed; None otherwise.
    items_descriptions_8k : list[str] | None
        Optional list of 8-K item descriptions aligned with `items_8k`; None
        when items are absent.

    Notes
    -----
    - Completeness for evidence-building requires all header fields to be
      non-None and `company_name` non-empty; 8-K item fields are optional.
    """

    accession_num: str | None
    cik: str | None
    form_type: str | None
    filed_at: pd.Timestamp | None
    company_name: str | None
    items_8k: List[str] | None
    items_descriptions_8k: List[str] | None


class FilledLinkData(TypedDict):
    """
    Purpose
    -------
    Type-affirmed, complete filing metadata ready for evidence construction.

    Key behaviors
    -------------
    - Guarantees presence of accession_num, cik, form_type, filed_at,
      company_name.
    - Optionally carries 8-K item metadata for eligible 8-K filings.
    - Returned by `create_filled_link_data(...)` and by successful extractors.

    Fields
    ------
    accession_num : str
        18-digit accession without hyphens.
    cik : str
        Zero-padded 10-character CIK.
    form_type : str
        Filing form (must be allowed by ELIGIBLE_FORM_TYPES upstream).
    filed_at : pandas.Timestamp
        Filing timestamp (UTC).
    company_name : str
        Normalized company name.
    items_8k : list[str] | None
        Optional list of 8-K item numbers; None for non-8-K forms or when item
        parsing is not performed.
    items_descriptions_8k : list[str] | None
        Optional list of 8-K item descriptions aligned with `items_8k`; None
        when items are absent.

    Notes
    -----
    - Intended to be immutable by convention after creation.
    """

    accession_num: str
    cik: str
    form_type: str
    filed_at: pd.Timestamp
    company_name: str
    items_8k: List[str] | None
    items_descriptions_8k: List[str] | None


def handle_alternate_link(
    link_url: str,
    raw_record: RawRecord,
    run_data: RunData,
    session: requests.Session,
) -> FilledLinkData | None:
    """
    Resolve an entry’s `rel="alternate"` link and extract filing metadata from
    the index page.

    Parameters
    ----------
    link_url : str
        Filing index page URL from the Atom entry.
    raw_record : RawRecord
        Mutable record to enrich with filing-page and HTTP provenance.
    run_data : RunData
        Context (ticker/ticker_or_candidate, logger, namespaces, window,
        oldest_filing_date).
    session : requests.Session
        Shared HTTP session.

    Returns
    -------
    FilledLinkData | None
        Complete parsed metadata when the index page yields all required
        fields; otherwise None.

    Raises
    ------
    None

    Notes
    -----
    - On success, sets `raw_record["filing_page"]` with
      `{page_url, anchor_used="index-page-extraction"}`.
    - Respects request pacing via `make_request(..., sleep_time=REQUEST_DELAY_SECONDS)`.
    """

    logger: InfraLogger = run_data["logger"]
    with make_request(
        link_url, expect_json=False, sleep_time=REQUEST_DELAY_SECONDS, session=session
    ) as response:
        possible_data: FilledLinkData | None = try_extract_evidence_from_index(response, run_data)
        if possible_data is not None:
            raw_record["filing_page"] = {
                "page_url": response.url,
                "anchor_used": "index-page-extraction",
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
        return None


def try_extract_evidence_from_index(
    response: requests.Response, run_data: RunData
) -> FilledLinkData | None:
    """
    Parse filing metadata directly from an index page HTML response.

    Parameters
    ----------
    response : requests.Response
        HTTP response containing the filing index HTML.
    run_data : RunData
        Context used to update `oldest_filing_date` if an earlier `filed_at`
        is seen.

    Returns
    -------
    FilledLinkData | None
        Complete metadata when all required header fields are present;
        otherwise None.

    Raises
    ------
    None

    Notes
    -----
    - Uses project XPaths (`XPATH_FORM_TYPE`, `XPATH_FILED_AT`,
      `XPATH_ACCESSION_NUM`, `XPATH_CIK`, `XPATH_COMPANY_NAME`,
      `XPATH_8K_ITEMS`).
    - Normalizes: accession hyphens removed, CIK zero-padded, dates parsed as
      UTC.
    - Updates `run_data["oldest_filing_date"]` if `filed_at` is earlier than
      the current value.
    - For 8-K/8-K/A forms, calls `parse_8k_items(...)` to attach optional item
      metadata.
    """

    try:
        html_root: Any = etree.HTML(response.content)
        form_type: str | None = html_root.xpath(XPATH_FORM_TYPE)[0].split(" ")[1]
        if not form_type:
            return None
        filed_at: str | None = html_root.xpath(XPATH_FILED_AT)[0]
        if not filed_at:
            return None
        filed_at_date = pd.to_datetime(filed_at, utc=True)
        if run_data["oldest_filing_date"] > filed_at_date:
            run_data["oldest_filing_date"] = filed_at_date
        accession_num: str | None = html_root.xpath(XPATH_ACCESSION_NUM)
        if not accession_num:
            return None
        cik: str | None = html_root.xpath(XPATH_CIK)[0].split(" ")[0]
        if not cik:
            return None
        company_name: str | None = (
            html_root.xpath(XPATH_COMPANY_NAME)[0].replace("(Filer)\n", "").strip()
        )
        if not company_name:
            return None
        item_numbers: List[str] | None = None
        item_descriptions: List[str] | None = None
        if form_type in {"8-K", "8-K/A"}:
            item_numbers, item_descriptions = parse_8k_items(html_root)
            if not item_numbers:
                return None
        return create_filled_link_data(
            {
                "accession_num": accession_num.replace("-", ""),
                "cik": cik.zfill(10),
                "form_type": form_type,
                "filed_at": filed_at_date,
                "company_name": company_name,
                "items_8k": item_numbers,
                "items_descriptions_8k": item_descriptions,
            }
        )
    except (IndexError, ValueError):
        return None


def parse_8k_items(html_root: Any) -> tuple[List[str], List[str]]:
    """
    Extract and filter 8-K item numbers and descriptions from the index page.

    Parameters
    ----------
    html_root : Any
        Parsed lxml HTML root node for the filing index page.

    Returns
    -------
    tuple[list[str], list[str]]
        A pair `(item_numbers, item_descriptions)` where:
        - `item_numbers` holds 8-K item codes (e.g., "1.01", "2.01") that are
          present in `ELIGIBLE_8K_ITEMS`, and
        - `item_descriptions` holds the corresponding descriptions.
        Both lists have the same length and may be empty when no eligible
        items are found.

    Raises
    ------
    None

    Notes
    -----
    - Text is collected using `XPATH_8K_ITEMS`, which targets the “Items”
      block on the index page.
    - Lines are expected to start with `"Item <number>:"`; non-matching lines
      are ignored.
    """

    raw_items: List[str] = html_root.xpath(XPATH_8K_ITEMS)
    item_numbers: List[str] = []
    item_descriptions: List[str] = []

    for item in raw_items:
        item_parts: List[str] = item.split()
        if len(item_parts) < 3 or item_parts[0] != "Item":
            continue
        item_number: str = item_parts[1].rstrip(":.")
        if item_number not in ELIGIBLE_8K_ITEMS:
            continue
        item_description: str = " ".join(item_parts[2:]).strip(" :")
        item_numbers.append(item_number)
        item_descriptions.append(item_description)

    return item_numbers, item_descriptions


def create_filled_link_data(link_data: LinkData) -> FilledLinkData:
    """
    Convert a complete `LinkData` into a type-affirmed `FilledLinkData`.

    Parameters
    ----------
    link_data : LinkData
        A structure containing all required header fields, plus optional 8-K
        item metadata.

    Returns
    -------
    FilledLinkData
        The same data with non-optional types for the header fields.

    Raises
    ------
    AssertionError
        If any required header field is None (defensive check).

    Notes
    -----
    - Does not mutate the incoming data; serves as a type-level assertion.
    - 8-K item metadata (`items_8k`, `items_descriptions_8k`) is passed
      through as-is and may be None.
    """

    assert link_data["accession_num"] is not None
    assert link_data["cik"] is not None
    assert link_data["form_type"] is not None
    assert link_data["filed_at"] is not None
    assert link_data["company_name"] is not None
    return {
        "accession_num": link_data["accession_num"],
        "cik": link_data["cik"],
        "form_type": link_data["form_type"],
        "filed_at": link_data["filed_at"],
        "company_name": link_data["company_name"],
        "items_8k": link_data["items_8k"],
        "items_descriptions_8k": link_data["items_descriptions_8k"],
    }
