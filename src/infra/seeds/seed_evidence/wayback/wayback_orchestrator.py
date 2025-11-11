"""
Purpose
-------
Query the Internet Archive CDX API for Wikipedia's S&P 500 components page
over a given time window, resolve archived snapshots, and extract candidate
CIK numbers per (ticker, validity_window) batch.

Key behaviors
-------------
- Build a single CDX query window that covers the full batch of
  (ticker, validity_window) pairs.
- Fetch and sort Wayback archive URLs chronologically within that window.
- For each archived snapshot, locate the S&P 500 components table,
  resolve the ticker and CIK columns, and attempt to match each batch
  ticker whose validity_window contains the snapshot date.
- Track first_seen / last_seen timestamps and URLs per
  (ticker, validity_window, candidate_cik) and return a deduplicated
  list of WayBackCandidate instances.

Conventions
-----------
- All timestamps are treated as UTC and represented as pandas.Timestamp.
- Validity windows use half-open [start, end) semantics.
- Batch items are (ticker, validity_window) pairs, where ticker is
  already normalized upstream.
- Each WayBackCandidate is unique per (ticker, validity_window, CIK)
  within a single batch call.

Downstream usage
----------------
This module is typically invoked from the seed_evidence orchestrator via
seed_wayback_table(...). Call batch_extract_candidates_wayback(...) with
a Batch of (ticker, validity_window) pairs, a shared requests.Session, and
a structured InfraLogger. Persist the resulting WayBackCandidate list
into the wayback_candidates table and use it as seed input for EDGAR
evidence collection.
"""

import datetime as dt
from typing import Any, List

import pandas as pd
import requests
from lxml import etree

from infra.logging.infra_logger import InfraLogger
from infra.seeds.seed_evidence.edgar_search.edgar_config import REQUEST_DELAY_SECONDS
from infra.seeds.seed_evidence.seed_evidence_types import ValidityWindow
from infra.seeds.seed_evidence.wayback.wayback_config import SNP_LISTS_WAYBACK_URL, WayBackCandidate
from infra.seeds.seed_evidence.wayback.wayback_snapshot_scrape import scrape_snapshot
from infra.seeds.seed_evidence.wayback.wayback_typing import Batch, SeenCandidates
from infra.utils.requests_utils import make_request


def batch_extract_candidates_wayback(
    batch: Batch,
    logger: InfraLogger,
    session: requests.Session,
) -> List[WayBackCandidate]:
    """
    Extract Wayback-sourced CIK candidates for a batch of (ticker, validity_window).

    Parameters
    ----------
    batch : Batch
        List of (ticker, validity_window) pairs to seed. The collective span
        of these windows defines the CDX query horizon.
    logger : InfraLogger
        Structured logger used for INFO/DEBUG/WARN instrumentation.
    session : requests.Session
        Shared HTTP session used for CDX and archive page requests.

    Returns
    -------
    List[WayBackCandidate]
        A deduplicated list of WayBackCandidate objects, one per
        (ticker, validity_window, candidate_cik) combination where a CIK
        could be resolved from at least one snapshot.

    Raises
    ------
    ValueError
        Propagated from `collect_archive_urls_from_cdx_data(...)` if a CDX
        timestamp string cannot be parsed using the expected YYYYMMDDhhmmss
        format.
    requests.HTTPError
        If any CDX or Wayback snapshot request fails with a non-retryable
        status code.
    BaseException
        Propagated from `make_request(...)` on transport or retry exhaustion
        in either the CDX or snapshot-fetch phase.

    Notes
    -----
    - If `batch` is empty, the function logs a warning and returns an empty list.
    - A single CDX query is issued for the minimal window that covers all
      validity_windows in the batch, amortizing network cost over many tickers.
    - Deduplication and first_seen / last_seen tracking are handled by
      `handle_candidate(...)` in the snapshot scraping layer.
    """

    logger.info(
        "batch_extracting_candidates_wayback",
        context={"batch_size": len(batch)},
    )
    if not batch:
        logger.warning(
            "empty_batch_provided_to_wayback_extraction",
        )
        return []
    seen_candidates: SeenCandidates = {(item[0], item[1]): {} for item in batch}
    candidate_list: List[WayBackCandidate] = []
    maximum_window: ValidityWindow = (
        min(item[1][0] for item in batch),
        max(item[1][1] for item in batch),
    )
    wayback_url: str = build_wayback_url(maximum_window)
    archive_urls: List[tuple[str, pd.Timestamp]] = extract_valid_archive_urls(
        wayback_url, session, logger
    )
    for archive_url, snapshot_date in archive_urls:
        extract_candidate_from_archive_batch(
            archive_url, snapshot_date, candidate_list, seen_candidates, batch, session, logger
        )
    logger.info(
        "batch_extracted_candidates_wayback",
        context={"num_candidates": len(candidate_list)},
    )
    return candidate_list


def build_wayback_url(validity_window: ValidityWindow) -> str:
    """
    Construct a CDX API URL for the S&P 500 components page over a window.

    Parameters
    ----------
    validity_window : ValidityWindow
        Half-open [start, end) UTC window. Both bounds are formatted as
        YYYYMMDDhhmmss for the CDX API.

    Returns
    -------
    str
        A fully composed CDX API URL with filters for status 200, text/html
        MIME type, and digest-based collapse to reduce near-duplicates.

    Raises
    ------
    None

    Notes
    -----
    - The URL is tailored to SNP_LISTS_WAYBACK_URL and assumes that this
      is the canonical Wikipedia page whose snapshots contain the table
      of S&P 500 components.
    """

    return (
        "https://web.archive.org/cdx/search/cdx"
        + f"?url={SNP_LISTS_WAYBACK_URL}"
        + f'&from={validity_window[0].strftime("%Y%m%d%H%M%S")}'
        + f'&to={validity_window[1].strftime("%Y%m%d%H%M%S")}'
        + "&matchType=exact&output=json"
        + "&filter=statuscode:200"
        + "&filter=mimetype:text/html"
        + "&collapse=digest"
        + "&fl=timestamp,original,mimetype,statuscode,digest,length"
    )


def extract_valid_archive_urls(
    wayback_url: str, session: requests.Session, logger: InfraLogger
) -> List[tuple[str, pd.Timestamp]]:
    """
    Query the CDX API and return sorted Wayback archive URLs for the page.

    Parameters
    ----------
    wayback_url : str
        A CDX API URL built by `build_wayback_url`.
    session : requests.Session
        HTTP session used for the CDX request.
    logger : InfraLogger
        Logger used for warnings when no captures or no valid archives exist.

    Returns
    -------
    List[tuple[str, pd.Timestamp]]
        Chronologically sorted list of Wayback archive URLs and their
        capture timestamps. Returns an empty list if no captures are
        found or the payload only contains a header row.

    Raises
    ------
    requests.HTTPError
        If the CDX request fails with a non-retryable status code.
    BaseException
        Propagated from `make_request(...)` on transport or retry exhaustion.
    ValueError
        Propagated from `collect_archive_urls_from_cdx_data(...)` if a CDX
        timestamp string cannot be parsed.

    Notes
    -----
    - The function logs a warning if no data rows are returned by the CDX API
      or if, after processing, no valid archive URLs remain.
    - Sorting is done by the capture timestamp decoded from the CDX payload.
    """

    with make_request(wayback_url, sleep_time=REQUEST_DELAY_SECONDS, session=session) as response:
        data: Any = response.json()
        if not data or len(data) < 2:
            logger.warning(
                "no_wayback_archives_found",
                context={"wayback_url": wayback_url},
            )
            return []
        archive_urls: List[tuple[str, pd.Timestamp]] = sorted(
            collect_archive_urls_from_cdx_data(data), key=lambda x: x[1]
        )
    if not archive_urls:
        logger.warning(
            "no_valid_wayback_archives_found",
            context={"wayback_url": wayback_url},
        )
    return [(url, snapshot_date) for url, snapshot_date in archive_urls]


def collect_archive_urls_from_cdx_data(data: Any) -> List[tuple[str, pd.Timestamp]]:
    """
    Transform a parsed CDX JSON payload into archive URLs and timestamps.

    Parameters
    ----------
    data : Any
        Decoded JSON array from the CDX API in the form
        [header_row, row1, row2, ...].

    Returns
    -------
    List[tuple[str, pd.Timestamp]]
        List of (archive_url, capture_timestamp) pairs where capture_timestamp
        is a tz-aware UTC pandas.Timestamp.

    Raises
    ------
    KeyError
        If required CDX fields such as 'timestamp' or 'original' are missing.

    Notes
    -----
    - The header row is used to build a name-to-index map, so the function
      is resilient to column reordering within the CDX response.
    - Timestamps are parsed using the YYYYMMDDhhmmss CDX format and assumed
      to be UTC.
    """

    archive_urls: List[tuple[str, pd.Timestamp]] = []
    header: Any = data[0]
    rows: Any = data[1:]
    header_indices: dict[str, int] = {key: index for index, key in enumerate(header)}
    for row in rows:
        timestamp = row[header_indices["timestamp"]]
        original_url = row[header_indices["original"]]
        archive_url = f"https://web.archive.org/web/{timestamp}id_/{original_url}"
        archive_urls.append(
            (archive_url, pd.Timestamp(dt.datetime.strptime(timestamp, "%Y%m%d%H%M%S"), tz="UTC"))
        )
    return archive_urls


def extract_candidate_from_archive_batch(
    archive_url: str,
    snapshot_date: pd.Timestamp,
    candidate_list: List[WayBackCandidate],
    seen_candidates: SeenCandidates,
    batch: Batch,
    session: requests.Session,
    logger: InfraLogger,
) -> None:
    """
    Fetch and parse a single Wayback snapshot and extract candidates from it.

    Parameters
    ----------
    archive_url : str
        Full Wayback archive URL to an HTML snapshot of the S&P 500 page.
    snapshot_date : pd.Timestamp
        tz-aware UTC timestamp when the snapshot was captured, typically
        derived from the CDX payload.
    candidate_list : List[WayBackCandidate]
        Accumulator list to which new WayBackCandidate instances are appended.
    seen_candidates : SeenCandidates
        Nested dict keyed by (ticker, validity_window)[candidate_cik] used
        for deduplication and first_seen / last_seen tracking.
    batch : Batch
        Original batch of (ticker, validity_window) pairs being processed.
    session : requests.Session
        HTTP session used to fetch the archived HTML page.
    logger : InfraLogger
        Logger for DEBUG instrumentation tying the snapshot to extraction.

    Returns
    -------
    None

    Raises
    ------
    requests.HTTPError
        If the Wayback HTML request fails with a non-retryable status code.
    BaseException
        Propagated from `make_request(...)` on transport or retry exhaustion.

    Notes
    -----
    - All HTML parsing and candidate resolution is delegated to
      `scrape_snapshot(...)` in the snapshot scraping module.
    """

    logger.debug(
        "extracting_candidates_from_wayback_archive",
        context={"archive_url": archive_url, "snapshot_date": snapshot_date},
    )
    with make_request(
        archive_url, expect_json=False, sleep_time=REQUEST_DELAY_SECONDS, session=session
    ) as response:
        html_root = etree.HTML(response.text)
        scrape_snapshot(
            batch, html_root, seen_candidates, candidate_list, archive_url, snapshot_date, logger
        )
    return None
