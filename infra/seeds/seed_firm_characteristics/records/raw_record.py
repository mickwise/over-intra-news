"""
Purpose
-------
Define a small, versioned schema for capturing raw provenance from the EDGAR
ingest pipeline (query → atom entry → filing page → submission.txt HTTP fetch),
so downstream code can persist an auditable JSONB payload alongside curated data.

Key behaviors
-------------
- Provides TypedDicts for each stage: RawQuery, AtomEntry, RawFiling, RawHTTP,
  and a top-level RawRecord envelope.
- Allows incremental population: non-core blocks are marked NotRequired
  to allow attachment as the pipeline progresses with correct typing.
- Stabilizes field formats (e.g., "YYYYMMDD" dates, absolute URLs, ISO-8601
  timestamps) to support deterministic hashing and reproducible audits.

Conventions
-----------
- Tickers are normalized upstream (UPPER/trimmed); CIKs are zero-padded 10 chars.
- Validity window bounds in RawQuery are serialized as "YYYYMMDD" strings.
- `AtomEntry.updated` is copied verbatim (ISO-8601 text) from the feed.
- URLs are absolute; `filing_page.page_url` is the HTML index page; the .txt
  URL lives in RawHTTP.request_url/response_url.
- `filing_page.anchor_used` ∈ {"complete-submission", "any-txt-file"}.

Downstream usage
----------------
- Serialize RawRecord to JSONB for the evidence table; never mutate in place
  after persistence.
"""

from typing import NotRequired, TypedDict

SCHEMA_VERSION: str = "1.0"


class RawQuery(TypedDict):
    """
    Purpose
    -------
    Capture the exact query parameters issued to the upstream EDGAR Atom feed.

    Key behaviors
    -------------
    - Records the intended search inputs as strings for stable audit/replay.
    - Uses compact "YYYYMMDD" formatting for date bounds.

    Parameters
    ----------
    (kind, ticker, window_start, window_end) : str
        See Attributes for meanings and formatting.

    Attributes
    ----------
    kind : str
        Query kind identifier, e.g., "edgar_atom".
    ticker : str
        Uppercased ticker used in the query.
    window_start : str
        Inclusive start of the validity window, "YYYYMMDD".
    window_end : str
        Exclusive end of the validity window, "YYYYMMDD".

    Notes
    -----
    None
    """

    kind: str
    ticker: str
    window_start: str
    window_end: str


class RawHTTP(TypedDict):
    """
    Purpose
    -------
    Snapshot the HTTP transaction used to fetch the submission text (.txt) file.

    Key behaviors
    -------------
    - Preserves the requested URL, final URL after redirects, status code, and headers.
    - Treats headers as opaque audit data (no normalization beyond str→str).

    Parameters
    ----------
    (request_url, response_url) : str
        Absolute URLs for requested and final locations.
    status_code : int
        HTTP status returned by the server.
    resp_headers : dict[str, str]
        Raw response headers at fetch time.

    Attributes
    ----------
    request_url : str
    response_url : str
    status_code : int
    resp_headers : dict[str, str]

    Notes
    -----
    - `response_url` may differ due to CDN/website redirects.
    - Header case/order is not guaranteed; downstream should not rely on it.
    """

    request_url: str
    response_url: str
    status_code: int
    resp_headers: dict[str, str]


class AtomEntry(TypedDict):
    """
    Purpose
    -------
    Record metadata from the chosen Atom feed entry that led to this filing.

    Key behaviors
    -------------
    - Mirrors the feed’s own identifiers/labels for provenance.
    - Allows missing `entry_id` or `title` when the feed omits them.

    Parameters
    ----------
    (alternate_link, updated) : str
        The entry's HTML index URL and the feed's updated timestamp (ISO-8601 text).
    (entry_id, title) : str | None
        Optional feed-provided ID and title.

    Attributes
    ----------
    alternate_link : str
        The "alternate" link (HTML index page) for the entry.
    updated : str
        Atom `updated` timestamp (ISO-8601 string as emitted by the feed).
    entry_id : str | None
        Entry URN if present; None otherwise.
    title : str | None
        Entry title if present; None otherwise.

    Notes
    -----
    - `updated` is Atom metadata, not the filing’s "FILED AS OF DATE".
    """

    alternate_link: str
    updated: str
    entry_id: str | None
    title: str | None


class RawFiling(TypedDict):
    """
    Purpose
    -------
    Describe the HTML filing index page used to resolve the submission .txt link.

    Key behaviors
    -------------
    - Records which anchor strategy produced the .txt (canonical vs fallback).
    - Enables downstream quality adjustments based on anchor choice.

    Parameters
    ----------
    page_url : str
        Absolute URL of the filing's HTML index page.
    anchor_used : str
        One of {"index-page-extraction", "complete-submission", "any-txt-file"}
        describing the link strategy.
    used_fallback : bool
        True when the canonical "Complete submission text file" was unavailable.

    Attributes
    ----------
    page_url : str
    anchor_used : str
    used_fallback : bool

    Notes
    -----
    - The .txt URL itself is captured in RawHTTP (request/response URL), not here.
    - A fallback implies slightly higher extraction risk (used by scoring).
    """

    page_url: str
    anchor_used: str
    used_fallback: bool


class RawRecord(TypedDict):
    """
    Purpose
    -------
    Provide a top-level envelope for all raw provenance related to a single filing
    evidence hit, suitable for JSONB persistence and audit.

    Key behaviors
    -------------
    - Always includes schema identity (`schema_version`, `source_chain`, `producer`).
    - Attaches subsequent blocks as stages complete: raw_query, atom_entry, filing_page, raw_http.
    - Designed for append-only audit; values are copied, not transformed.

    Parameters
    ----------
    schema_version : str
        Version string of this schema (e.g., "1.0").
    source_chain : str
        Human-readable pipeline tag, e.g., "edgar:atom→filing-index→submission.txt".
    producer : str
        Component/function name that produced this record.
    raw_query : RawQuery
    raw_http : RawHTTP
    atom_entry : AtomEntry
    filing_page : RawFiling

    Attributes
    ----------
    schema_version : str
    source_chain : str
    producer : str
    raw_query : NotRequired[RawQuery]
    raw_http : NotRequired[RawHTTP]
    atom_entry : NotRequired[AtomEntry]
    filing_page : NotRequired[RawFiling]

    Notes
    -----
    - `NotRequired` sections are attached only once that stage has executed.
    - Accession number and form type are passed separately to identity/scoring logic
      and are intentionally not duplicated here.
    """

    schema_version: str
    source_chain: str
    producer: str
    raw_query: NotRequired[RawQuery]
    raw_http: NotRequired[RawHTTP]
    atom_entry: NotRequired[AtomEntry]
    filing_page: NotRequired[RawFiling]
