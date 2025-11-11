"""
Purpose
-------
Define a small, versioned schema for capturing raw provenance from the EDGAR
ingest pipeline (query → Atom entry → filing index page → optional HTTP
fetches), so downstream code can persist an auditable JSONB payload alongside
curated data.

Key behaviors
-------------
- Provides TypedDicts for each stage: RawQuery, AtomEntry, RawFiling, RawHTTP,
  and a top-level RawRecord envelope.
- Allows incremental population: non-core blocks are marked NotRequired to
  allow attachment as the pipeline progresses with correct typing.
- Stabilizes field formats (e.g., "YYYYMMDD" dates, absolute URLs, ISO-8601
  timestamps) to support deterministic hashing and reproducible audits.

Conventions
-----------
- Tickers / candidate identifiers are normalized upstream (UPPER/trimmed for
  tickers; CIKs zero-padded to 10 chars).
- Validity window bounds in RawQuery are serialized as "YYYYMMDD" strings.
- `AtomEntry.updated` is copied verbatim (ISO-8601 text) from the feed.
- URLs are absolute; `filing_page.page_url` is the HTML index page; any
  submission.txt or other HTTP fetches live in RawHTTP.request_url /
  response_url when captured.
- `filing_page.anchor_used` indicates how the index page was interpreted
  (e.g., "index-page-extraction").

Downstream usage
----------------
- Serialize RawRecord to JSONB for the evidence table; never mutate in place
  after persistence.
- Treat absent NotRequired blocks as "not captured" rather than errors.
"""

from typing import NotRequired, TypedDict

from infra.seeds.seed_evidence.edgar_search.edgar_config import PRODUCER, SOURCE
from infra.seeds.seed_evidence.seed_evidence_types import ValidityWindow

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
    kind : str
        Logical query kind identifier (e.g., "edgar_atom").
    candidate : str
        CIK used in the EDGAR query.
    window_start : str
        Inclusive start of the validity window, formatted "YYYYMMDD".
    window_end : str
        Exclusive end of the validity window, formatted "YYYYMMDD".

    Attributes
    ----------
    kind : str
    candidate : str
    window_start : str
    window_end : str

    Notes
    -----
    - `ticker_or_candidate` is recorded exactly as used in the query, even when
      the logical “ticker” for the run is different.
    """

    kind: str
    candidate: str
    window_start: str
    window_end: str


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
    alternate_link : str
        The entry's HTML index URL.
    updated : str
        The feed's `updated` timestamp (ISO-8601 text).
    entry_id : str | None
        Optional feed-provided ID.
    title : str | None
        Optional feed-provided title.

    Attributes
    ----------
    alternate_link : str
    updated : str
    entry_id : str | None
    title : str | None

    Notes
    -----
    - `updated` is Atom metadata, not the filing’s "FILED AS OF DATE".
    """

    alternate_link: str
    updated: str
    entry_id: NotRequired[str | None]
    title: NotRequired[str | None]


class RawFiling(TypedDict):
    """
    Purpose
    -------
    Describe the HTML filing index page associated with this evidence.

    Key behaviors
    -------------
    - Records which index-page parsing strategy was used.
    - Enables downstream quality adjustments based on how the filing was
      interpreted.

    Parameters
    ----------
    page_url : str
        Absolute URL of the filing's HTML index page.
    anchor_used : str
        Label describing the parsing strategy, e.g. "index-page-extraction".

    Attributes
    ----------
    page_url : str
    anchor_used : str

    Notes
    -----
    - Additional strategies (e.g., "complete-submission") can be added over
      time without changing the schema shape.
    """

    page_url: str
    anchor_used: str


class RawRecord(TypedDict):
    """
    Purpose
    -------
    Top-level envelope for raw provenance captured during EDGAR ingest.

    Key behaviors
    -------------
    - Carries a schema version plus source/producer identifiers.
    - Allows optional attachment of query, Atom entry, filing-page, and HTTP
      metadata as the pipeline progresses.

    Parameters
    ----------
    schema_version : str
        Logical version of this raw-record schema.
    source : str
        Upstream/source label (e.g., "edgar_fts").
    producer : str
        Logical producer name (e.g., "fetch_edgar_evidence").
    raw_query : RawQuery, optional
        Recorded EDGAR Atom query parameters.
    atom_entry : AtomEntry, optional
        Metadata for the Atom `<entry>` chosen for this filing.
    filing_page : RawFiling, optional
        Metadata for the HTML filing index page.

    Attributes
    ----------
    schema_version : str
    source : str
    producer : str
    raw_query : RawQuery, optional
    atom_entry : AtomEntry, optional
    filing_page : RawFiling, optional

    Notes
    -----
    - Fields marked optional (`NotRequired[...]`) may be absent when a stage
      has not yet been executed.
    - Once persisted as JSONB, records should be treated as immutable.
    """

    schema_version: str
    source: str
    producer: str
    raw_query: NotRequired[RawQuery]
    atom_entry: NotRequired[AtomEntry]
    filing_page: NotRequired[RawFiling]


def generate_initial_raw_record(candidate: str, validity_window: ValidityWindow) -> RawRecord:
    """
    Initialize a `RawRecord` with schema metadata and a `RawQuery` block.

    Parameters
    ----------
    ticker_or_candidate : str
        Ticker or CIK used in the EDGAR Atom query.
    validity_window : ValidityWindow
        Half-open `(start_utc, end_utc)` window for the search; bounds are
        serialized into "YYYYMMDD" strings.

    Returns
    -------
    RawRecord
        A new raw-record envelope seeded with `schema_version`, `source`,
        `producer`, and `raw_query`, ready to be enriched by later stages.

    Raises
    ------
    None

    Notes
    -----
    - The returned record does not yet include Atom entry or filing-page,
      those are attached downstream.
    """

    raw_query: RawQuery = {
        "kind": "edgar_atom",
        "candidate": candidate,
        "window_start": validity_window[0].strftime("%Y%m%d"),
        "window_end": validity_window[1].strftime("%Y%m%d"),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "producer": PRODUCER,
        "raw_query": raw_query,
    }
