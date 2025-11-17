"""
Purpose
-------
Define shared data structures and DB-backed helpers used by the CC-NEWS
parsing pipeline to describe firms, run context, parsed articles, and
per-sample scan statistics.

Key behaviors
-------------
- Provide lightweight `@dataclass` containers for run configuration,
  article payloads, and aggregated metadata.
- Expose helpers to:
  - fetch a year-long trading calendar slice grouped by month, and
  - derive the active S&P 500 firm universe for a given trading day.
- Offer a simple word canonicalization helper used in name-matching
  logic elsewhere in the pipeline.

Conventions
-----------
- Trading days are sourced from the `trading_calendar` table and are
  assumed to represent NYSE business dates in UTC.
- Firm membership is derived by joining `snp_membership` with
  `security_profile_history` under half-open validity windows.
- `word_canonicalizer` strips non-alphanumeric characters and uppercases
  tokens so name matching can be performed in a case- and punctuation-
  insensitive manner.

Downstream usage
----------------
Import these dataclasses and helpers from
`aws.ccnews_parser.news_parser_utils` to construct `RunData` in
orchestrators, to type article records and sample metadata in parsers,
and to seed firm universes and trading calendars for CC-NEWS ingestion
jobs and tests.
"""

import datetime as dt
from dataclasses import dataclass
from typing import Any, List

from infra.logging.infra_logger import InfraLogger
from infra.utils.db_utils import connect_to_db


@dataclass
class FirmInfo:
    """
    Purpose
    -------
    Represent a single firm's identity as needed for CC-NEWS name matching.

    Key behaviors
    -------------
    - Stores the firm's SEC CIK identifier.
    - Holds a human-readable company name used for token-level matching
      against news article text.

    Parameters
    ----------
    cik : str
        The SEC Central Index Key uniquely identifying the firm.
    firm_name : str
        Canonical company name as recorded in `security_profile_history`
        (e.g., "ACME HOLDINGS INC").

    Attributes
    ----------
    cik : str
        Stable firm identifier used as the key in firm dictionaries and
        downstream mappings.
    firm_name : str
        Name string that will be tokenized and canonicalized for matching
        against article text.

    Notes
    -----
    - This dataclass is intentionally small and immutable in spirit; it is
      created from DB query results and not mutated thereafter.
    """

    cik: str
    firm_name: str


@dataclass
class RunData:
    """
    Purpose
    -------
    Capture the execution context for parsing CC-NEWS WARC samples for a
    single (trading_date × session × bucket) slice.

    Key behaviors
    -------------
    - Bundles together the trading date, session label, S3 bucket, and
      active firm universe for use by session parsing code.
    - Provides access to a structured logger and an S3 client for IO-heavy
      operations.

    Parameters
    ----------
    date : datetime.date
        Trading date whose WARC samples are being parsed; also used as the
        `ny_date` for downstream articles.
    session : str
        Session label (typically "intraday" or "overnight") used in logging,
        path layout, and downstream partitioning.
    bucket : str
        Name of the S3 bucket containing input WARC manifests and receiving
        output artifacts.
    firm_info_dict : dict[str, FirmInfo]
        Mapping from CIK to `FirmInfo` describing the firm universe active
        on `date`.
    firm_name_parts : dict[str, set[str]]
        Pre-computed mapping from CIK to the set of canonicalized name
        parts used for efficient firm-name matching.
    samples : List[str]
        List of S3 URIs pointing to WARC sample files to be processed for
        this (date, session).
    logger : InfraLogger
        Structured logger used for debug, info, and warning messages during
        parsing.
    s3_client : Any
        Pre-configured `boto3` S3 client used to fetch WARC samples and
        manifests.

    Attributes
    ----------
    date : datetime.date
        Trading date associated with this run.
    session : str
        Session label used consistently across logs and output paths.
    bucket : str
        S3 bucket name used for both reads and writes.
    firm_info_dict : dict[str, FirmInfo]
        Active firm universe keyed by CIK for this trading date.
    firm_name_parts : dict[str, set[str]]
        Pre-computed canonicalized name parts for each firm in the universe.
    samples : List[str]
        WARC sample URIs that the session parser will iterate over.
    logger : InfraLogger
        Logging handle scoped to this run.
    s3_client : Any
        S3 client instance shared across WARC fetch operations.

    Notes
    -----
    - Instances are constructed by an orchestrator function that
      has already resolved the trading calendar and firm universe.
    - This dataclass is passed by reference into parsing helpers and is not
      meant to be mutated.
    """

    date: dt.date
    session: str
    bucket: str
    firm_info_dict: dict[str, FirmInfo]
    firm_name_parts: dict[str, set[str]]
    samples: List[str]
    logger: InfraLogger
    s3_client: Any


@dataclass
class ArticleData:
    """
    Purpose
    -------
    Represent a single parsed and filtered news article extracted from a
    CC-NEWS WARC record.

    Key behaviors
    -------------
    - Stores canonical identifiers for the source record (WARC path, WARC
      headers, and payload digest).
    - Captures trading-date alignment, session label, firm matches, and
      basic text statistics for downstream modeling.

    Parameters
    ----------
    warc_path : str
        S3 URI or path of the WARC sample file from which the article was
        extracted.
    warc_date_utc : str
        WARC-level timestamp (ISO-like string) indicating when the response
        was captured, in UTC.
    url : str
        Target URL of the HTTP response, taken from `WARC-Target-URI`.
    http_status : int
        HTTP status code of the response (e.g., 200).
    http_content_type : str
        Normalized `Content-Type` header, typically including the MIME type
        and optional charset.
    payload_digest : str
        WARC payload digest (e.g., "sha1:...") used to deduplicate content.
    ny_date : datetime.date
        Trading date in New York time associated with this article, usually
        derived from CC-NEWS session metadata.
    session : str
        Session label (e.g., "intraday" or "overnight") indicating how the
        article was bucketed relative to the trading day.
    cik_list : List[str]
        List of matched CIKs corresponding to firms referenced in the
        article text.
    word_count : int
        Number of whitespace-separated tokens in the canonicalized article
        body.
    language_confidence : float
        Confidence score (0.0–1.0) from the language detection model for
        the detected language from langdetect.
    full_text : str
        Canonicalized article text (typically ASCII uppercased) used as the
        basis for topic modeling and further processing.

    Attributes
    ----------
    warc_path : str
        Path or URI used for traceability back to the original WARC sample.
    warc_date_utc : str
        Raw UTC capture timestamp string from the WARC headers.
    url : str
        Final URL associated with the article content.
    http_status : int
        Response status used for gating and diagnostics.
    http_content_type : str
        Content type string preserved for auditability.
    payload_digest : str
        Digest used to identify and possibly deduplicate payloads.
    ny_date : datetime.date
        Trading day alignment used for return matching and labeling.
    session : str
        Session bucket for intraday vs overnight partitioning.
    cik_list : List[str]
        Zero or more firm identifiers associated with the article.
    word_count : int
        Basic length measure of the article after canonicalization.
    language_confidence : float
        Confidence score (0.0–1.0) from the language detection model for
        the detected language from langdetect.
    full_text : str
        Canonicalized text for downstream NLP.

    Notes
    -----
    - This dataclass is intended to be serialized directly (e.g., via
      `.__dict__`) into Parquet rows.
    - No heavy text processing (tokenization, stop-word removal) is performed
      here; downstream modeling code further transforms `full_text`.
    """

    warc_path: str
    warc_date_utc: str
    url: str
    http_status: int
    http_content_type: str
    payload_digest: str
    ny_date: dt.date
    session: str
    cik_list: List[str]
    word_count: int
    language_confidence: float
    full_text: str


@dataclass
class SampleMetadata:
    """
    Purpose
    -------
    Capture per-WARC-sample scan statistics for diagnostic and monitoring
    purposes.

    Key behaviors
    -------------
    - Track how many records were scanned and how many passed successive
      gating criteria (HTML status, length, language, firm match).
    - Count unhandled and decompression-specific errors so they can be
      monitored separately from normal gating drops.

    Parameters
    ----------
    records_scanned : int
        Total number of WARC records examined in the sample.
    html_200_count : int
        Count of records with HTTP status 200, regardless of content type.
    unhandled_errors : int
        Count of records (or iterator steps) that raised an unexpected
        exception during processing.
    decompression_errors : int
        Count of records that failed during decompression or low-level I/O
        (e.g., `gzip.BadGzipFile`, `EOFError`, `OSError`).
    ge_25_words : int
        Count of records whose canonicalized visible text contained at least
        25 tokens.
    too_long_articles : int
        Count of records whose canonicalized visible text exceeded the
        configured maximum token threshold.
    english_count : int
        Count of records whose detected language was English.
    matched_any_firm : int
        Count of records where at least one firm CIK was matched in the text.
    articles_kept : int
        Count of records that survived all gates and were converted into
        `ArticleData`.

    Attributes
    ----------
    records_scanned : int
        Updated for every record encountered in the WARC file.
    html_200_count : int
        Incremented for each HTTP-200 response before content-type gating.
    unhandled_errors : int
        Incremented when an unexpected exception occurs while iterating or
        processing records.
    decompression_errors : int
        Incremented when decompression or low-level I/O for a record fails.
    ge_25_words : int
        Incremented for each response with sufficient token length
        (≥ 25 tokens).
    too_long_articles : int
        Incremented for each response whose token count exceeds the maximum
        allowed threshold.
    english_count : int
        Incremented when language detection returns `"en"`.
    matched_any_firm : int
        Incremented when the firm matcher yields at least one CIK.
    articles_kept : int
        Incremented when a record results in a retained `ArticleData`.

    Notes
    -----
    - These counters are meant to be accumulated within a single WARC sample
      and then written to a Parquet dataset for monitoring.
    - All fields are integers; they can be safely summed across samples to
      get per-day or per-session totals.
    """

    records_scanned: int
    html_200_count: int
    unhandled_errors: int
    decompression_errors: int
    ge_25_words: int
    too_long_articles: int
    english_count: int
    matched_any_firm: int
    articles_kept: int


@dataclass
class SampleData:
    """
    Purpose
    -------
    Group all retained articles and their associated scan statistics for a
    single WARC sample.

    Key behaviors
    -------------
    - Hold the list of `ArticleData` instances extracted from one WARC file.
    - Associate that list with the `SampleMetadata` describing how many
      records were scanned and gated.

    Parameters
    ----------
    article_data : List[ArticleData]
        All articles retained from the sample after applying filters.
    sample_metadata : SampleMetadata
        Aggregate counters describing how the sample was processed.

    Attributes
    ----------
    article_data : List[ArticleData]
        Per-article payloads suitable for direct serialization to Parquet.
    sample_metadata : SampleMetadata
        Per-sample diagnostics suitable for monitoring and yield analysis.

    Notes
    -----
    - `SampleData` instances are typically collected into lists for a
      `(date, session)` slice and then flattened into article-level and
      metadata-level Parquet datasets by the orchestrator.
    """

    article_data: List[ArticleData]
    sample_metadata: SampleMetadata


def extract_trading_calendar_slice(year: int) -> dict[int, List[dt.date]]:
    """
    Fetch trading days for a given year, grouped by month, from the database.

    Parameters
    ----------
    year : int
        Four-digit calendar year (e.g., 2020) for which trading days should
        be retrieved from the `trading_calendar` table.

    Returns
    -------
    dict[int, list[datetime.date]]
        A mapping from month number (1–12) to a list of trading dates in
        that month, sorted in ascending order.

    Raises
    ------
    Exception
        Any database connection or query-related exceptions raised by
        `connect_to_db` or the underlying driver are propagated to the
        caller.

    Notes
    -----
    - The SQL query groups trading days by `EXTRACT(MONTH FROM trading_day)`
      and uses `array_agg` to preserve ordering.
    """

    query: str = """
    SELECT EXTRACT(MONTH FROM trading_day)::int AS month,
        array_agg(trading_day ORDER BY trading_day) AS trading_days
    FROM trading_calendar
    WHERE EXTRACT(YEAR FROM trading_day) = %s
    AND is_trading_day = TRUE
    GROUP BY 1
    ORDER BY 1;
    """
    with connect_to_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (year,))
            results: List[tuple] = cursor.fetchall()
    trading_days: dict[int, List[dt.date]] = {}
    for month, days in results:
        trading_days[month] = days
    return trading_days


def extract_firm_info_per_day(current_date: dt.date) -> dict[str, FirmInfo]:
    """
    Retrieve the active firm universe for a specific trading date.

    Parameters
    ----------
    current_date : datetime.date
        Trading date for which to resolve active S&P 500 constituents and
        their company names.

    Returns
    -------
    dict[str, FirmInfo]
        Dictionary mapping CIK (as a string) to `FirmInfo` instances for all
        firms that are members of `snp_membership` on `current_date` and
        whose security profiles are valid per `security_profile_history`.

    Raises
    ------
    Exception
        Any database connection or query-related exceptions raised by
        `connect_to_db` or the underlying driver are propagated to the
        caller.

    Notes
    -----
    - Membership is determined by joining `snp_membership` and
      `security_profile_history` on CIK with a `<@` validity-window
      condition, ensuring the profile is valid on `current_date`.
    - Results are de-duplicated with `SELECT DISTINCT` and ordered by CIK to
      provide stable iteration order.
    """

    query: str = """
    SELECT DISTINCT
        m.cik,
        h.company_name
    FROM snp_membership AS m
    JOIN security_profile_history AS h
    ON h.cik = m.cik
    AND m.trading_day <@ h.validity_window
    WHERE m.trading_day = %s
    ORDER BY m.cik;
    """
    with connect_to_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (current_date,))
            results: List[tuple] = cursor.fetchall()
    firm_info_dict: dict[str, FirmInfo] = {}
    for cik, firm_name in results:
        firm_info_dict[cik] = FirmInfo(cik=cik, firm_name=firm_name)
    return firm_info_dict


def initialize_sample_metadata() -> SampleMetadata:
    """
    Construct a zero-initialized `SampleMetadata` instance.

    Parameters
    ----------
    None
        All counter fields are initialized to zero; no input arguments are
        required.

    Returns
    -------
    SampleMetadata
        A new metadata object with all counters set to 0, ready to be
        incremented during WARC sample processing.

    Raises
    ------
    None
        This helper is pure and cannot fail under normal conditions.

    Notes
    -----
    - This function centralizes metadata initialization so that any future
      changes to default values need to be made in only one place.
    """

    return SampleMetadata(
        records_scanned=0,
        html_200_count=0,
        unhandled_errors=0,
        decompression_errors=0,
        ge_25_words=0,
        too_long_articles=0,
        english_count=0,
        matched_any_firm=0,
        articles_kept=0,
    )


def word_canonicalizer(word: str) -> str:
    """
    Normalize a token for case- and punctuation-insensitive matching.

    Parameters
    ----------
    word : str
        Input token (possibly containing punctuation or mixed case) to be
        canonicalized.

    Returns
    -------
    str
        A canonical form of `word` where all non-alphanumeric characters
        have been removed and the remaining characters are converted to
        uppercase. Returns the empty string if no alphanumeric characters
        are present.

    Raises
    ------
    None
        The function operates purely on the string and does not perform any
        IO or external calls.

    Notes
    -----
    - This helper is used by firm-name matching logic to ensure that
      variations in punctuation (e.g., "Corp.", "CORP") and case do not
      affect equality checks.
    - The behavior is locale-insensitive and relies solely on Python's
      definition of `str.isalnum()`.
    """

    return "".join(char for char in word if char.isalnum()).upper()
