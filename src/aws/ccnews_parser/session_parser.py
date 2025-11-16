"""
Purpose
-------
Parse CC-NEWS WARC samples for a single (date, session) slice and emit
in-memory article records plus per-sample scan statistics.

Key behaviors
-------------
- Iterate over `RunData.samples`, opening each WARC file from S3 and
  streaming records via `warcio.ArchiveIterator`.
- Gate records by HTTP status, content type, minimum word count,
  language, and firm-name matches derived from `RunData.firm_info_dict`.
- Normalize HTML into uppercase ASCII text and return structured
  `ArticleData` objects alongside `SampleMetadata` counters.

Conventions
-----------
- All text is normalized to uppercase ASCII before tokenization.
- `ny_date` is taken directly from `RunData.date` and represents the
  canonical trading day label for downstream joins.
- A record yields at most one `ArticleData`; most records are filtered
  out early by cheap gates.
- Functions prefer to return `None` on non-fatal gating failures rather
  than raising, so that callers can maintain simple streaming loops.

Downstream usage
----------------
`parse_session` is invoked by the news parser orchestrator for each
(date, session) pair; its `SampleData` outputs are subsequently
serialized (e.g., to Parquet) for topic modeling, signal construction,
and diagnostics on article coverage and filtering behavior.
"""

import gzip
from typing import Any, List

import langdetect
from lxml import html
from warcio import ArchiveIterator

from aws.ccnews_parser.news_parser_config import (
    ARTICLE_BODY_XPATHS,
    ARTICLE_ROOT_XPATHS,
    COMPRESSED_CONTENT_TYPES,
    NAME_SUFFIXES_SET,
    NON_VISIBLE_TAGS,
)
from aws.ccnews_parser.news_parser_utils import (
    ArticleData,
    RunData,
    SampleData,
    SampleMetadata,
    initialize_sample_metadata,
    word_canonicalizer,
)


def parse_session(run_data: RunData) -> List[SampleData]:
    """
    Parse all WARC samples for a given run and collect per-sample results.

    Parameters
    ----------
    run_data : RunData
        Execution context for the session, including trading date, session
        label, S3 bucket and client, firm universe, logger, and the list of
        WARC sample URIs to process.

    Returns
    -------
    list[SampleData]
        A list of `SampleData` objects, one per sample path in
        `run_data.samples`, preserving the original order.

    Raises
    ------
    None
        This function does not raise intentionally; any I/O or parsing
        errors originating from lower-level libraries will propagate.

    Notes
    -----
    - Each sample is processed sequentially via `process_sample(...)`.
    - Debug logs are emitted before and after each sample with the
      associated `SampleMetadata` attached as structured context.
    """

    samples_data: List[SampleData] = []
    for sample in run_data.samples:
        run_data.logger.debug(f"Processing sample: {sample}")
        current_sample: SampleData = process_sample(sample, run_data)
        samples_data.append(current_sample)
        run_data.logger.debug(
            f"Completed sample: {sample}", context=current_sample.sample_metadata.__dict__
        )
    return samples_data


def process_sample(sample: str, run_data: RunData) -> SampleData:
    """
    Scan a single WARC sample and aggregate article records plus metadata.

    Parameters
    ----------
    sample : str
        S3 URI of the WARC.gz file to process (e.g.,
        ``"s3://bucket/year/mm/dd/session/sample.warc.gz"``).
    run_data : RunData
        Run context providing date, session label, logger, S3 client, and
        firm universe used when extracting and filtering records.

    Returns
    -------
    SampleData
        Container holding a list of `ArticleData` objects for all records
        that pass gating, along with `SampleMetadata` counters summarizing
        scan behavior for this sample.

    Raises
    ------
    None
        This function does not raise intentionally; fatal errors from
        S3 access, decompression, or WARC parsing will propagate.

    Notes
    -----
    - All WARC records are counted in `records_scanned`, but only those with
      `rec_type == "response"` are considered for article extraction.
    - Structured debug logs are emitted per record with the evolving
      `SampleMetadata` to aid in diagnosing filter behavior.
    """

    articles: list[ArticleData] = []
    sample_metadata = initialize_sample_metadata()
    with extract_warc_sample(sample, run_data) as warc_file:
        for record in ArchiveIterator(warc_file):
            run_data.logger.debug(
                f"Processing record: {record.rec_headers.get_header('WARC-Record-ID')}"
            )
            sample_metadata.records_scanned += 1
            if record.rec_type != "response":
                continue
            article_data: ArticleData | None = extract_data_from_record(
                sample, record, sample_metadata, run_data
            )
            if article_data:
                articles.append(article_data)
            run_data.logger.debug(
                f"Completed record: {record.rec_headers.get_header('WARC-Record-ID')}",
                context=sample_metadata.__dict__,
            )
    return SampleData(article_data=articles, sample_metadata=sample_metadata)


def extract_warc_sample(sample: str, run_data: RunData) -> gzip.GzipFile:
    """
    Open a WARC.gz sample from S3 and return a gzip file-like wrapper.

    Parameters
    ----------
    sample : str
        S3 URI of the WARC.gz object, prefixed with ``"s3://"`` and
        containing both bucket and key components.
    run_data : RunData
        Run context providing the `s3_client` used to fetch the object;
        other fields are unused by this function.

    Returns
    -------
    gzip.GzipFile
        A `GzipFile` object wrapping the S3 response body, suitable for
        streaming into `warcio.ArchiveIterator`.

    Raises
    ------
    botocore.exceptions.ClientError
        If the underlying `get_object` call fails (e.g., object not found or
        permission denied).
    ValueError
        If the `sample` URI cannot be split into bucket and key components.

    Notes
    -----
    - The caller is responsible for closing the returned `GzipFile`, which
      is typically handled via a context manager (`with` block).
    - The `"s3://"` prefix is stripped before splitting into bucket and key.
    """

    sample = sample.replace("s3://", "")
    bucket, key = sample.split("/", 1)
    response: Any = run_data.s3_client.get_object(Bucket=bucket, Key=key)
    return gzip.GzipFile(fileobj=response["Body"])


def extract_data_from_record(
    sample: str, record: Any, sample_metadata: SampleMetadata, run_data: RunData
) -> ArticleData | None:
    """
    Apply gating logic to a single WARC record and optionally build an article.

    Parameters
    ----------
    sample : str
        S3 URI of the WARC sample from which this record originated; stored
        in the resulting `ArticleData.warc_path` when accepted.
    record : Any
        A WARC record object exposing `rec_type`, `http_headers`,
        `rec_headers`, and `content_stream()` as provided by `warcio`.
    sample_metadata : SampleMetadata
        Mutable counters tracking record-level statistics for the enclosing
        sample; updated in place as gates are evaluated.
    run_data : RunData
        Run context providing the trading date, session label, firm
        universe, and logger used for language and firm-name filtering.

    Returns
    -------
    ArticleData | None
        An `ArticleData` instance when the record passes all filters;
        otherwise `None` to indicate the record was discarded.

    Raises
    ------
    None
        The function does not raise intentionally; decode, HTML, or language
        detection errors are handled by returning `None`, while unexpected
        low-level exceptions from libraries will propagate.

    Notes
    -----
    - Gates are applied in the following order:
        1. HTTP status must be 200 (non-200 responses are dropped).
        2. Content-Type must include ``"text/html"``.
        3. HTML must decode and yield visible ASCII text.
        4. Text must contain at least 25 whitespace-separated words.
        5. Detected language (via `langdetect`) must be exactly ``"en"``.
        6. Firm detection must yield between 1 and 3 CIKs, inclusive.
    - `SampleMetadata` counters (`html_200_count`, `ge_25_words`,
      `english_count`, `matched_any_firm`, `articles_kept`) are updated only
      when the corresponding gates are satisfied.
    """

    http_status: int = int(record.http_headers.get_statuscode())
    if http_status != 200:
        return None
    sample_metadata.html_200_count += 1
    http_content_type: str = record.http_headers.get_header("Content-Type", "").lower()
    if http_content_type.find("text/html") == -1:
        return None
    http_content_encoding: str = record.http_headers.get_header("Content-Encoding", "")
    html_text: str = to_text(record, http_content_type, http_content_encoding)
    visible_text: str | None = convert_to_visible_ascii(html_text)
    if not visible_text:
        return None
    words: list[str] = visible_text.split()
    word_count: int = len(words)
    sample_metadata.ge_25_words += 1
    try:
        detected_language: str = langdetect.detect(visible_text)
    except langdetect.lang_detect_exception.LangDetectException:
        return None
    if detected_language != "en":
        return None
    sample_metadata.english_count += int(detected_language == "en")
    name_set = detect_firms(words, run_data)
    if len(name_set) > 3 or len(name_set) == 0:
        return None
    sample_metadata.matched_any_firm += int(len(name_set) > 0)
    sample_metadata.articles_kept += 1
    return ArticleData(
        warc_path=sample,
        warc_date_utc=record.rec_headers.get_header("WARC-Date"),
        url=record.rec_headers.get_header("WARC-Target-URI"),
        http_status=http_status,
        http_content_type=http_content_type,
        payload_digest=record.rec_headers.get_header("WARC-Payload-Digest"),
        ny_date=run_data.date,
        session=run_data.session,
        cik_list=list(name_set),
        word_count=word_count,
        language=detected_language,
        full_text=visible_text,
    )


def to_text(record: Any, http_content_type: str, http_content_encoding: str) -> str:
    """
    Decode the HTTP response body into a best-effort Unicode HTML string.

    Parameters
    ----------
    record : Any
        WARC record exposing `content_stream()` that returns a raw byte
        stream for the HTTP payload.
    http_content_type : str
        Lowercased Content-Type header value, used to extract an explicit
        charset when present.
    http_content_encoding : str
        Content-Encoding header value, used to determine whether the body
        is compressed (e.g., gzip, deflate).

    Returns
    -------
    str
        A decoded HTML string, using the declared charset when available or
        a UTF-8-plus-fallback strategy otherwise.

    Raises
    ------
    OSError
        If gzip decompression fails for a compressed payload.
    UnicodeDecodeError
        If decoding fails in both the primary and fallback code paths.

    Notes
    -----
    - When `http_content_encoding` is in `COMPRESSED_CONTENT_TYPES`, the
      payload is transparently decompressed before decoding.
    - If no charset is declared, the function first attempts UTF-8 decoding
      with replacement; if the replacement character ``"�"`` appears, it
      retries by re-encoding to UTF-8 bytes and decoding as Latin-1 to
      salvage mis-labeled content.
    """

    raw_stream: Any
    if http_content_encoding.lower() in COMPRESSED_CONTENT_TYPES:
        raw_stream = gzip.GzipFile(fileobj=record.content_stream())
    else:
        raw_stream = record.content_stream()
    if "charset=" in http_content_type:
        charset: str = http_content_type.split("charset=")[-1].split(";")[0].strip()
        return raw_stream.read().decode(charset, errors="replace")
    else:
        decoded_text: str = raw_stream.read().decode("utf-8", errors="replace")
        if "�" in decoded_text:
            decoded_text = decoded_text.encode("utf-8", errors="replace").decode(
                "latin-1", errors="replace"
            )
        return decoded_text


def convert_to_visible_ascii(html_text: str) -> str | None:
    """
    Strip non-visible HTML content, locate an article-like subtree, and normalize
    its text to uppercase ASCII.

    Parameters
    ----------
    html_text : str
        Raw HTML document as a Unicode string, typically the output of
        `to_text(...)`.

    Returns
    -------
    str | None
        Uppercase ASCII text for the best-effort article body with whitespace
        collapsed on success; `None` if the HTML cannot be parsed or no
        sufficiently long article-like region is found.

    Raises
    ------
    None
        Parsing errors from `lxml.html` are caught and converted to `None`.

    Notes
    -----
    - The function first parses the HTML into a tree and scans
      `ARTICLE_ROOT_XPATHS` in order to find candidate article roots
      (e.g., ``<article>``, main content containers, or common CMS IDs/classes).
    - For each XPath that yields elements, the longest element by
      ``len(el.text_content())`` is chosen as the root candidate.
    - Within this root candidate, the function iterates over
      `ARTICLE_BODY_XPATHS` to search for more specific article-body
      containers (e.g., elements with ``itemprop="articleBody"`` or
      body-like classes). For each body XPath, the longest matching
      element is passed to `extract_text_from_element`.
    - If any body element produces non-empty text from
      `extract_text_from_element`, that text is returned immediately.
    - Non-visible tags listed in `NON_VISIBLE_TAGS` are stripped inside
      `extract_text_from_element`; this function is concerned only with
      structural selection of the most article-like subtree.
    """

    try:
        html_tree: Any = html.fromstring(html_text)
        for xpath in ARTICLE_ROOT_XPATHS:
            elements: Any = html_tree.xpath(xpath)
            if elements:
                max_length_element: Any = max(elements, key=lambda el: len(el.text_content() or ""))
                for body_xpath in ARTICLE_BODY_XPATHS:
                    body_elements: Any = max_length_element.xpath(body_xpath)
                    if body_elements:
                        max_length_element = max(
                            body_elements, key=lambda el: len(el.text_content() or "")
                        )
                        extracted_text: str = extract_text_from_element(max_length_element)
                        if extracted_text:
                            return extracted_text
    except (ValueError, TypeError, html.etree.XMLSyntaxError):
        return None
    return None


def extract_text_from_element(element: Any) -> str:
    """
    Extract visible text from an HTML subtree and normalize it to uppercase ASCII.

    Parameters
    ----------
    element : Any
        Root HTML element from which to extract visible text. This is
        a node selected via `ARTICLE_ROOT_XPATHS` and then `ARTICLE_BODY_XPATHS`.

    Returns
    -------
    str
        Uppercase ASCII text with HTML tags removed and whitespace collapsed when
        the element contains at least 25 whitespace-separated tokens; an empty
        string otherwise, indicating that the subtree is too short to be treated
        as an article body.

    Raises
    ------
    None
        Parsing operations on the already-constructed element are not expected to
        raise; failures are represented by returning an empty string.

    Notes
    -----
    - All descendants whose tags are listed in `NON_VISIBLE_TAGS` (e.g.,
      ``<script>``, ``<style>``, ``<head>``) are removed before text extraction.
    - Text is obtained via ``element.text_content()``, whitespace is collapsed
      to single spaces, and non-ASCII characters are dropped via ASCII encoding
      with ``errors="ignore"``.
    - The minimum-length check (25 tokens) is applied here so that callers like
      `convert_to_visible_ascii` can treat short matches as failures and continue
      searching other XPath candidates.
    """

    for tag in NON_VISIBLE_TAGS:
        for elem in element.findall(f".//{tag}"):
            elem.drop_tree()
    raw_text: str = element.text_content()
    normalized_text: str = " ".join(raw_text.split())
    ascii_text: str = normalized_text.encode("ascii", errors="ignore").decode("ascii")
    if len(normalized_text.split()) < 25:
        return ""
    return ascii_text.upper()


def detect_firms(words: list[str], run_data: RunData) -> set[str]:
    """
    Identify firms mentioned in tokenized article text via name-part matches.

    Parameters
    ----------
    words : list[str]
        Tokenized article text, typically derived from visible ASCII by
        splitting on whitespace before canonicalization.
    run_data : RunData
        Run context providing `firm_info_dict`, a mapping from CIK to
        `FirmInfo` containing the canonical firm names for the day.

    Returns
    -------
    set[str]
        Set of CIK strings for firms whose canonicalized name parts all
        appear in the article word set and whose non-suffix name tokens
        each occur at least twice.

    Raises
    ------
    None
        The function does not raise intentionally; it operates purely on
        in-memory data structures.

    Notes
    -----
    - Words and firm-name parts are canonicalized by `word_canonicalizer`
      (alphanumeric-only, uppercased) and empty tokens are discarded.
    - Firm-name tokens (including common suffixes such as ``"INC"``,
      ``"CORP"``, ``"LLC"``) must all be present in the canonicalized
      article word set for a firm to be considered a candidate match.
    - Common corporate suffixes are listed in `NAME_SUFFIXES_SET`. After
      the initial presence check, these suffix tokens are removed and the
      remaining name parts must each appear at least twice in the article
      for the match to be accepted.
    - This frequency guard helps avoid spurious matches on very short
      mentions or noisy contexts, trading some recall for higher precision.
    """

    word_frequency_dict: dict[str, int] = {}
    for word in words:
        canonical_word: str = word_canonicalizer(word)
        if canonical_word:
            word_frequency_dict[canonical_word] = word_frequency_dict.get(canonical_word, 0) + 1
    name_dict: dict[str, set[str]] = {}
    for firm_info in run_data.firm_info_dict.values():
        parts = {word_canonicalizer(part) for part in firm_info.firm_name.split()}
        if parts:
            name_dict[firm_info.cik] = parts
    matched_firms_by_name: set[str] = set()
    for cik, name_parts in name_dict.items():
        if name_parts.issubset(word_frequency_dict.keys()):
            name_parts_no_suffixes: set[str] = name_parts - NAME_SUFFIXES_SET
            appearance_count_no_suffixes: int = min(
                word_frequency_dict[part] for part in name_parts_no_suffixes
            )
            if appearance_count_no_suffixes > 1:
                matched_firms_by_name.add(cik)
    return matched_firms_by_name
