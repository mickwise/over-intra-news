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
- Record decompression and record-processing failures in
  `SampleMetadata` (`decompression_errors`, `unhandled_errors`) and
  log them via the provided `InfraLogger` without aborting the entire
  session.

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
    ENCODING_ALIASES,
    LANGUAGE_ACCEPTANCE_PROBABILITY,
    MAXIMUM_ALLOWED_TOKENS,
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
        This function does not raise intentionally; fatal errors from S3
        access, gzip decompression, or `ArchiveIterator` construction may
        still propagate, but errors during record iteration or per-record
        processing are caught and recorded in `SampleMetadata`.

    Notes
    -----
    - All WARC records are counted in `records_scanned`, but only those with
      `rec_type == "response"` are considered for article extraction.
    - Exceptions raised while iterating records or extracting data from a
      single record increment `unhandled_errors`, log an `"warc_iter_error"`
      or `"record_processing_error"` event, and cause that record (or the
      remainder of the sample, in the iterator case) to be skipped without
      killing the surrounding session or month.
    - Structured debug logs are emitted per sample (pre- and post-scan), with
      `SampleMetadata` attached as structured context to aid in diagnosing
      filter behavior.
    """

    articles: List[ArticleData] = []
    sample_metadata = initialize_sample_metadata()

    with extract_warc_sample(sample, run_data) as warc_file:
        iterator = ArchiveIterator(warc_file)
        while True:
            try:
                record = next(iterator)
            except StopIteration:
                break
            except Exception as exc:  # pylint: disable=W0718
                sample_metadata.unhandled_errors += 1
                run_data.logger.error(
                    "warc_iter_error",
                    context={
                        "sample": sample,
                        "exception": str(exc),
                    },
                )
                # Bail out of this *sample*, but don't kill the session/month
                break

            try:
                sample_metadata.records_scanned += 1
                if record.rec_type != "response":
                    continue

                article_data: ArticleData | None = extract_data_from_record(
                    sample, record, sample_metadata, run_data
                )
                if article_data:
                    articles.append(article_data)
            except Exception as exc:  # pylint: disable=W0718
                sample_metadata.unhandled_errors += 1
                run_data.logger.error(
                    "record_processing_error",
                    context={
                        "sample": sample,
                        "record_id": record.rec_headers.get_header("WARC-Record-ID"),
                        "exception": str(exc),
                    },
                )
                continue

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
    Apply the full gating pipeline to a single WARC record and construct an
    `ArticleData` object only when all filters pass.

    Parameters
    ----------
    sample : str
        S3 URI of the WARC sample from which this record originated; embedded
        into the returned `ArticleData` for traceability.
    record : Any
        WARC record object exposing `rec_type`, `http_headers`, `rec_headers`,
        and `content_stream()`.
    sample_metadata : SampleMetadata
        Mutable counters updated as each gating step succeeds.
    run_data : RunData
        Execution context providing trading date, session, firm universe,
        structured logger, and S3 client.

    Returns
    -------
    ArticleData | None
        Populated article object when all gates are satisfied; otherwise `None`.

    Notes
    -----
    - Gates are applied strictly in this order:
        1. HTTP status must be 200.
        2. Content-Type must include ``"text/html"``.
        3. HTML is decoded to Unicode and canonicalized to visible ASCII.
        4. Canonicalized text must contain at least 25 tokens.
        5. Token count must not exceed `MAXIMUM_ALLOWED_TOKENS`.
        6. Language detection via `langdetect.detect_langs` must yield:
            - top language `"en"`, and
            - probability ≥ `LANGUAGE_ACCEPTANCE_PROBABILITY`.
        7. Firm matching must yield between 1 and 3 CIKs inclusive.
    - Gating-related counters in `sample_metadata` are incremented as records
      pass each threshold (status, length, language, firm match); a single
      record may contribute to multiple counters.
    - Any decoding, parsing, or language-detection failure is handled by
      returning `None`; unexpected low-level exceptions propagate outward and
      are recorded by the caller.
    - Only fully accepted records increment `articles_kept`.
    """

    http_status: int = int(record.http_headers.get_statuscode())
    if http_status != 200:
        return None
    sample_metadata.html_200_count += 1
    http_content_type: str = record.http_headers.get_header("Content-Type", "").lower()
    if http_content_type.find("text/html") == -1:
        return None
    http_content_encoding: str = record.http_headers.get_header("Content-Encoding", "")
    try:
        html_text: str = to_text(record, http_content_type, http_content_encoding)
        visible_text: str | None = convert_to_visible_ascii(html_text)
        if not visible_text:
            return None
        words: List[str] = visible_text.split()
        word_count: int = len(words)
        if word_count < 25:
            return None
        sample_metadata.ge_25_words += 1
        if word_count > MAXIMUM_ALLOWED_TOKENS:
            sample_metadata.too_long_articles += 1
            return None
        try:
            probabilities: List[Any] = langdetect.detect_langs(visible_text)
        except langdetect.lang_detect_exception.LangDetectException:
            return None
        top_language: Any = probabilities[0]
        if top_language.lang != "en" or top_language.prob < LANGUAGE_ACCEPTANCE_PROBABILITY:
            return None
        sample_metadata.english_count += 1
        name_set = detect_firms(words, run_data)
        if len(name_set) > 3:
            sample_metadata.matched_any_firm += 1
            return None
        if len(name_set) == 0:
            return None
        sample_metadata.matched_any_firm += 1
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
            language_confidence=top_language.prob,
            full_text=visible_text,
        )
    except (EOFError, OSError, gzip.BadGzipFile) as exc:
        sample_metadata.decompression_errors += 1
        run_data.logger.error(
            "Decompression error processing record",
            context={
                "sample": sample,
                "record_id": record.rec_headers.get_header("WARC-Record-ID"),
                "exception": str(exc),
            },
        )
        return None


def to_text(record: Any, http_content_type: str, http_content_encoding: str) -> str:
    """
    Purpose
    -------
    Decode the HTTP response body from a WARC record into a best-effort Unicode
    HTML string, handling compressed payloads and common mis-labelled charsets
    without aborting the record.

    Key behaviors
    -------------
    - Transparently decompress the payload when `Content-Encoding` indicates a
      compressed body (e.g. `gzip`, `deflate`), otherwise stream the raw bytes.
    - When a `charset` parameter is present in `Content-Type`, normalize it via
      `ENCODING_ALIASES` and attempt a single decode using that codec with
      `errors="replace"`.
    - When no `charset` is declared, decode as UTF-8 with replacement and, if
      the decoded string contains the replacement character `�`, perform a
      Latin-1 salvage pass to recover additional text.
    - Callers treat malformed characters as noise that will be filtered by
      downstream gates.

    Conventions
    -----------
    - The response body is read exactly once into memory as a `bytes` object so
      that multiple decode attempts do not re-hit S3 or exhaust the stream.
    - `ENCODING_ALIASES` maps common mis-labelled charsets (such as `"cp-1251"`
      or `"en_us.utf-8"`) onto valid Python codec names (e.g. `"cp1251"`,
      `"utf-8"`), preventing avoidable `LookupError`s in practice.
    - All decode operations use `errors="replace"` so that decoding never raises
      `UnicodeDecodeError`; truly unknown or invalid charsets in the presence of
      a `charset` parameter will surface as codec lookup errors rather than
      decoding errors.
    - Charset issues that do not raise (i.e. when a codec is found) are handled
      internally; callers receive a best-effort HTML string and apply
      higher-level gates (length, language, firm matches) afterwards.

    Downstream usage
    ----------------
    This helper is called from `extract_data_from_record(...)` before HTML
    parsing. It guarantees that either a Unicode HTML string is returned or a
    low-level I/O/codec error is raised; such errors are handled by the caller
    as `record_processing_error` or `decompression_errors` without killing the
    entire session.
    """

    raw_stream: Any
    if http_content_encoding.lower() in COMPRESSED_CONTENT_TYPES:
        raw_stream = gzip.GzipFile(fileobj=record.content_stream())
    else:
        raw_stream = record.content_stream()
    if "charset=" in http_content_type:
        charset: str = http_content_type.split("charset=")[-1].split(";")[0].strip()
        normalized_charset: str = ENCODING_ALIASES.get(charset.lower(), charset)
        return raw_stream.read().decode(normalized_charset, errors="replace")
    else:
        decoded_text: str = raw_stream.read().decode("utf-8", errors="replace")
        if "�" in decoded_text:
            decoded_text = decoded_text.encode("utf-8", errors="replace").decode(
                "latin-1", errors="replace"
            )
        return decoded_text


def convert_to_visible_ascii(html_text: str) -> str | None:
    """
    Locate an article-like subtree in the HTML, extract visible text, and
    normalize it to uppercase ASCII.

    Parameters
    ----------
    html_text : str
        Raw Unicode HTML string, usually produced by `to_text(...)`.

    Returns
    -------
    str | None
        Canonicalized article body on success; `None` when no sufficiently long
        article-like region can be found or when HTML parsing fails.

    Notes
    -----
    - The HTML is parsed into a tree and scanned using `ARTICLE_ROOT_XPATHS`.
      The longest element (by text length) among matches is used as the root.
    - Within that root, `ARTICLE_BODY_XPATHS` are evaluated to locate more
      specific body containers; the longest matching body element is chosen.
    - `extract_text_from_element` performs non-visible tag removal,
      whitespace normalization, ASCII filtering, and enforces the minimum
      25-token requirement.
    - If the selected element yields an empty string (too short) the search
      continues; only a non-empty result is returned.
    - All parsing errors (XML syntax, malformed HTML) result in `None`.
    """

    if not html_text.strip():
        return None
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
    except (ValueError, TypeError, html.etree.XMLSyntaxError, html.etree.ParserError):
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


def detect_firms(words: List[str], run_data: RunData) -> set[str]:
    """
    Identify firms referenced in an article by matching canonicalized name
    tokens against `run_data.firm_name_parts`.

    Parameters
    ----------
    words : List[str]
        Whitespace-split article tokens prior to canonicalization.
    run_data : RunData
        Run context providing `firm_name_parts`, a mapping from CIK to the set
        of canonicalized firm-name tokens used for matching.

    Returns
    -------
    set[str]
        Set of CIKs whose canonicalized name parts all appear in the article’s
        canonicalized word-frequency dictionary, subject to frequency guards.

    Notes
    -----
    - Article tokens are canonicalized via `word_canonicalizer`, discarding
      non-alphanumeric characters and uppercasing the result.
    - A firm is a candidate match when all canonicalized name parts appear in
      the article word set.
    - Corporate suffixes (from `NAME_SUFFIXES_SET`) are removed from the name
      before frequency checks.
    - Remaining name parts must each occur **at least twice** in the article.
    - This approach intentionally favors precision over recall and avoids
      spurious matches caused by short or noisy mentions.
    """

    word_frequency_dict: dict[str, int] = {}
    for word in words:
        canonical_word: str = word_canonicalizer(word)
        if canonical_word:
            word_frequency_dict[canonical_word] = word_frequency_dict.get(canonical_word, 0) + 1
    matched_firms_by_name: set[str] = set()
    for cik, name_parts in run_data.firm_name_parts.items():
        if name_parts.issubset(word_frequency_dict.keys()):
            name_parts_no_suffixes: set[str] = name_parts - NAME_SUFFIXES_SET
            appearance_count_no_suffixes: int = min(
                word_frequency_dict[part] for part in name_parts_no_suffixes
            )
            if appearance_count_no_suffixes > 1:
                matched_firms_by_name.add(cik)
    return matched_firms_by_name
