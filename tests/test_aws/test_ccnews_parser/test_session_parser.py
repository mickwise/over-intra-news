"""
Purpose
-------
Unit tests for `aws.ccnews_parser.session_parser`.

Key behaviors
-------------
- Verify that `parse_session`:
  - iterates over all configured WARC sample paths, and
  - returns one `SampleData` object per sample.
- Confirm that `process_sample`:
  - increments `records_scanned` for every WARC record seen,
  - only invokes `extract_data_from_record` for `rec_type == "response"`,
  - aggregates any non-None `ArticleData` into the returned `SampleData`.
- Validate that `extract_data_from_record`:
  - discards non-200 responses and non-HTML content types,
  - drops records with too few words or non-English language,
  - discards records with zero or more than three matched firms, and
  - increments `SampleMetadata` counters when gates are passed,
    constructing `ArticleData` on success.
- Ensure that `convert_to_visible_ascii`:
  - strips non-visible tags, normalizes whitespace, and uppercases text,
  - removes non-ASCII characters, and
  - returns None when HTML parsing fails or no article-like body can be
    extracted.
- Check that `detect_firms`:
  - canonicalizes article words via `word_canonicalizer`, and
  - returns CIKs whose canonicalized firm-name tokens (including suffixes)
    are all present in the article.

Conventions
-----------
- A minimal `_DummyLogger` mimics the subset of `InfraLogger` used by the
  parser, recording `debug`, `info`, `warning`, and `error` events.
- Lightweight `_DummyRecord` and header stubs stand in for `warcio`
  records and headers so tests exercise only local control flow.
- Language detection is deterministic via monkeypatching
  `langdetect.detect(...)`.
- Firm universes are reduced to a small synthetic mapping to make CIK
  matching behavior explicit.

Downstream usage
----------------
Run this module with `pytest` as part of the CI suite. The tests act as
executable documentation for how the news parser filters WARC records
into `ArticleData` payloads and how firm-name matching is implemented.
"""

from __future__ import annotations

import datetime as dt
import gzip
import io
import typing
from typing import Any, Dict, List, cast

import pytest

from aws.ccnews_parser import session_parser as sp
from aws.ccnews_parser.news_parser_utils import (
    ArticleData,
    FirmInfo,
    RunData,
    SampleData,
    SampleMetadata,
)
from infra.logging.infra_logger import InfraLogger


class _DummyLogger:
    """
    Purpose
    -------
    Minimal stand-in for `InfraLogger` capturing debug/info/warning/error calls.

    Key behaviors
    -------------
    - Records `.debug(...)`, `.info(...)`, `.warning(...)`, and `.error(...)`
      invocations.
    - Stores event name, positional args, and keyword args for assertions.

    Parameters
    ----------
    None

    Attributes
    ----------
    debugs : list[dict[str, Any]]
        Recorded debug-level events.
    infos : list[dict[str, Any]]
        Recorded info-level events.
    warnings : list[dict[str, Any]]
        Recorded warning-level events.
    errors : list[dict[str, Any]]
        Recorded error-level events.
    """

    def __init__(self) -> None:
        self.debugs: List[Dict[str, Any]] = []
        self.infos: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []

    def debug(self, event: str, *args: Any, **kwargs: Any) -> None:
        self.debugs.append({"event": event, "args": args, "kwargs": kwargs})

    def info(self, event: str, *args: Any, **kwargs: Any) -> None:
        self.infos.append({"event": event, "args": args, "kwargs": kwargs})

    def warning(self, event: str, *args: Any, **kwargs: Any) -> None:
        self.warnings.append({"event": event, "args": args, "kwargs": kwargs})

    def error(self, event: str, *args: Any, **kwargs: Any) -> None:
        self.errors.append({"event": event, "args": args, "kwargs": kwargs})


class _DummyHeaders:
    """
    Purpose
    -------
    Lightweight stand-in for warcio HTTP/WARC headers.

    Key behaviors
    -------------
    - Exposes `.get_statuscode()`, `.get(...)`, and `.get_header(...)` methods
      matching the subset used by `session_parser`.

    Parameters
    ----------
    status : int
        HTTP status code to be returned by `.get_statuscode()`.
    headers : dict[str, str] | None
        Mapping of header names to values used by `.get(...)` and
        `.get_header(...)`.
    """

    def __init__(self, status: int = 200, headers: Dict[str, str] | None = None) -> None:
        self._status = status
        self._headers = headers or {}

    def get_statuscode(self) -> int:
        return self._status

    def get(self, key: str, default: str | None = None) -> str:
        return self._headers.get(key, default or "")

    # warcio API compatibility: session_parser calls `get_header(...)`
    def get_header(self, key: str, default: str | None = None) -> str:
        return self.get(key, default)


class _DummyRecord:
    """
    Purpose
    -------
    Simplified warcio record representation used by parsing tests.

    Key behaviors
    -------------
    - Exposes `.rec_type`, `.http_headers`, `.rec_headers`, and
      `.content_stream()` as expected by `session_parser`.

    Parameters
    ----------
    rec_type : str
        WARC record type (e.g., "response" or "request").
    http_status : int
        HTTP status code returned by `.http_headers.get_statuscode()`.
    http_headers : dict[str, str] | None
        HTTP headers used by `.http_headers.get_header(...)`.
    rec_headers : dict[str, str] | None
        WARC headers used by `.rec_headers.get_header(...)`.
    body : bytes | None
        Raw payload returned from `.content_stream().read()`.
    """

    def __init__(
        self,
        rec_type: str = "response",
        http_status: int = 200,
        http_headers: Dict[str, str] | None = None,
        rec_headers: Dict[str, str] | None = None,
        body: bytes | None = None,
    ) -> None:
        self.rec_type = rec_type
        self.http_headers = _DummyHeaders(status=http_status, headers=http_headers)
        self.rec_headers = _DummyHeaders(status=http_status, headers=rec_headers)
        self._body = body or b""

    def content_stream(self) -> io.BytesIO:
        return io.BytesIO(self._body)


class _DummyS3Client:
    """
    Purpose
    -------
    Minimal S3 client stub for `extract_warc_sample` tests.

    Key behaviors
    -------------
    - Returns a preconfigured gzip-compressed payload for any object key.
    - Records requests so tests can assert on bucket/key usage.

    Parameters
    ----------
    payload : bytes
        Gzip-compressed payload that will be exposed via the returned
        object's `"Body"` attribute.
    """

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.calls: List[Dict[str, str]] = []

    def get_object(self, Bucket: str, Key: str) -> Dict[str, Any]:
        self.calls.append({"Bucket": Bucket, "Key": Key})
        return {"Body": io.BytesIO(self._payload)}


def test_parse_session_invokes_process_sample_for_each_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that `parse_session` calls `process_sample` once per sample and
    returns a `SampleData` list of matching length.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `process_sample` so that calls can be intercepted and
        a synthetic `SampleData` object returned for each sample without
        touching WARC or S3.

    Returns
    -------
    None
        The test passes if all sample paths are observed by the stub,
        `parse_session` returns the same number of `SampleData` objects as
        there are samples, and each element of the result is a `SampleData`
        instance.

    Raises
    ------
    AssertionError
        If any sample is not passed to `process_sample`, if the result list
        length does not match the number of samples, or if any element of
        the result is not a `SampleData` instance.

    Notes
    -----
    - Also asserts that the logger records two debug events per sample
      (pre- and post-processing), serving as an indirect check that the
      logging hooks are exercised for each sample.
    """

    samples = [
        "s3://bucket/2020/01/01/intraday/sample1.warc.gz",
        "s3://bucket/2020/01/01/intraday/sample2.warc.gz",
        "s3://bucket/2020/01/01/intraday/sample3.warc.gz",
    ]
    dummy_logger = _DummyLogger()
    run_data = RunData(
        date=dt.date(2020, 1, 1),
        session="intraday",
        bucket="bucket",
        firm_info_dict={},
        samples=samples,
        logger=cast(InfraLogger, dummy_logger),
        s3_client=object(),
    )

    seen_samples: List[str] = []

    def fake_process_sample(sample: str, rd: RunData) -> SampleData:
        seen_samples.append(sample)
        md = SampleMetadata(
            records_scanned=1,
            html_200_count=0,
            unhandled_errors=0,
            decompression_errors=0,
            ge_25_words=0,
            too_long_articles=0,
            english_count=0,
            matched_any_firm=0,
            articles_kept=0,
        )
        return SampleData(article_data=[], sample_metadata=md)

    monkeypatch.setattr(sp, "process_sample", fake_process_sample)

    result = sp.parse_session(run_data)

    assert seen_samples == samples
    assert len(result) == len(samples)
    assert all(isinstance(sd, SampleData) for sd in result)
    # parse_session logs pre- and post-sample debug entries
    assert len(dummy_logger.debugs) == 2 * len(samples)


def test_process_sample_scans_all_records_and_collects_articles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Confirm that `process_sample` scans all WARC records, filters by record
    type, and aggregates only non-None `ArticleData` results.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `extract_warc_sample`, `ArchiveIterator`, and
        `extract_data_from_record` so that record iteration and article
        collection can be controlled without real I/O.

    Returns
    -------
    None
        The test passes if `records_scanned` equals the number of synthetic
        records, only `response` records are passed to
        `extract_data_from_record`, and exactly one `ArticleData` object is
        kept as configured by the stub.

    Raises
    ------
    AssertionError
        If the records scanned count is incorrect, if the number of records
        passed to `extract_data_from_record` is wrong, or if the number of
        collected `ArticleData` objects does not match the expected value.

    Notes
    -----
    - Uses a mixed set of `request` and `response` records to ensure that
      non-response records are ignored for article extraction while still
      contributing to the `records_scanned` counter.
    """

    records = [
        _DummyRecord(rec_type="request"),
        _DummyRecord(rec_type="response"),
        _DummyRecord(rec_type="response"),
    ]

    class _Context:
        def __init__(self, fileobj: Any) -> None:
            self._fileobj = fileobj

        def __enter__(self) -> Any:
            return self._fileobj

        def __exit__(self, exc_type, exc, tb) -> typing.Literal[False]:
            return False

    def fake_extract_warc_sample(sample: str, run_data: RunData) -> _Context:
        return _Context(fileobj="dummy-file")

    def fake_archive_iterator(fileobj: Any):
        assert fileobj == "dummy-file"
        # Return an iterator, not a list, so `next(iterator)` works in `process_sample`.
        return iter(records)

    seen_records: List[_DummyRecord] = []

    def fake_extract_data_from_record(
        sample: str,
        record: Any,
        sample_metadata: SampleMetadata,
        run_data: RunData,
    ) -> ArticleData | None:
        seen_records.append(record)
        # Only the second record yields an article; others are dropped.
        if record is records[1]:
            return ArticleData(
                warc_path=sample,
                warc_date_utc="2020-01-01T00:00:00Z",
                url="http://example.com/1",
                http_status=200,
                http_content_type="text/html",
                payload_digest="digest",
                ny_date=run_data.date,
                session=run_data.session,
                cik_list=["0001"],
                word_count=100,
                language="en",
                full_text="TEXT",
            )
        return None

    monkeypatch.setattr(sp, "extract_warc_sample", fake_extract_warc_sample)
    monkeypatch.setattr(sp, "ArchiveIterator", fake_archive_iterator)
    monkeypatch.setattr(sp, "extract_data_from_record", fake_extract_data_from_record)

    dummy_logger = _DummyLogger()
    run_data = RunData(
        date=dt.date(2020, 1, 1),
        session="intraday",
        bucket="bucket",
        firm_info_dict={},
        samples=[],
        logger=cast(InfraLogger, dummy_logger),
        s3_client=object(),
    )

    sample_data = sp.process_sample("s3://bucket/sample.warc.gz", run_data)

    # All records should be counted.
    assert sample_data.sample_metadata.records_scanned == len(records)
    # Only "response" records should be passed to extract_data_from_record.
    assert len(seen_records) == 2
    # Only one article is kept by our stub.
    assert len(sample_data.article_data) == 1


def test_extract_warc_sample_opens_gzip_from_s3_object() -> None:
    """
    Ensure that `extract_warc_sample` reads a gzip-compressed WARC payload
    from S3 and returns a `gzip.GzipFile` that yields the original bytes.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if `get_object` is called once with the expected
        bucket and key derived from the `s3://` URL and if reading from the
        returned `gzip.GzipFile` produces the original uncompressed payload.

    Raises
    ------
    AssertionError
        If the S3 call is not made with the correct bucket/key or if the
        decompressed data read from the returned file object does not match
        the original payload.

    Notes
    -----
    - Relies on a `_DummyS3Client` that records `get_object` calls and
      returns an in-memory gzip-compressed byte stream to avoid external
      S3 dependencies.
    """

    original_payload = b"hello world"
    gz_payload = gzip.compress(original_payload)
    s3_client = _DummyS3Client(payload=gz_payload)
    dummy_logger = _DummyLogger()

    run_data = RunData(
        date=dt.date(2020, 1, 1),
        session="intraday",
        bucket="my-bucket",
        firm_info_dict={},
        samples=[],
        logger=cast(InfraLogger, dummy_logger),
        s3_client=s3_client,
    )

    sample = "s3://my-bucket/path/to/file.warc.gz"
    warc_file = sp.extract_warc_sample(sample, run_data)

    # S3 client should have been called with the stripped bucket/key.
    assert len(s3_client.calls) == 1
    call = s3_client.calls[0]
    assert call["Bucket"] == "my-bucket"
    assert call["Key"] == "path/to/file.warc.gz"

    with warc_file as f:
        data = f.read()
    assert data == original_payload


def test_extract_data_from_record_filters_non_200_responses() -> None:
    """
    Verify that `extract_data_from_record` discards non-200 HTTP responses
    and does not increment article-related counters.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if the function returns None, `html_200_count`
        remains zero for a 404 response, and `articles_kept` is not
        incremented.

    Raises
    ------
    AssertionError
        If an `ArticleData` object is returned for a non-200 response or if
        any of the relevant metadata counters are incorrectly updated.

    Notes
    -----
    - This test isolates the HTTP status gate, ensuring that later filters
      (such as content type and language) are never reached when the status
      code is not 200.
    """

    dummy_logger = _DummyLogger()
    run_data = RunData(
        date=dt.date(2020, 1, 1),
        session="intraday",
        bucket="bucket",
        firm_info_dict={},
        samples=[],
        logger=cast(InfraLogger, dummy_logger),
        s3_client=object(),
    )
    sample_metadata = sp.initialize_sample_metadata()

    record = _DummyRecord(
        rec_type="response",
        http_status=404,
        http_headers={"Content-Type": "text/html"},
        rec_headers={
            "WARC-Date": "2020-01-01T00:00:00Z",
            "WARC-Target-URI": "http://example.com/",
            "WARC-Payload-Digest": "digest",
        },
        body=b"<html>ignored</html>",
    )

    result = sp.extract_data_from_record(
        "s3://bucket/sample.warc.gz", record, sample_metadata, run_data
    )
    assert result is None
    assert sample_metadata.html_200_count == 0
    assert sample_metadata.articles_kept == 0


def test_extract_data_from_record_filters_non_html_content_type() -> None:
    """
    Ensure that `extract_data_from_record` drops non-HTML responses even
    when the HTTP status is 200.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if the function returns None for a 200 response with
        `Content-Type` not containing `"text/html"`, `html_200_count` is
        incremented once, and `ge_25_words` and `articles_kept` remain zero.

    Raises
    ------
    AssertionError
        If an `ArticleData` object is returned for a non-HTML content type
        or if the associated metadata counters are not updated as expected.

    Notes
    -----
    - Confirms that the status code gate is applied before the content-type
      filter, and that the content-type check correctly excludes non-HTML
      payloads from further processing.
    """

    dummy_logger = _DummyLogger()
    run_data = RunData(
        date=dt.date(2020, 1, 1),
        session="intraday",
        bucket="bucket",
        firm_info_dict={},
        samples=[],
        logger=cast(InfraLogger, dummy_logger),
        s3_client=object(),
    )
    sample_metadata = sp.initialize_sample_metadata()

    record = _DummyRecord(
        rec_type="response",
        http_status=200,
        http_headers={"Content-Type": "application/json"},
        rec_headers={
            "WARC-Date": "2020-01-01T00:00:00Z",
            "WARC-Target-URI": "http://example.com/api",
            "WARC-Payload-Digest": "digest",
        },
        body=b"{}",
    )

    result = sp.extract_data_from_record(
        "s3://bucket/sample.warc.gz", record, sample_metadata, run_data
    )
    assert result is None
    # HTML 200 count incremented before Content-Type gate
    assert sample_metadata.html_200_count == 1
    assert sample_metadata.articles_kept == 0
    assert sample_metadata.ge_25_words == 0


def test_extract_data_from_record_returns_article_when_all_gates_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Confirm that `extract_data_from_record` returns a fully populated
    `ArticleData` when all filtering gates (status, content type, length,
    language, firm match) are satisfied.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `convert_to_visible_ascii` and `langdetect.detect` so
        that visible text and language detection are deterministic.

    Returns
    -------
    None
        The test passes if an `ArticleData` instance is returned with the
        expected URL, date, language, word count, and CIK list, and if all
        relevant `SampleMetadata` counters are incremented exactly once.

    Raises
    ------
    AssertionError
        If the function returns None, if any field on the resulting
        `ArticleData` is incorrect, or if the metadata counters do not match
        the expected values after a successful pass through all gates.

    Notes
    -----
    - Uses a single synthetic firm (`ACME HOLDINGS INC`) and repeated text
      to guarantee at least 25 words and a single firm match, exercising
      the firm-matching logic and the English-only language filter.
    """

    visible_text = ("ACME HOLDINGS INC REPORTS STRONG EARNINGS " * 10).strip()

    def fake_convert_to_visible_ascii(html_text: str) -> str | None:
        return visible_text

    def fake_detect(text: str) -> str:
        return "en"

    monkeypatch.setattr(sp, "convert_to_visible_ascii", fake_convert_to_visible_ascii)
    monkeypatch.setattr(sp.langdetect, "detect", fake_detect)

    dummy_logger = _DummyLogger()
    firm_info_dict = {
        "0001": FirmInfo(cik="0001", firm_name="ACME HOLDINGS INC"),
    }
    run_data = RunData(
        date=dt.date(2020, 1, 2),
        session="intraday",
        bucket="bucket",
        firm_info_dict=firm_info_dict,
        samples=[],
        logger=cast(InfraLogger, dummy_logger),
        s3_client=object(),
    )
    sample_metadata = sp.initialize_sample_metadata()

    record = _DummyRecord(
        rec_type="response",
        http_status=200,
        http_headers={"Content-Type": "text/html; charset=utf-8"},
        rec_headers={
            "WARC-Date": "2020-01-02T03:04:05Z",
            "WARC-Target-URI": "http://example.com/story",
            "WARC-Payload-Digest": "sha1:ABC",
        },
        body=b"<html>ignored</html>",
    )

    article = sp.extract_data_from_record(
        "s3://bucket/2020/01/02/intraday/sample.warc.gz",
        record,
        sample_metadata,
        run_data,
    )

    assert isinstance(article, ArticleData)
    assert article.url == "http://example.com/story"
    assert article.ny_date == run_data.date
    assert set(article.cik_list) == {"0001"}
    assert article.word_count == len(visible_text.split())
    assert article.language == "en"

    assert sample_metadata.html_200_count == 1
    assert sample_metadata.ge_25_words == 1
    assert sample_metadata.english_count == 1
    assert sample_metadata.matched_any_firm == 1
    assert sample_metadata.articles_kept == 1


def test_to_text_uses_declared_charset() -> None:
    """
    Check that `to_text` honors an explicit charset specified in the HTTP
    Content-Type header when decoding the response body.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if the decoded string exactly matches the original
        text encoded with the declared charset.

    Raises
    ------
    AssertionError
        If the decoded text differs from the original when a correct
        `charset=` parameter is provided in the content type.

    Notes
    -----
    - Uses a body encoded with `latin-1` and a matching
      `"text/html; charset=latin-1"` header to confirm that the function
      does not fall back to UTF-8 in the presence of an explicit charset.
    """

    body = "héllo".encode("latin-1")
    record = _DummyRecord(
        body=body,
        http_headers={},
        rec_headers={},
    )

    text = sp.to_text(
        record,
        http_content_type="text/html; charset=latin-1",
        http_content_encoding="",
    )
    assert text == "héllo"


def test_to_text_falls_back_when_replacement_char_present() -> None:
    """
    Ensure that `to_text` performs a secondary decode path when the initial
    UTF-8 decode introduces replacement characters (�).

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if the returned string does not contain the Unicode
        replacement character and still includes the valid tail substring
        from the original byte sequence.

    Raises
    ------
    AssertionError
        If the decoded text contains replacement characters or if the valid
        portion of the original payload is not preserved in the result.

    Notes
    -----
    - Uses an invalid UTF-8 byte prefix followed by ASCII characters to
      force the first decode to produce `�`, triggering the fallback logic
      intended to improve robustness for mis-encoded content.
    """

    # Invalid UTF-8 sequence to trigger replacement characters.
    body = b"\xff\xfeabc"
    record = _DummyRecord(
        body=body,
        http_headers={},
        rec_headers={},
    )

    text = sp.to_text(
        record,
        http_content_type="text/html",
        http_content_encoding="",
    )
    assert "�" not in text
    assert "abc" in text


def test_convert_to_visible_ascii_strips_non_visible_and_uppercases() -> None:
    """
    Verify that `convert_to_visible_ascii` removes non-visible HTML sections
    and returns normalized uppercase ASCII text for an article-like subtree.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if content inside non-visible tags (head, style,
        script, etc.) is removed, whitespace is collapsed, and the remaining
        visible article body text is returned as uppercase ASCII.

    Raises
    ------
    AssertionError
        If non-visible content appears in the result, if casing is not
        converted to uppercase, or if extraneous whitespace is not
        collapsed.

    Notes
    -----
    - Constructs a small HTML snippet with a semantic `<article>` root and
      nested body content so that `ARTICLE_ROOT_XPATHS` and
      `ARTICLE_BODY_XPATHS` both match.
    - Includes `head/style/script` blocks to exercise tag stripping,
      whitespace normalization, and ASCII-only encoding behavior.
    """

    html_text = """
    <html>
      <head>
        <title>Ignored</title>
        <style>.cls { color: red; }</style>
        <script>headScript()</script>
      </head>
      <body>
        <article>
          <div class="article-body">
            Hello world this is an example article text repeated many times
            to ensure we cross the twenty five word threshold hello world
            this is an example article text repeated many times to ensure
            we cross the twenty five word threshold
            <script>bad()</script>
          </div>
        </article>
      </body>
    </html>
    """

    result = sp.convert_to_visible_ascii(html_text)
    assert result is not None

    # Script/style/title content should be stripped.
    assert "bad()" not in result
    assert "headScript()" not in result
    assert "Ignored" not in result

    # Text should be uppercased and reasonably collapsed.
    assert result == result.upper()
    assert "  " not in result

    # Core visible words should still be present.
    assert "HELLO WORLD" in result
    assert len(result.split()) >= 25


def test_convert_to_visible_ascii_returns_none_on_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Confirm that `convert_to_visible_ascii` returns None when HTML parsing
    fails due to an exception.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `html.fromstring` so that it raises a parsing exception
        for any input.

    Returns
    -------
    None
        The test passes if the function returns None after the underlying
        parser raises, without propagating the exception.

    Raises
    ------
    AssertionError
        If a non-None value is returned or if the parsing exception escapes
        instead of being handled.

    Notes
    -----
    - Simulates a low-level parser failure (e.g., `ValueError`) to validate
      that the function's try/except block correctly treats such errors as
      an inability to extract visible text.
    """

    def fake_fromstring(text: str) -> Any:
        raise ValueError("bad html")

    monkeypatch.setattr(sp.html, "fromstring", fake_fromstring)

    result = sp.convert_to_visible_ascii("<not really html>")
    assert result is None


def test_detect_firms_matches_when_all_name_tokens_present() -> None:
    """
    Check that `detect_firms` matches firms by canonicalized name tokens
    when all tokens are present and non-suffix tokens appear with sufficient
    frequency.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if the function returns only the CIK for the firm
        whose canonicalized name tokens all appear in the article words and
        whose non-suffix tokens occur at least twice, while excluding firms
        that do not meet this criterion.

    Raises
    ------
    AssertionError
        If the expected CIK is not present in the result set or if an
        ineligible firm is incorrectly included.

    Notes
    -----
    - Exercises canonicalization and subset matching by constructing a small
      firm universe where only one firm's stripped name tokens
      (`ACME HOLDINGS`) are present in the article word list, and the
      non-suffix token (`ACME`) appears at least twice to satisfy the
      frequency guard in `detect_firms`.
    """

    words = [
        "Acme",
        "Holdings",
        "Inc",
        "Acme",
        "reports",
        "strong",
        "profits",
    ]
    dummy_logger = _DummyLogger()
    firm_info_dict = {
        "0001": FirmInfo(cik="0001", firm_name="ACME HOLDINGS INC"),
        "0002": FirmInfo(cik="0002", firm_name="FOO CORPORATION"),
    }

    run_data = RunData(
        date=dt.date(2020, 1, 1),
        session="intraday",
        bucket="bucket",
        firm_info_dict=firm_info_dict,
        samples=[],
        logger=cast(InfraLogger, dummy_logger),
        s3_client=object(),
    )

    matched = sp.detect_firms(words, run_data)
    assert matched == {"0001"}
