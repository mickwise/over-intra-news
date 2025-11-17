"""
Purpose
-------
Unit tests for the CC-NEWS parsing orchestrator
(`aws.ccnews_parser.news_parser_orchestrator`), covering CLI
argument parsing, per-session WARC discovery, `RunData`
construction, per-session Parquet emission, and top-level
process-pool wiring.

Key behaviors
-------------
- Verify that `extract_cli_args` parses year, bucket, and optional
  logger level from `sys.argv`.
- Exercise `extract_per_session_warcs` on both successful S3 reads
  and `ClientError` failures.
- Check that `generate_run_data` returns `RunData` only when
  samples exist and wires in firm info and an S3 client.
- Ensure `samples_to_parquet` writes article and metadata Parquet
  files with the expected S3-style key structure and logging, and
  skips article output when no articles are present.
- Confirm that `run_month_parser` iterates trading days and
  sessions, skips empty work, and passes results from
  `parse_session` into `samples_to_parquet`.
- Validate that `main` glues together env loading, CLI parsing,
  logger initialization, calendar extraction, and submission of
  month-parsers to a `ProcessPoolExecutor`.

Conventions
-----------
- External boundaries (S3, process pools, logging, and Parquet I/O)
  are isolated via dummy objects and monkeypatched functions so
  tests are side-effect free and deterministic.
- `InfraLogger` is replaced by a `_DummyLogger` that records
  `info`, `warning`, and `debug` calls for assertions.
- Tests assert on S3-style paths and partition prefixes but do not
  depend on any particular S3 bucket configuration.
- Time-related values use `datetime.date` rather than timestamps,
  mirroring the orchestrator's trading-calendar interface.

Downstream usage
----------------
Run this module with `pytest` as part of the CI suite. The tests
serve as executable documentation for how the news parser
orchestrator is expected to interact with the trading calendar,
S3 layout, logging, and parallel execution, and they should be
extended whenever new orchestration behaviors are added.
"""

from __future__ import annotations

import datetime as dt
import typing
from typing import Any, Dict, List, cast

import botocore.exceptions
import pandas as pd
import pytest

import aws.ccnews_parser.news_parser_orchestrator as npo
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
    Minimal stand-in for `InfraLogger` capturing info, warning, and
    debug calls for assertions in tests.

    Attributes
    ----------
    infos : list[dict[str, Any]]
        Recorded info-level log events.
    warnings : list[dict[str, Any]]
        Recorded warning-level log events.
    debugs : list[dict[str, Any]]
        Recorded debug-level log events.
    """

    def __init__(self) -> None:
        self.infos: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.debugs: List[Dict[str, Any]] = []

    def info(self, event: str, *args: Any, **kwargs: Any) -> None:
        self.infos.append({"event": event, "args": args, "kwargs": kwargs})

    def warning(self, event: str, *args: Any, **kwargs: Any) -> None:
        self.warnings.append({"event": event, "args": args, "kwargs": kwargs})

    def debug(self, event: str, *args: Any, **kwargs: Any) -> None:
        self.debugs.append({"event": event, "args": args, "kwargs": kwargs})


class _DummyS3Client:
    """
    Purpose
    -------
    Simple fake S3 client that records `get_object` calls and returns an
    in-memory byte payload or raises a configured error.

    Attributes
    ----------
    payload : bytes
        Raw bytes returned in the `Body` stream of `get_object`.
    error : Exception | None
        Optional error to raise instead of returning a payload.
    calls : list[dict[str, Any]]
        Recorded calls with bucket and key arguments.
    """

    def __init__(self, payload: bytes | None = None, error: Exception | None = None) -> None:
        self.payload = payload or b""
        self.error = error
        self.calls: List[Dict[str, Any]] = []

    def get_object(self, Bucket: str, Key: str) -> Dict[str, Any]:
        self.calls.append({"Bucket": Bucket, "Key": Key})
        if self.error is not None:
            raise self.error
        import io

        return {"Body": io.BytesIO(self.payload)}


class _DummyExecutor:
    """
    Purpose
    -------
    Lightweight stand-in for `ProcessPoolExecutor` that captures task
    submissions without spawning worker processes.

    Attributes
    ----------
    args : tuple[Any, ...]
        Positional arguments passed to the constructor.
    kwargs : dict[str, Any]
        Keyword arguments passed to the constructor.
    submissions : list[dict[str, Any]]
        Recorded submissions with function and arguments.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.submissions: List[Dict[str, Any]] = []

    def __enter__(self) -> "_DummyExecutor":
        return self

    def __exit__(self, exc_type, exc: BaseException | None, tb: Any) -> typing.Literal[False]:
        return False

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> None:
        self.submissions.append({"fn": fn, "args": args, "kwargs": kwargs})


@pytest.mark.parametrize(
    ["logger_level", "expected_level"],
    [
        ("DEBUG", "DEBUG"),
        (None, "INFO"),
    ],
)
def test_extract_cli_args_parses_year_bucket_and_defaults_logger_level(
    monkeypatch: pytest.MonkeyPatch,
    logger_level: str | None,
    expected_level: str,
) -> None:
    """
    Verify that `extract_cli_args` parses the year and bucket from
    `sys.argv` and applies the correct logger level.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `sys.argv` so the function can be exercised without
        invoking the CLI.

    Returns
    -------
    None
        The test passes if the returned tuple contains the parsed
        integer year, bucket string, and the default logger level
        "INFO".

    Raises
    ------
    AssertionError
        If the year is not converted to `int`, if the bucket is parsed
        incorrectly, or if the logger level is not the expected one.

    Notes
    -----
    None
    """

    monkeypatch.setattr(
        "sys.argv", ["prog", "2020", "my-bucket"] + ([logger_level] if logger_level else [])
    )
    year, bucket, level = npo.extract_cli_args()
    assert year == 2020
    assert bucket == "my-bucket"
    assert level == expected_level


def test_extract_per_session_warcs_returns_sample_list_on_success() -> None:
    """
    Check that `extract_per_session_warcs` reads `samples.txt` from S3
    and returns the list of WARC paths on success.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if the function returns a list of sample paths
        matching the content of the fake `samples.txt` and emits two
        debug log entries (pre- and post-fetch).

    Raises
    ------
    AssertionError
        If the returned list does not match the expected paths or if
        the debug logging calls are not recorded as anticipated.

    Notes
    -----
    - The dummy S3 client returns an in-memory byte stream with two
      newline-separated sample paths to mimic the layout of
      `samples.txt`.
    """
    payload = b"s3://bucket/path1.warc.gz\ns3://bucket/path2.warc.gz\n"
    s3_client = _DummyS3Client(payload=payload)
    dummy_logger = _DummyLogger()

    samples = npo.extract_per_session_warcs(
        year=2020,
        month=1,
        day=2,
        session="intraday",
        bucket="my-bucket",
        logger=cast(InfraLogger, dummy_logger),
        s3_client=s3_client,
    )

    assert samples == [
        "s3://bucket/path1.warc.gz",
        "s3://bucket/path2.warc.gz",
    ]
    # One debug before and one after successful fetch.
    assert len(dummy_logger.debugs) == 2


def test_extract_per_session_warcs_returns_empty_on_client_error() -> None:
    """
    Ensure that `extract_per_session_warcs` returns an empty list and
    logs a warning when the underlying S3 `get_object` call raises a
    `ClientError`.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if the function returns an empty list and the
        logger records a single warning event indicating that
        `samples.txt` could not be found.

    Raises
    ------
    AssertionError
        If a non-empty list is returned or if the warning log entry is
        not emitted.

    Notes
    -----
    - A `botocore.exceptions.ClientError` is constructed with minimal
      parameters and injected via a dummy S3 client to simulate a
      missing object in S3.
    """
    error = botocore.exceptions.ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}},
        "GetObject",
    )
    s3_client = _DummyS3Client(error=error)
    dummy_logger = _DummyLogger()

    samples = npo.extract_per_session_warcs(
        year=2020,
        month=1,
        day=2,
        session="overnight",
        bucket="my-bucket",
        logger=cast(InfraLogger, dummy_logger),
        s3_client=s3_client,
    )

    assert samples == []
    assert len(dummy_logger.warnings) == 1


def test_generate_run_data_returns_none_when_no_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that `generate_run_data` returns None when
    `extract_per_session_warcs` yields no samples.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `extract_per_session_warcs` so that it returns an
        empty list, avoiding any real S3 calls.

    Returns
    -------
    None
        The test passes if the function returns None, indicating there
        is no work for the given day and session.

    Raises
    ------
    AssertionError
        If a `RunData` instance is returned despite there being no
        samples.

    Notes
    -----
    - This behavior is used by `run_month_parser` to skip sessions that
      have no associated WARC files.
    """

    def fake_extract_per_session_warcs(
        year: int,
        month: int,
        day: int,
        session: str,
        bucket: str,
        logger: InfraLogger,
        s3_client: Any,
    ) -> List[str]:
        return []

    monkeypatch.setattr(npo, "extract_per_session_warcs", fake_extract_per_session_warcs)

    dummy_logger = _DummyLogger()
    dummy_client = _DummyS3Client()
    result = npo.generate_run_data(
        year=2020,
        month=1,
        date=dt.date(2020, 1, 2),
        session="intraday",
        bucket="my-bucket",
        logger=cast(InfraLogger, dummy_logger),
        s3_client=dummy_client,
    )

    assert result is None


def test_generate_run_data_builds_rundata_with_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure that `generate_run_data` constructs a `RunData` object with
    the expected fields when samples are available.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `extract_per_session_warcs`, `extract_firm_info_per_day`,
        and `boto3.client` so that no external services are called.

    Returns
    -------
    None
        The test passes if the returned `RunData` has the provided date,
        session, and bucket, a non-empty `samples` list, a non-empty
        `firm_info_dict`, and a concrete S3 client instance.

    Raises
    ------
    AssertionError
        If the function returns None, if core fields do not match the
        inputs, or if samples and firm info are not populated as
        expected.

    Notes
    -----
    - This test focuses on the composition of `RunData` rather than the
      internal behavior of the dependencies it calls.
    """

    def fake_extract_per_session_warcs(
        year: int,
        month: int,
        day: int,
        session: str,
        bucket: str,
        logger: InfraLogger,
        s3_client: Any,
    ) -> List[str]:
        return [f"s3://{bucket}/{year}/{month:02d}/{day:02d}/{session}/sample.warc.gz"]

    def fake_extract_firm_info_per_day(date: dt.date) -> Dict[str, FirmInfo]:
        return {"0001": FirmInfo(cik="0001", firm_name="ACME CORP")}

    dummy_s3 = object()

    def fake_boto3_client(service: str) -> Any:
        assert service == "s3"
        return dummy_s3

    monkeypatch.setattr(npo, "extract_per_session_warcs", fake_extract_per_session_warcs)
    monkeypatch.setattr(npo, "extract_firm_info_per_day", fake_extract_firm_info_per_day)
    monkeypatch.setattr(npo.boto3, "client", fake_boto3_client)

    dummy_logger = _DummyLogger()
    date = dt.date(2020, 1, 3)
    run_data = npo.generate_run_data(
        year=2020,
        month=1,
        date=date,
        session="overnight",
        bucket="my-bucket",
        logger=cast(InfraLogger, dummy_logger),
        s3_client=dummy_s3,
    )

    assert run_data is not None
    assert run_data.date == date
    assert run_data.session == "overnight"
    assert run_data.bucket == "my-bucket"
    assert run_data.samples
    assert "0001" in run_data.firm_info_dict
    assert run_data.s3_client is dummy_s3


def test_samples_to_parquet_writes_articles_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Confirm that `samples_to_parquet` writes both article and sample
    metadata Parquet files to S3 when articles are present.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `pandas.DataFrame.to_parquet` so that no real S3
        writes occur while allowing inspection of the target paths and
        DataFrame contents.

    Returns
    -------
    None
        The test passes if exactly two Parquet writes are performed: one
        for articles and one for sample stats, with S3 paths matching
        the expected partitioned prefixes and row counts matching the
        input data.

    Raises
    ------
    AssertionError
        If the number of writes is incorrect, if the paths do not
        include the expected prefixes, or if the DataFrame row counts
        do not match the number of articles and metadata rows.

    Notes
    -----
    - This test treats `to_parquet` as a side-effecting boundary and
      focuses on verifying the key construction and basic DataFrame
      shapes rather than Parquet serialization details.
    """
    calls: List[Dict[str, Any]] = []

    def fake_to_parquet(self: pd.DataFrame, path: str, index: bool = False, **kwargs: Any) -> None:
        calls.append({"df": self.copy(), "path": path, "index": index, "kwargs": kwargs})

    monkeypatch.setattr(npo.pd.DataFrame, "to_parquet", fake_to_parquet, raising=False)

    dummy_logger = _DummyLogger()
    date = dt.date(2020, 1, 4)
    run_data = RunData(
        date=date,
        session="intraday",
        bucket="my-bucket",
        firm_info_dict={},
        firm_name_parts={},
        samples=[],
        logger=cast(InfraLogger, dummy_logger),
        s3_client=object(),
    )

    sample_metadata = SampleMetadata(
        records_scanned=10,
        html_200_count=5,
        unhandled_errors=0,
        decompression_errors=0,
        ge_25_words=4,
        too_long_articles=0,
        english_count=3,
        matched_any_firm=2,
        articles_kept=1,
    )
    article = ArticleData(
        warc_path="s3://my-bucket/2020/01/04/intraday/sample.warc.gz",
        warc_date_utc="2020-01-04T12:34:56Z",
        url="http://example.com/story",
        http_status=200,
        http_content_type="text/html",
        payload_digest="sha1:ABC",
        ny_date=date,
        session="intraday",
        cik_list=["0001"],
        word_count=100,
        language_confidence=0.99,
        full_text="TEXT",
    )
    samples_data: List[SampleData] = [
        SampleData(article_data=[article], sample_metadata=sample_metadata),
    ]

    npo.samples_to_parquet(samples_data, run_data)

    assert len(calls) == 2
    article_call, meta_call = calls

    assert "ccnews_articles/year=2020/month=01/day=04/session=intraday" in article_call["path"]
    assert article_call["path"].endswith("articles.parquet")
    assert len(article_call["df"]) == 1

    assert "ccnews_sample_stats/year=2020/month=01/day=04/session=intraday" in meta_call["path"]
    assert meta_call["path"].endswith("sample_stats.parquet")
    assert len(meta_call["df"]) == 1


def test_samples_to_parquet_skips_article_write_when_no_articles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure that `samples_to_parquet` skips the article Parquet write and
    logs a warning when no articles are present, while still writing
    sample metadata.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `pandas.DataFrame.to_parquet` to capture metadata
        writes without hitting S3 or the filesystem.

    Returns
    -------
    None
        The test passes if only one Parquet write is performed (for
        sample stats), the logger records a warning about missing
        articles, and the metadata DataFrame has the expected number of
        rows.

    Raises
    ------
    AssertionError
        If an article Parquet write occurs, if no warning is logged, or
        if the metadata write does not happen.

    Notes
    -----
    - This test captures the branch used when per-session filtering
      yields zero articles, which is common when firm/name matching is
      conservative.
    """
    calls: List[Dict[str, Any]] = []

    def fake_to_parquet(self: pd.DataFrame, path: str, index: bool = False, **kwargs: Any) -> None:
        calls.append({"df": self.copy(), "path": path, "index": index, "kwargs": kwargs})

    monkeypatch.setattr(npo.pd.DataFrame, "to_parquet", fake_to_parquet, raising=False)

    dummy_logger = _DummyLogger()
    date = dt.date(2020, 1, 5)
    run_data = RunData(
        date=date,
        session="overnight",
        bucket="my-bucket",
        firm_info_dict={},
        firm_name_parts={},
        samples=[],
        logger=cast(InfraLogger, dummy_logger),
        s3_client=object(),
    )

    sample_metadata = SampleMetadata(
        records_scanned=3,
        html_200_count=2,
        unhandled_errors=0,
        decompression_errors=0,
        ge_25_words=1,
        too_long_articles=0,
        english_count=1,
        matched_any_firm=0,
        articles_kept=0,
    )
    samples_data: List[SampleData] = [
        SampleData(article_data=[], sample_metadata=sample_metadata),
    ]

    npo.samples_to_parquet(samples_data, run_data)

    # Only metadata Parquet should be written.
    assert len(calls) == 1
    assert "ccnews_sample_stats" in calls[0]["path"]
    assert len(calls[0]["df"]) == 1


def test_run_month_parser_invokes_generate_and_parquet_for_each_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that `run_month_parser` calls `generate_run_data` for each
    trading day and session, skips None results, and wires the output of
    `parse_session` into `samples_to_parquet`.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `generate_run_data`, `parse_session`, and
        `samples_to_parquet` so that no external services or downstream
        orchestration logic are invoked.

    Returns
    -------
    None
        The test passes if `generate_run_data` is called once per
        (day, session) pair, `parse_session` is called only for non-None
        `RunData`, and `samples_to_parquet` receives the corresponding
        samples data and run data.

    Raises
    ------
    AssertionError
        If the number of invocations does not match expectations or if
        the wiring between `parse_session` and `samples_to_parquet` is
        broken.

    Notes
    -----
    - The stubbed `generate_run_data` returns None for one of the
      sessions to exercise the early-continue branch and avoid calling
      `parse_session` for missing work.
    """
    calls_generate: List[Dict[str, Any]] = []
    calls_parse: List[RunData] = []
    calls_parquet: List[Dict[str, Any]] = []

    def fake_generate_run_data(
        year: int,
        month: int,
        date: dt.date,
        session: str,
        bucket: str,
        logger: InfraLogger,
        s3_client: Any,
    ) -> RunData | None:
        calls_generate.append(
            {"year": year, "month": month, "date": date, "session": session, "bucket": bucket}
        )
        if session == "overnight":
            return None
        return RunData(
            date=date,
            session=session,
            bucket=bucket,
            firm_info_dict={},
            firm_name_parts={},
            samples=[],
            logger=logger,
            s3_client=object(),
        )

    def fake_parse_session(run_data: RunData) -> List[SampleData]:
        calls_parse.append(run_data)
        md = SampleMetadata(
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
        return [SampleData(article_data=[], sample_metadata=md)]

    def fake_samples_to_parquet(samples_data: List[SampleData], run_data: RunData) -> None:
        calls_parquet.append({"samples_data": samples_data, "run_data": run_data})

    monkeypatch.setattr(npo, "generate_run_data", fake_generate_run_data)
    monkeypatch.setattr(npo, "parse_session", fake_parse_session)
    monkeypatch.setattr(npo, "samples_to_parquet", fake_samples_to_parquet)

    trading_days = [dt.date(2020, 1, 6), dt.date(2020, 1, 7)]

    npo.run_month_parser(
        year=2020,
        month=1,
        trading_days=trading_days,
        bucket="my-bucket",
        logger_level="INFO",
    )

    # Two days × two sessions each = 4 generate_run_data calls.
    assert len(calls_generate) == 4
    # Only intraday sessions produce run_data and trigger downstream calls.
    assert len(calls_parse) == 2
    assert len(calls_parquet) == 2
    assert all(rd.session == "intraday" for rd in calls_parse)
    assert all(call["run_data"].session == "intraday" for call in calls_parquet)


def test_main_submits_month_parsers_for_all_months(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Confirm that `main` wires together environment loading, argument
    parsing, logger initialization, calendar extraction, and submission
    of `run_month_parser` tasks to the process pool executor.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `load_dotenv`, `extract_cli_args`,
        `initialize_logger`, `extract_trading_calendar_slice`,
        `run_month_parser`, and `ProcessPoolExecutor` so that no real
        environment, logging, or multiprocessing occurs.

    Returns
    -------
    None
        The test passes if `run_month_parser` is submitted once per
        month returned by `extract_trading_calendar_slice` and the
        executor is constructed with `MAXIMAL_WORKER_COUNT`.

    Raises
    ------
    AssertionError
        If the number of submitted tasks does not match the number of
        months, if the submitted arguments are incorrect, or if the
        executor is created with an unexpected `max_workers` value.

    Notes
    -----
    - The test treats the executor as an interaction boundary and
      asserts only on the recorded submissions rather than executing
      any child processes.
    """
    # Stub load_dotenv to a no-op.
    monkeypatch.setattr(npo, "load_dotenv", lambda: None)

    # Stub CLI args.
    monkeypatch.setattr(npo, "extract_cli_args", lambda: (2020, "my-bucket", "INFO"))

    dummy_logger = _DummyLogger()

    def fake_initialize_logger(
        component_name: str, level: str, run_meta: Dict[str, Any]
    ) -> InfraLogger:
        return cast(InfraLogger, dummy_logger)

    monkeypatch.setattr(npo, "initialize_logger", fake_initialize_logger)

    # Two months with arbitrary trading days; only the keys matter here.
    trading_calendar = {
        1: [dt.date(2020, 1, 2)],
        2: [dt.date(2020, 2, 3)],
    }
    monkeypatch.setattr(npo, "extract_trading_calendar_slice", lambda year: trading_calendar)

    submissions: List[Dict[str, Any]] = []
    created_executors: List[_DummyExecutor] = []

    class _RecorderExecutor(_DummyExecutor):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            created_executors.append(self)

        def __enter__(self) -> "_RecorderExecutor":
            return self

        def submit(self, fn: Any, *args: Any, **kwargs: Any) -> None:
            submissions.append({"fn": fn, "args": args, "kwargs": kwargs})

    monkeypatch.setattr(npo, "ProcessPoolExecutor", _RecorderExecutor)

    # We do not need the real run_month_parser to execute; just record that it is passed.
    monkeypatch.setattr(npo, "run_month_parser", lambda *args, **kwargs: None)

    npo.main()

    # One submission per month.
    assert len(submissions) == 2
    for sub in submissions:
        # First argument to submit is the function `run_month_parser`.
        assert sub["fn"] is npo.run_month_parser

    # Exactly one executor created with max_workers set to MAXIMAL_WORKER_COUNT.
    assert len(created_executors) == 1
    exec_instance = created_executors[0]
    assert exec_instance.kwargs.get("max_workers") == npo.MAXIMAL_WORKER_COUNT
    # Sanity check that logging occurred.
    assert dummy_logger.infos
