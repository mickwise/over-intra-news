"""
Purpose
-------
Unit tests for `aws.ccnews_sampler.uniform_sampling`.

Key behaviors
-------------
- Verify that `extract_cli_args` parses the S3 bucket name and daily cap from
  `sys.argv`.
- Validate that `extract_warc_path_dict` walks S3 listings and groups objects by
  `(year, month)`.
- Exercise `run_month_loop`, including chronological ordering of months and
  cross-month spillover threading.
- Check that `fill_session_dir` and `write_samples_to_s3` build the expected S3
  keys and payloads and delegate correctly.

Conventions
-----------
- All external dependencies (S3, NYSE calendar, `RunData`) are stubbed via
  `pytest.MonkeyPatch` to avoid real network or database I/O.
- Dummy logger and RunData stand-ins (_DummyLogger, _DummyRunData) capture the
  minimal behavior needed by the code under test.
- Sampling structures use the same `(year, month)` and per-session conventions
  as the production code.

Downstream usage
----------------
- Run via `pytest` as part of the CI suite.
- Serves as executable documentation for the expected orchestration behavior of
  the uniform sampling pipeline without touching live AWS resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd
import pytest

from aws.ccnews_sampler import uniform_sampling
from infra.logging.infra_logger import InfraLogger


@dataclass
class _DummyLogger:
    """
    Purpose
    -------
    Minimal logger stub that records `.info(...)` and `.debug(...)` calls.

    Key behaviors
    -------------
    - Appends each info/debug message and its context dictionary to an internal
      list so tests can assert on log usage.
    - Mirrors the subset of the `InfraLogger` interface that
      `uniform_sampling` relies on.

    Parameters
    ----------
    infos : list[tuple[str, dict[str, Any]]] or None
        Optional initial list of info log entries; normally left as None and
        initialized to an empty list in `__post_init__`.
    debugs : list[tuple[str, dict[str, Any]]] or None
        Optional initial list of debug log entries; normally left as None and
        initialized to an empty list in `__post_init__`.

    Attributes
    ----------
    infos : list[tuple[str, dict[str, Any]]]
        Collected `(message, context)` pairs for `.info(...)` calls.
    debugs : list[tuple[str, dict[str, Any]]]
        Collected `(message, context)` pairs for `.debug(...)` calls.

    Notes
    -----
    - This stub deliberately ignores log levels, formatting, and sinks; tests
      only care that log calls happen with the expected context.
    - The type annotations use `Any` for context values to match the production
      logger’s flexibility.
    """

    infos: List[Tuple[str, Dict[str, Any]]] = None  # type: ignore[assignment]
    debugs: List[Tuple[str, Dict[str, Any]]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.infos = []
        self.debugs = []

    def info(self, msg: str, context: Dict[str, Any] | None = None) -> None:
        self.infos.append((msg, context or {}))

    def debug(self, msg: str, context: Dict[str, Any] | None = None) -> None:
        self.debugs.append((msg, context or {}))


@dataclass
class _DummyRunData:
    bucket: str
    key: str
    year: str
    month: str
    daily_cap: int
    nyse_cal: pd.DataFrame
    logger: Any
    rng: np.random.Generator
    spillover_in: Dict[str, List[str]]
    spillover_out: Dict[str, List[str]]


@pytest.mark.parametrize(
    ["logger_level", "expected_logger_level"], [("DEBUG", "DEBUG"), (None, "INFO")]
)
def test_extract_cli_args_parses_bucket_and_cap(
    monkeypatch: pytest.MonkeyPatch,
    logger_level: str | None,
    expected_logger_level: str,
) -> None:
    """
    Verify that `extract_cli_args` reads bucket and daily cap from `sys.argv`.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace `uniform_sampling.sys` with a controlled
        argv namespace.
    logger_level : str | None
        Optional logger level to inject into `argv`; if `None`, only bucket and
        cap are provided.
    expected_logger_level : str
        Expected logger level returned by `extract_cli_args`, based on the
        injected `logger_level`.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the parsed bucket or cap do not match the injected `argv` values
        or if the logger level does not match expectations.


    Notes
    -----
    - This test only exercises argument parsing; validation of the bucket or
      cap value is performed elsewhere in the pipeline.
    """

    monkeypatch.setattr(
        uniform_sampling,
        "sys",
        SimpleNamespace(
            argv=(
                ["prog", "my-bucket", "25", logger_level]
                if logger_level
                else ["prog", "my-bucket", "25"]
            )
        ),
    )

    bucket, daily_cap, logger_level_result = uniform_sampling.extract_cli_args()

    assert bucket == "my-bucket"
    assert daily_cap == 25
    assert logger_level_result == expected_logger_level


def test_extract_warc_path_dict_lists_objects_and_groups_by_year_month(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure `extract_warc_path_dict` paginates S3 listings and groups keys by
    `(year, month)`.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to stub `boto3.client` with a dummy S3 client that exposes
        controlled `list_objects_v2` pages.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If pagination is not respected or the resulting mapping does not contain
        the expected `(year, month) -> key` pair. Or if pairs with ending different
        from "warc_queue.txt" are included in the result.

    Notes
    -----
    - The dummy client simulates two pages of results with three keys spread
      across January–March 2019; the test asserts that all are present in the
      final dictionary.
    """

    class DummyS3Client:
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []
            self._page = 0

        def list_objects_v2(self, **kwargs: Any) -> Dict[str, Any]:
            self.calls.append(kwargs)
            # First page: truncated, two objects.
            if self._page == 0:
                self._page += 1
                return {
                    "Contents": [
                        {"Key": "2019/01/warc_queue.txt"},
                        {"Key": "2019/02/other.txt"},
                    ],
                    "IsTruncated": True,
                    "NextContinuationToken": "tok-1",
                }
            # Second (final) page: one more object.
            return {
                "Contents": [
                    {"Key": "2019/03/foo.txt"},
                ],
                "IsTruncated": False,
            }

    class DummyBoto3:
        @staticmethod
        def client(_service: str) -> DummyS3Client:
            return DummyS3Client()

    monkeypatch.setattr(uniform_sampling, "boto3", DummyBoto3)

    result = uniform_sampling.extract_warc_path_dict("ignored-bucket")

    # We should have entries only for the first object in each (year, month).
    assert result == {
        ("2019", "01"): "2019/01/warc_queue.txt",
    }


def test_run_month_loop_orders_months_and_threads_spillover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Check that `run_month_loop` processes months in sorted order and threads
    spillover from one month into the next.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to stub `extract_nyse_cal`, `RunData`, and
        `extract_sample` so the test can observe spillover behavior without
        hitting real dependencies.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If months are not processed in chronological `(year, month)` order or
        if `spillover_out` from one month is not fed into `spillover_in` for
        the next month.

    Notes
    -----
    - The stubbed `extract_sample`:
        * records `spillover_in` for each month,
        * populates `spillover_out` with a synthetic entry, and
        * returns a minimal per-day/session sample dict.
    - The test asserts both the processing order and the observed spillover
      sequence.
    """

    # Stub extract_nyse_cal to return a trivial calendar frame.
    def fake_extract_nyse_cal(_year: str, _month: str) -> pd.DataFrame:
        return pd.DataFrame(index=pd.to_datetime(["2024-01-01"]))

    monkeypatch.setattr(uniform_sampling, "extract_nyse_cal", fake_extract_nyse_cal)

    # Replace RunData with our dummy that just holds the arguments.
    monkeypatch.setattr(uniform_sampling, "RunData", _DummyRunData)

    # Stub extract_sample to:
    # - record spillover_in seen per call
    # - set spillover_out so the next month sees it
    # - return a small sample dict
    spillovers_seen: List[Dict[str, List[str]]] = []

    def fake_extract_sample(run_data: _DummyRunData) -> Dict[str, Dict[str, List[str]]]:
        spillovers_seen.append(dict(run_data.spillover_in))
        # Set spillover_out so the next month can see something non-empty.
        run_data.spillover_out = {f"{run_data.year}-{run_data.month}": ["spill"]}
        # Return a trivial per-day/session mapping.
        return {
            "2024-01-01": {
                "intraday": [f"{run_data.year}{run_data.month}-i"],
                "overnight": [f"{run_data.year}{run_data.month}-o"],
            }
        }

    monkeypatch.setattr(uniform_sampling, "extract_sample", fake_extract_sample)

    logger = cast(InfraLogger, _DummyLogger())

    # Intentionally unsorted keys to prove sorting happens.
    warc_path_dict: Dict[Tuple[str, str], str] = {
        ("2019", "02"): "2019/02/warc_queue.txt",
        ("2019", "01"): "2019/01/warc_queue.txt",
    }

    result = uniform_sampling.run_month_loop(
        logger=logger,
        bucket="sample-bucket",
        daily_cap=10,
        warc_path_dict=warc_path_dict,
    )

    # Months should be processed in sorted order.
    assert list(result.keys()) == [("2019", "01"), ("2019", "02")]

    # First month sees empty spillover; second month sees output from first.
    assert spillovers_seen[0] == {}
    assert spillovers_seen[1] == {"2019-01": ["spill"]}

    # Structure of the returned sampling dict should match what fake_extract_sample returns.
    first_month_samples = result[("2019", "01")]
    assert "2024-01-01" in first_month_samples
    assert set(first_month_samples["2024-01-01"].keys()) == {"intraday", "overnight"}


def test_fill_session_dir_puts_expected_object() -> None:
    """
    Ensure `fill_session_dir` issues a `put_object` call with the expected
    bucket, key, body, and content type.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the constructed key does not match the expected
        `"2019/01/02/intraday/samples.txt"` path or if the body/content type
        are incorrect.

    Notes
    -----
    - Uses a simple recording S3 client that captures kwargs passed to
      `put_object` so the test can assert on them directly.
    """

    class RecordingS3Client:
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        def put_object(self, **kwargs: Any) -> None:
            self.calls.append(kwargs)

    client = RecordingS3Client()

    uniform_sampling.fill_session_dir(
        s3_client=client,
        session="intraday",
        output_prefix="2019/01",
        current_day="02",
        bucket="my-bucket",
        samples=["a", "b"],
    )

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["Bucket"] == "my-bucket"
    assert call["Key"] == "2019/01/02/intraday/samples.txt"
    assert call["Body"] == b"a\nb"
    assert call["ContentType"].startswith("text/plain")


def test_write_samples_to_s3_delegates_to_fill_session_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that `write_samples_to_s3` iterates the sampling dictionary and
    delegates per-session writes to `fill_session_dir`.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace `fill_session_dir` with a stub that records
        calls without touching real S3.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If `fill_session_dir` is not called once per session or if the threaded
        parameters (day, bucket, samples) do not match expectations.

    Notes
    -----
    - Constructs a minimal `sampling_dict` with a single `(year, month, day)`
      and one sample in each session.
    - Asserts that there are exactly two calls (intraday and overnight) and
      that both share the correct `current_day` and `bucket`.
    """

    logger = cast(InfraLogger, _DummyLogger())

    # Single month/day with one intraday and one overnight sample.
    sampling_dict: uniform_sampling.SamplingDict = {
        ("2019", "01"): {
            "2019-01-02": {
                "intraday": ["i1"],
                "overnight": ["o1"],
            }
        }
    }

    calls: List[Dict[str, Any]] = []

    def fake_fill_session_dir(
        _s3_client: Any,
        session: str,
        output_prefix: str,
        current_day: str,
        bucket: str,
        samples: List[str],
    ) -> None:
        calls.append(
            {
                "session": session,
                "output_prefix": output_prefix,
                "current_day": current_day,
                "bucket": bucket,
                "samples": list(samples),
            }
        )

    monkeypatch.setattr(uniform_sampling, "fill_session_dir", fake_fill_session_dir)

    uniform_sampling.write_samples_to_s3(
        bucket="sample-bucket",
        logger=logger,
        sampling_dict=sampling_dict,
    )

    # Expect one call per session.
    assert len(calls) == 2
    sessions = {c["session"] for c in calls}
    assert sessions == {"intraday", "overnight"}

    # Day and bucket should be threaded through correctly.
    for call in calls:
        assert call["current_day"] == "02"
        assert call["bucket"] == "sample-bucket"
        assert call["samples"] in (["i1"], ["o1"])
