"""
Purpose
-------
Coordinate end-to-end uniform sampling of CC-NEWS WARC queues stored in S3 into
per-day, per-session (intraday/overnight) sample files using the NYSE calendar.

Key behaviors
-------------
- Parses CLI arguments to obtain the target S3 bucket and daily sampling cap.
- Discovers monthly WARC queue objects in S3 and groups them by (year, month).
- Iterates months in chronological order, seeding a deterministic RNG per month
  and delegating sampling to `extract_sample(...)`.
- Threads cross-month overnight spillover via `spillover_in` / `spillover_out`
  on `RunData` so end-of-month overnight news is attributed to the next month’s
  trading day.
- Writes sampled WARC paths back to S3 under a structured
  `<year>/<month>/<day>/<session>/samples.txt` layout.

Conventions
-----------
- S3 keys for monthly queues are grouped by `"<year>/<month>/..."` with
  zero-padded components (e.g., `"2019/03/..."`).
- Months are processed in sorted `(year, month)` order to ensure spillover from
  one month is available to the next.
- Session buckets are `"intraday"` and `"overnight"`; downstream code assumes
  these exact string keys.
- Logging uses `InfraLogger` with structured context for high-level summaries.

Downstream usage
----------------
- Invoke this module as a CLI entry point (`python -m ...uniform_sampling`) to
  generate CC-NEWS samples into S3.
- Programmatic consumers may call `run_month_loop(...)` directly to obtain
  in-memory samples, or `write_samples_to_s3(...)` to persist existing sample
  dictionaries.
"""

import hashlib
import posixpath
import sys
from typing import Any, List

import boto3
import numpy as np
import pandas as pd

from aws.ccnews_sampler.calendar_utils import extract_nyse_cal
from aws.ccnews_sampler.ccnews_sampler_types import SamplingDict, YearMonth
from aws.ccnews_sampler.extract_sample import extract_sample
from aws.ccnews_sampler.run_data import RunData
from infra.logging.infra_logger import InfraLogger, initialize_logger


def main() -> None:
    """
    Entry point for the CC-NEWS uniform sampling pipeline.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If required CLI arguments are missing or invalid, via `sys.argv` usage or
        downstream failures.

    Notes
    -----
    - Initializes the structured logger, parses CLI arguments, discovers monthly
      WARC queues, runs sampling for each month, and writes results back to S3.
    - Intended to be invoked as the module's `__main__` entry point, not called
      directly from other code.
    """

    bucket, daily_cap, logger_level = extract_cli_args()
    logger: InfraLogger = initialize_logger(
        component_name="ccnews_sampler.uniform_sampling",
        level=logger_level,
        run_meta={"bucket": bucket, "daily_cap": daily_cap},
    )
    warc_path_dict: dict[YearMonth, str] = extract_warc_path_dict(bucket)
    samples_dict: SamplingDict = run_month_loop(
        logger,
        bucket,
        daily_cap,
        warc_path_dict,
    )
    write_samples_to_s3(bucket, logger, samples_dict)


def extract_cli_args() -> tuple[str, int, str]:
    """
    Parse and return the bucket name and daily sampling cap from CLI arguments.

    Parameters
    ----------
    None

    Returns
    -------
    tuple[str, int, str]
        A `(bucket, daily_cap)` tuple where:
        - `bucket` is the S3 bucket name.
        - `daily_cap` is the per-day total sample cap as an integer.
        - `logger_level` is the desired logging level as a string (set to INFO if None).

    Raises
    ------
    IndexError
        If the expected CLI arguments are not present in `sys.argv`.
    ValueError
        If the second argument cannot be converted to an integer.

    Notes
    -----
    - Expects `sys.argv[1]` to be the S3 bucket name and `sys.argv[2]` to be an
      integer daily cap.
    - This helper is intentionally minimal; validation of bucket existence or
      cap reasonableness is handled downstream.
    """

    bucket: str = sys.argv[1]
    daily_cap: int = int(sys.argv[2])
    logger_level: str = "INFO"
    if len(sys.argv) == 4:
        logger_level = sys.argv[3]
    return bucket, daily_cap, logger_level


def extract_warc_path_dict(bucket: str) -> dict[YearMonth, str]:
    """
    Discover monthly WARC queue objects in S3 and group them by (year, month).

    Parameters
    ----------
    bucket : str
        Name of the S3 bucket containing CC-NEWS monthly WARC queue objects.

    Returns
    -------
    dict[tuple[str, str], str]
        A mapping `(year, month) -> key` where:
        - `year` is a four-digit string (e.g., "2019"),
        - `month` is a two-digit string (e.g., "03"),
        - `key` is the S3 object key for that month's WARC queue.

    Raises
    ------
    botocore.exceptions.BotoCoreError
        If S3 communication fails.
    botocore.exceptions.ClientError
        If the bucket does not exist or permissions are insufficient.

    Notes
    -----
    - Uses `list_objects_v2` with pagination to walk the entire bucket.
    - Assumes keys are structured as `"<year>/<month>/..."` and uses the first two
      path segments to derive `(year, month)`.
    - If multiple objects share the same `(year, month)` prefix, the last one seen
      in the listing wins; bucket naming conventions should avoid ambiguity.
    """

    s3_client: Any = boto3.client("s3")
    warc_path_dict: dict[YearMonth, str] = {}
    should_continue: bool = True
    continuation_token: str = ""
    while should_continue:
        kwargs: dict[str, Any] = {
            "Bucket": bucket,
        }
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        response: dict = s3_client.list_objects_v2(**kwargs)
        for obj in response.get("Contents", []):
            key: str = obj["Key"]
            split_key: List[str] = key.split("/")
            if split_key[-1] != "warc_queue.txt":
                continue
            year: str = split_key[0]
            month: str = split_key[1]
            warc_path_dict[(year, month)] = key
        if not response.get("IsTruncated"):
            should_continue = False
        continuation_token = response.get("NextContinuationToken", "")
    return warc_path_dict


def run_month_loop(
    logger: InfraLogger,
    bucket: str,
    daily_cap: int,
    warc_path_dict: dict[YearMonth, str],
) -> SamplingDict:
    """
    Run the sampling pipeline for each discovered month, handling cross-month spillover.

    Parameters
    ----------
    logger : InfraLogger
        Structured logger used for emitting run-level events and diagnostics.
    bucket : str
        Name of the S3 bucket containing monthly WARC queues and receiving samples.
    daily_cap : int
        Target total number of samples per civil day (intraday + overnight).
    warc_path_dict : dict[tuple[str, str], str]
        Mapping `(year, month) -> WARC queue key` as produced by `extract_warc_path_dict`.

    Returns
    -------
    dict[tuple[str, str], SampleForMonth]
        A nested mapping:
        `{(year, month): {date_str: {"intraday": [...], "overnight": [...]}, ...}}`
        containing the sampled WARC paths for each month, day, and session.

    Raises
    ------
    Exception
        Propagates errors from calendar extraction, sampling, or S3 access.

    Notes
    -----
    - Processes months in sorted `(year, month)` order to preserve chronological
      consistency and enable cross-month overnight spillover.
    - For each month, constructs a `RunData` instance with a deterministic RNG seed
      derived from `year + month` and passes it to `extract_sample(...)`.
    - Threads cross-month overnight candidates via `spillover_in` and `spillover_out`
      on `RunData`, so that late-session items after the monthly queue end can be
      attributed to the next month’s trading day.
    """

    sampling_dict: SamplingDict = {}
    spill_over_candidates: dict[str, List[str]] = {}
    for (year, month), warc_path in sorted(warc_path_dict.items()):
        hash_func = hashlib.new("sha256")
        run_seed: str = year + month
        hash_func.update(run_seed.encode("utf-8"))
        rng: np.random.Generator = np.random.default_rng(int(hash_func.hexdigest(), 16) % (2**64))
        nyse_cal: pd.DataFrame = extract_nyse_cal(year, month)
        run_data: RunData = RunData(
            bucket=bucket,
            key=warc_path,
            year=year,
            month=month,
            daily_cap=daily_cap,
            nyse_cal=nyse_cal,
            logger=logger,
            rng=rng,
            spillover_in=spill_over_candidates,
            spillover_out={},
        )
        sampling_dict[(year, month)] = extract_sample(run_data)
        spill_over_candidates = run_data.spillover_out
    return sampling_dict


def write_samples_to_s3(
    bucket: str,
    logger: InfraLogger,
    sampling_dict: SamplingDict,
) -> None:
    """
    Persist sampled WARC links to S3 under per-day, per-session text files.

    Parameters
    ----------
    bucket : str
        Name of the S3 bucket to which sample files will be written.
    logger : InfraLogger
        Logger used to emit high-level information about the writing process.
    sampling_dict : dict[tuple[str, str], SampleForMonth]
        Nested mapping from `(year, month)` to per-day/session samples, as
        produced by `run_month_loop(...)`.

    Returns
    -------
    None

    Raises
    ------
    botocore.exceptions.BotoCoreError
        If any S3 interaction fails at the client level.
    botocore.exceptions.ClientError
        If S3 rejects a `put_object` call (e.g., due to permissions).

    Notes
    -----
    - For each `(year, month)` entry, iterates over all `date_str` keys and writes
      one file per session ("intraday" and "overnight") with keys of the form:
        `<year>/<month>/<day>/<session>/samples.txt`.
    - Each file body is a UTF-8 encoded, newline-separated list of sampled WARC
      paths for that session.
    - Logging includes the number of months processed and per-month debug entries
      so the caller can trace which shards were written.
    """

    logger.info("Writing samples to S3", context={"amount_of_samples": len(sampling_dict)})
    for (year, month), month_sample_dict in sampling_dict.items():
        logger.debug(
            "Writing samples for year/month",
            context={"year": year, "month": month},
        )
        s3_client: Any = boto3.client("s3")
        output_prefix: str = posixpath.join(year, month)
        for date_str, samples in month_sample_dict.items():
            current_day: str = date_str.split("-")[-1]
            for session in ["intraday", "overnight"]:
                fill_session_dir(
                    s3_client, session, output_prefix, current_day, bucket, samples[session]
                )


def fill_session_dir(
    s3_client: Any,
    session: str,
    output_prefix: str,
    current_day: str,
    bucket: str,
    samples: List[str],
) -> None:
    """
    Write a single session's sampled WARC links for one day to S3.

    Parameters
    ----------
    s3_client : Any
        Boto3 S3 client used to perform the `put_object` call.
    session : str
        Session name, typically `"intraday"` or `"overnight"`.
    output_prefix : str
        Base S3 key prefix for the current month (e.g., `"2019/03"`).
    current_day : str
        Two-digit day of month (e.g., `"01"`), used to create the day subdirectory.
    bucket : str
        S3 bucket name where the object will be written.
    samples : list[str]
        List of sampled WARC paths to serialize into the session file.

    Returns
    -------
    None

    Raises
    ------
    botocore.exceptions.BotoCoreError
        If the S3 client encounters a low-level error.
    botocore.exceptions.ClientError
        If S3 rejects the `put_object` request.

    Notes
    -----
    - Constructs an S3 object key of the form:
      `"{output_prefix}/{current_day}/{session}/samples.txt"`.
    - Writes the `samples` list as a newline-separated UTF-8 encoded text file
      with ContentType `"text/plain; charset=utf-8"`.
    - Does not perform any validation on `session`; callers are responsible for
      providing canonical session names.
    """

    output_path: str = posixpath.join(output_prefix, f"{current_day}/{session}/samples.txt")
    s3_client.put_object(
        Bucket=bucket,
        Key=output_path,
        Body="\n".join(samples).encode("utf-8"),
        ContentType="text/plain; charset=utf-8",
    )


if __name__ == "__main__":
    main()
