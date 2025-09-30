"""
Purpose
-------
Orchestrate a single-month sampling run:
1) seed a deterministic RNG from (year, month),
2) load the NYSE calendar,
3) stream the monthly WARC queue and sample per day/session,
4) write results back to S3 as text files under the month prefix.

Key behaviors
-------------
- Deterministic RNG per (year, month) for reproducible sampling.
- Single S3 pass read of the monthly manifest; bounded-memory sampling.
- Writes per-day, per-session outputs to S3 in a stable folder layout.

Conventions
-----------
- Input: (bucket, key) where `key` is like 'YYYY/MM/warc_queue.txt'.
- Output prefix for the run: posix dirname of `key` → 'YYYY/MM'.
- Output layout per day/session:
    YYYY/MM/<DD>/intraday/samples.txt
    YYYY/MM/<DD>/overnight/samples.txt

Downstream usage
----------------
This module is launched by the shell wrapper. The main entry point is `main()`,
which returns no value and writes logs to STDOUT and samples to S3.
"""

import hashlib
import json
import os
import posixpath
import sys
from typing import Any, List, TypeAlias, TypedDict

import boto3
import numpy as np
import pandas as pd

from aws.ccnews_sampler.calendar_utils import extract_nyse_cal
from aws.ccnews_sampler.extract_sample import extract_sample
from aws.ccnews_sampler.reservoir_sampling import OverIntraSamples
from aws.ccnews_sampler.run_data import RunData
from aws.ccnews_sampler.run_logger import RunLogger

CLIArguments: TypeAlias = tuple[str, str, str, str, int]
ExportedVariables: TypeAlias = tuple[str, str, dict]


class FinalLogData(TypedDict):
    """
    Purpose
    -------
    Structured dictionary representing the summary payload for a monthly
    sampling run. Exists to enforce key presence and type safety for log
    emission and downstream consumers.

    Key behaviors
    -------------
    - Provides fixed keys for year/month identifiers, output path, and counters.
    - Starts counters at zero; values are incremented during the run.

    Parameters
    ----------
    year : str
        Four-digit year token (e.g., "2024").
    month : str
        Two-digit month token (e.g., "09").
    output_prefix : str
        S3 prefix under which sample files will be written (e.g., "2019/03").
    days_processed : int
        Number of trading days processed; initialized to 0 and incremented
        by the writer.
    files_written : int
        Number of sample files written; initialized to 0 and incremented
        by the writer.

    Attributes
    ----------
    year : str
        Four-digit year string, immutable once set.
    month : str
        Two-digit month string, immutable once set.
    output_prefix : str
        Target S3 prefix for run output, constant throughout the run.
    days_processed : int
        Mutable counter tracking processed trading days.
    files_written : int
        Mutable counter tracking number of output files written.

    Notes
    -----
    - `FinalLogData` is intended for logging/serialization only and does not
      provide methods or validation.
    - TypedDict is used instead of a dataclass to remain lightweight and
      JSON-serializable with minimal overhead.
    """

    year: str
    month: str
    output_prefix: str
    days_processed: int
    files_written: int


def main() -> None:
    """
    Entry point for a single-month run.

    Steps
    -----
    1. Initialize structured logger from exported env (RUN_ID, SHARD_NAME, RUN_META_JSON).
    2. Parse CLI args: (bucket, key, year, month, daily_cap).
    3. Seed a deterministic RNG from (year, month).
    4. Load the NYSE calendar for that month.
    5. Build a `RunData` container and call `extract_sample(...)`.
    6. Persist the sampled links to S3 via `write_samples_to_s3(...)`.

    Returns
    -------
    None

    Notes
    -----
    - Determinism: identical (year, month) yields the same RNG seed and sampling outcome
      given the same input manifest and code version.
    """
    # Initialize event logger
    logger: RunLogger = init_logger()

    # Extract CLI args
    bucket, key, year, month, daily_cap = extract_cli_args()

    # Initialize random seed for reproducibility
    hash_func = hashlib.new("sha256")
    run_seed: str = year + month
    hash_func.update(run_seed.encode("utf-8"))
    rng: np.random.Generator = np.random.default_rng(int(hash_func.hexdigest(), 16) % (2**64))

    # Extract NYSE trading calendar for month
    nyse_cal: pd.DataFrame = extract_nyse_cal(year, month)

    run_data: RunData = RunData(
        bucket=bucket,
        key=key,
        year=year,
        month=month,
        daily_cap=daily_cap,
        nyse_cal=nyse_cal,
        logger=logger,
        rng=rng,
    )

    sampling_dict: dict[str, OverIntraSamples] = extract_sample(run_data)
    write_samples_to_s3(sampling_dict, run_data)


def init_logger() -> RunLogger:
    """
    Construct a run-scoped structured logger and emit a start marker.

    Returns
    -------
    RunLogger
        Logger with (run_id, shard_name, run_meta) baked into every event.

    Notes
    -----
    - Reads exported vars from the parent shell script for correlation across processes.
    """
    run_id, shard_name, run_meta = extract_exported_vars()
    logger = RunLogger(run_id, shard_name, run_meta)
    logger.emit("run_started", "INFO")
    return logger


def extract_exported_vars() -> ExportedVariables:
    """
    Load run-scoped metadata exported by the launcher.

    Environment
    -----------
    RUN_ID : str
        Unique id for the whole run (e.g., uuid4 from the shell script).
    SHARD_NAME : str
        Logical label for the shard (defaults to YEAR).
    RUN_META_JSON : str
        JSON payload with stable config (e.g., year, caps, output_prefix).

    Returns
    -------
    tuple[str, str, dict]
        (run_id, shard_name, run_meta) with run_meta parsed from JSON.
    """
    run_id: str = os.environ["RUN_ID"]
    shard_name: str = os.environ["SHARD_NAME"]
    run_meta: dict = json.loads(os.environ["RUN_META_JSON"])
    return run_id, shard_name, run_meta


def extract_cli_args() -> CLIArguments:
    """
    Parse positional CLI arguments provided by the launcher.

    Returns
    -------
    tuple[str, str, str, str, int]
        (bucket, key, year, month, daily_cap).

    Notes
    -----
    - Assumes the wrapper always provides all five arguments in order.
    """
    bucket: str = sys.argv[1]
    key: str = sys.argv[2]
    year: str = sys.argv[3]
    month: str = sys.argv[4]
    daily_cap: int = int(sys.argv[5])
    return bucket, key, year, month, daily_cap


def write_samples_to_s3(
    sampling_dict: dict[str, OverIntraSamples],
    run_data: RunData,
) -> None:
    """
    Persist sampled links to S3 under the month prefix derived from `run_data.key`.

    Parameters
    ----------
    sampling_dict : dict[str, OverIntraSamples]
        Mapping:
            {
            "YYYY-MM-DD": {
                "intraday":  [link, ...],
                "overnight": [link, ...]
            },
            ...
            }
    run_data : RunData
        Provides destination `bucket`, input `key` (whose dirname becomes the month
        output prefix), and the run-scoped logger.

    Returns
    -------
    None

    Notes
    -----
    - Output keys written:
        <dirname(run_data.key)>/<DD>/intraday/samples.txt
        <dirname(run_data.key)>/<DD>/overnight/samples.txt
    - S3 prefixes are virtual; `PutObject` is sufficient to “create” folders.
    - Emits a final "samples_written" event with counts via `run_data.logger`.
    """
    bucket: str = run_data.bucket
    key: str = run_data.key
    s3_client: Any = boto3.client("s3")
    output_prefix: str = posixpath.dirname(key)
    final_log_dict: FinalLogData = create_final_log_dict(
        run_data,
        output_prefix,
    )
    for date_str, samples in sampling_dict.items():
        final_log_dict["days_processed"] += 1
        current_day: str = date_str.split("-")[-1]
        for session in ["intraday", "overnight"]:
            final_log_dict["files_written"] += 1
            fill_session_dir(
                s3_client, session, output_prefix, current_day, bucket, samples[session]
            )
    run_data.logger.samples_emitted(final_log_dict)


def create_final_log_dict(
    run_data: RunData,
    output_prefix: str,
) -> FinalLogData:
    """
    Build the initial payload for the final samples summary log.

    Parameters
    ----------
    run_data : RunData
        Execution context (year, month, logger, etc.).
    output_prefix : str
        S3 prefix under which this month’s sample files will be written
        (e.g., '2019/03').

    Returns
    -------
    FinalLogData
        Dict with shape:
        {
            "year": "<YYYY>",
            "month": "<MM>",
            "output_prefix": "<YYYY/MM>",
            "days_processed": 0,
            "files_written": 0
        }

    Notes
    -----
    - Counters start at zero and are incremented by the writer.
    """
    return {
        "year": run_data.year,
        "month": run_data.month,
        "output_prefix": output_prefix,
        "days_processed": 0,
        "files_written": 0,
    }


def fill_session_dir(
    s3_client: Any,
    session: str,
    output_prefix: str,
    current_day: str,
    bucket: str,
    samples: List[str],
) -> None:
    """
    Write one session’s sample file for a given day.

    Parameters
    ----------
    s3_client :
        Boto3 S3 client.
    session : str
        "intraday" or "overnight".
    output_prefix : str
        Month prefix (dirname of the queue key), e.g. 'YYYY/MM'.
    current_day : str
        Two-digit day of month, e.g. '05'.
    bucket : str
        Destination S3 bucket.
    samples : List[str]
        Collected links for the given day/session.

    Returns
    -------
    None

    Notes
    -----
    - Writes to `<output_prefix>/<current_day>/<session>/samples.txt`.
    - Content is UTF-8 text, one link per line.
    """
    output_path: str = posixpath.join(output_prefix, f"{current_day}/{session}/samples.txt")
    s3_client.put_object(
        Bucket=bucket,
        Key=output_path,
        Body="\n".join(samples).encode("utf-8"),
        ContentType="text/plain",
        charset="utf-8",
    )


if __name__ == "__main__":
    main()
