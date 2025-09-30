"""
Purpose
-------
Scan a month’s CC-NEWS WARC path queue from S3 and perform streaming,
fixed-capacity reservoir sampling per trading day and session
(intraday/overnight). Emits run context for observability.

Key behaviors
-------------
- Single pass over the monthly queue (O(N) over lines).
- Bounded memory via per-day/session reservoirs (at most per-cap items kept).
- Trading calendar rules (intraday vs. overnight) drive session assignment.

Conventions
-----------
- Timezone: America/New_York for trading-day bucketing.
- Dates keyed as 'YYYY-MM-DD'.
- Session keys: "intraday" and "overnight".
- Input is a monthly queue file at s3://<bucket>/<key> with CC-NEWS WARC paths.

Downstream usage
----------------
- Call `extract_sample(run_data)` to return `{date: {"intraday": [...], "overnight": [...]}}`.
- Logs include totals, matched/unmatched lines, and sample warnings.
"""

import re
from typing import List, cast

import boto3
import pandas as pd

from aws.ccnews_sampler.data_maps import DataMaps, build_data_maps, to_seconds_int
from aws.ccnews_sampler.quota import compute_daily_caps
from aws.ccnews_sampler.reservoir_sampling import OverIntraSamples, ReservoirManager
from aws.ccnews_sampler.run_data import RunData
from aws.ccnews_sampler.run_logger import RunLogger

DATE_TZ = "America/New_York"
DATE_FMT = "%Y-%m-%d"


def extract_sample(run_data: RunData) -> dict[str, OverIntraSamples]:
    """
    Stream the monthly CC-NEWS queue from S3 and produce per-day/session samples.

    Parameters
    ----------
    run_data : RunData
        Execution context: (bucket, key, year, month, daily_cap, nyse_cal, logger, rng).

    Returns
    -------
    dict[str, OverIntraSamples]
        Nested mapping:
        {
          "YYYY-MM-DD": {
            "intraday":  [link, ...],
            "overnight": [link, ...]
          },
          ...
        }

    Notes
    -----
    - Uses `compute_daily_caps` and `build_data_maps` to derive per-day caps and
      session windows.
    - Emits start/finish events and a warning summary if unmatched lines were seen.
    """
    year: str = run_data.year
    month: str = run_data.month
    logger: RunLogger = run_data.logger
    nyse_cal: pd.DataFrame = run_data.nyse_cal

    dt_index: pd.DatetimeIndex = cast(pd.DatetimeIndex, nyse_cal.index)
    str_date_list: List[str] = dt_index.strftime(DATE_FMT).tolist()

    run_context: dict = generate_run_context(str_date_list, year, month)

    logger.initial_emission(run_data)

    nyse_cal = compute_daily_caps(run_data.daily_cap, nyse_cal, run_data.rng)
    data_maps: DataMaps = build_data_maps(nyse_cal)
    reservoir_manager: ReservoirManager = ReservoirManager(data_maps.cap_dict, run_data.rng)

    fill_reservoirs(run_context, run_data, data_maps, reservoir_manager)

    logger.emit("month_scan_finished", "INFO", run_context)

    logger.check_line_count(run_context, year, month)
    return reservoir_manager.extract_sample_dict()


def generate_run_context(days: List[str], year: str, month: str) -> dict:
    """
    Initialize counters and metadata for a month’s scan.

    Parameters
    ----------
    days : list[str]
        Trading-day keys ('YYYY-MM-DD') for the month.
    year : str
        Four-digit year.
    month : str
        Two-digit month.

    Returns
    -------
    dict
        Mutable run context with totals, per-day counters, and example errors.
    """
    return {
        "year": year,
        "month": month,
        "lines_total": 0,
        "lines_matched": 0,
        "lines_unmatched": 0,
        "per_day_intraday_count": {date: 0 for date in days},
        "per_day_overnight_count": {date: 0 for date in days},
        "unknown_or_offmonth_examples": [],
    }


def fill_reservoirs(
    run_context: dict, run_data: RunData, data_maps: DataMaps, reservoir_manager: ReservoirManager
) -> None:
    """
    Read the monthly queue stream and update per-day/session reservoirs.

    Parameters
    ----------
    run_context : dict
        Mutable counters for totals, matched/unmatched, and per-day counts.
    run_data : RunData
        Execution context including (bucket, key).
    data_maps : DataMaps
        Provides per-day caps and session open/close windows.
    reservoir_manager : ReservoirManager
        Holds `{date: {"intraday": Reservoir, "overnight": Reservoir}}` and
        applies reservoir sampling via `sample(...)`.

    Returns
    -------
    None

    Notes
    -----
    - Lines that fail date extraction or fall outside `valid_date_set` are recorded
      as unmatched with up to 5 examples for diagnostics.
    - Admission into reservoirs is always attempted; memory stays bounded per cap.
    """
    s3_client = boto3.client("s3")
    queue_stream = s3_client.get_object(Bucket=run_data.bucket, Key=run_data.key)["Body"]
    date_pattern: re.Pattern[str] = re.compile(
        rf"CC-NEWS-({run_data.year}{run_data.month}\d{{2}})-(\d{{6}})"
    )
    with queue_stream as stream:
        for line in stream.iter_lines():
            line = line.decode("utf-8")
            run_context["lines_total"] += 1
            parsed: tuple[pd.Timestamp, str] | None = extract_link_date(line, date_pattern)
            if not parsed:
                handle_erroneous_line(line, run_context)
                continue
            else:
                utc_date, date_key = parsed
                run_context["lines_matched"] += 1
                if date_key not in data_maps.valid_date_set:
                    handle_erroneous_line(line, run_context)
                    continue
                else:
                    handle_correct_line(
                        line, utc_date, date_key, run_context, data_maps, reservoir_manager
                    )


def extract_link_date(line: str, date_pattern: re.Pattern[str]) -> tuple[pd.Timestamp, str] | None:
    """
    Parse the CC-NEWS timestamp from a WARC path and derive the trading date key.

    Parameters
    ----------
    line : str
        Input line from the monthly queue (a WARC path).
    date_pattern : re.Pattern[str]
        Compiled regex capturing yyyymmdd and hhmmss parts.

    Returns
    -------
    tuple[pd.Timestamp, str] | None
        (UTC timestamp, date key in 'YYYY-MM-DD'), or None if no match.

    Notes
    -----
    - Converts the UTC timestamp to America/New_York to produce the date key.
    """
    regex_result: re.Match[str] | None = re.search(date_pattern, line)
    if regex_result:
        date_groups: tuple[str] = cast(tuple[str], regex_result.groups())
        utc_date: pd.Timestamp = pd.to_datetime(
            "".join(date_groups), format="%Y%m%d%H%M%S", utc=True
        )
        date_key = utc_date.tz_convert(DATE_TZ).strftime(DATE_FMT)
        return utc_date, date_key
    return None


def handle_erroneous_line(
    line: str,
    run_context: dict,
) -> None:
    """
    Record an unmatched or off-month line in the run context.

    Parameters
    ----------
    line : str
        The raw line that failed parsing or validation.
    run_context : dict
        Mutable counters and example buffer.

    Returns
    -------
    None

    Notes
    -----
    - Adds up to 5 examples to `unknown_or_offmonth_examples` for debugging.
    """
    run_context["lines_unmatched"] += 1
    if len(run_context["unknown_or_offmonth_examples"]) < 5:
        run_context["unknown_or_offmonth_examples"].append(line.strip())


def handle_correct_line(
    line: str,
    utc_date: pd.Timestamp,
    date_key: str,
    run_context: dict,
    data_maps: DataMaps,
    reservoir_manager: ReservoirManager,
) -> None:
    """
    Route a valid line to the correct session reservoir and update counters.

    Parameters
    ----------
    line : str
        The WARC path line (kept as-is).
    utc_date : pd.Timestamp
        Extracted UTC timestamp for the item.
    date_key : str
        Trading-day key ('YYYY-MM-DD').
    run_context : dict
        Mutable counters for totals and per-day session counts.
    data_maps : DataMaps
        Provides per-day caps and session open/close windows.
    reservoir_manager : ReservoirManager
        Receives the item via `sample(line, date_key, session)`.

    Returns
    -------
    None

    Notes
    -----
    - Session is determined by whether the UTC timestamp falls between
      session open/close (intraday) or not (overnight), after converting to
      America/New_York for date bucketing.
    - Per-day kept counters are bounded by the caps for observability only; the
      reservoir itself enforces the true capacity.
    """
    intraday_cap, overnight_cap = data_maps.cap_dict[date_key]
    current_session_open, current_session_close = data_maps.session_dict[date_key]
    current_date_seconds = to_seconds_int(utc_date)
    intra_day_condition = (
        current_session_open is not None
        and current_session_close is not None
        and current_session_open <= current_date_seconds < current_session_close
    )
    if intra_day_condition:
        if run_context["per_day_intraday_count"][date_key] < intraday_cap:
            run_context["per_day_intraday_count"][date_key] += 1
        reservoir_manager.sample(line, date_key, "intraday")
    else:
        if run_context["per_day_overnight_count"][date_key] < overnight_cap:
            run_context["per_day_overnight_count"][date_key] += 1
        reservoir_manager.sample(line, date_key, "overnight")
