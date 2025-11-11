"""
Purpose
-------
Stream a month’s CC-NEWS WARC queue from S3 and perform per-day, per-session
(intraday/overnight) reservoir sampling using NYSE trading calendar metadata.

Key behaviors
-------------
- Reads a monthly queue file line-by-line from S3 and parses CC-NEWS timestamps.
- Uses trading-calendar–derived caps to construct per-day/session reservoirs and
  perform single-pass, fixed-capacity sampling.
- Incorporates spillover candidates from the previous month and records
  spillover for the next month when items fall beyond the current calendar slice.
- Emits a structured run context with aggregate and per-day counters for
  observability and debugging.

Conventions
-----------
- Timezone for trading-day bucketing is America/New_York.
- Date keys are New York trading days formatted as '%Y-%m-%d'.
- Session labels are the strings "intraday" and "overnight".
- Input is a monthly queue file at 's3://<bucket>/<key>' containing CC-NEWS
  WARC paths that embed UTC timestamps in their filenames.

Downstream usage
----------------
Orchestration code constructs a `RunData` instance for a given month and calls
`extract_sample(run_data)`. The returned mapping
`{date: {"intraday": [...], "overnight": [...]}}` is then written to storage.
Spillover dictionaries on `RunData` allow consecutive months to be processed
in sequence while maintaining cross-month overnight continuity.
"""

import re
from typing import TYPE_CHECKING, Any, List, cast

import boto3
import pandas as pd

from aws.ccnews_sampler.calendar_utils import extract_next_trading_day
from aws.ccnews_sampler.ccnews_sampler_config import DATE_FMT, DATE_TZ
from aws.ccnews_sampler.data_maps import DataMaps, build_data_maps, to_seconds_int
from aws.ccnews_sampler.quota import compute_daily_caps
from aws.ccnews_sampler.reservoir_sampling import OverIntraSamples, ReservoirManager
from infra.logging.infra_logger import InfraLogger

if TYPE_CHECKING:
    from aws.ccnews_sampler.run_data import RunData


def extract_sample(run_data: "RunData") -> dict[str, OverIntraSamples]:
    """
    Run the full sampling pipeline for a single month’s CC-NEWS queue.

    Parameters
    ----------
    run_data : RunData
        Execution context for the month, including S3 location, NYSE calendar,
        daily_cap, logger, RNG, and spillover mappings.

    Returns
    -------
    dict[str, OverIntraSamples]
        Nested mapping of New York trading day ('YYYY-MM-DD') to dictionaries with
        "intraday" and "overnight" sample lists.

    Raises
    ------
    KeyError
        If required calendar columns are missing when building data maps.
    Exception
        Propagates unexpected I/O or S3 errors when fetching the monthly queue.

    Notes
    -----
    - Computes per-day intraday/overnight caps from `run_data.nyse_cal`, builds
      constant-time lookup maps, flushes any incoming spillover, and then streams
      the S3 queue to populate reservoirs.
    - Logging marks "month_scan_started" and "month_scan_finished" with a structured
      run context to aid observability.
    """

    year: str = run_data.year
    month: str = run_data.month
    logger: InfraLogger = run_data.logger
    nyse_cal: pd.DataFrame = run_data.nyse_cal

    dt_index: pd.DatetimeIndex = cast(pd.DatetimeIndex, nyse_cal.index)
    str_date_list: List[str] = dt_index.strftime(DATE_FMT).tolist()

    run_context: dict[str, Any] = generate_run_context(str_date_list, year, month)

    logger.debug("month_scan_started", context=run_context)

    nyse_cal = compute_daily_caps(run_data.daily_cap, nyse_cal, run_data.rng)
    data_maps: DataMaps = build_data_maps(nyse_cal)
    reservoir_manager: ReservoirManager = ReservoirManager(data_maps.cap_dict, run_data.rng)
    flush_spillover(run_data, reservoir_manager)
    fill_reservoirs(run_context, run_data, data_maps, reservoir_manager)

    logger.debug("month_scan_finished", context=run_context)

    return reservoir_manager.extract_sample_dict()


def generate_run_context(days: List[str], year: str, month: str) -> dict[str, Any]:
    """
    Initialize a structured run context for tracking scan-level counters.

    Parameters
    ----------
    days : list[str]
        List of New York trading day keys ('YYYY-MM-DD') for the month.
    year : str
        Four-digit year identifier for the run.
    month : str
        Two-digit month identifier for the run.

    Returns
    -------
    dict
        Mutable dictionary containing run metadata (year, month), global counters
        (lines_total, lines_matched, lines_unmatched), per-day session counters,
        and a small buffer of example unmatched/off-month lines.

    Raises
    ------
    None

    Notes
    -----
    - Per-day counters are initialized to zero for every date in `days`.
    - The `unknown_or_offmonth_examples` list is used only for diagnostics and is
      intentionally small to keep logging payloads bounded.
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


def flush_spillover(run_data: "RunData", reservoir_manager: ReservoirManager) -> None:
    """
    Seed the current month’s reservoirs with overnight spillover candidates.

    Parameters
    ----------
    run_data : RunData
        Run context holding `spillover_in`, a mapping from date keys to lists of
        candidate WARC paths carried over from the previous month.
    reservoir_manager : ReservoirManager
        Manager exposing per-day/session reservoirs for the current month.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If `spillover_in` contains a date key that is not present in
        `reservoir_manager.reservoir_dict`.

    Notes
    -----
    - All spillover candidates are treated as "overnight" items for their
      associated date keys and are fed through the reservoir sampler as if they
      appeared at the start of the current month’s stream.
    - Because reservoir sampling produces a uniform size-k subset of all items it processes
      for a given date, feeding in spillover items before in-month items still yields a uniform
      sample over the combined set of spillover and in-month candidates for that date (i.e.
      data ingestion order is of no statistical importance in this case).
    """

    for date_key, candidate_list in run_data.spillover_in.items():
        for candidate in candidate_list:
            reservoir_manager.sample(candidate, date_key, "overnight")


def fill_reservoirs(
    run_context: dict, run_data: "RunData", data_maps: DataMaps, reservoir_manager: ReservoirManager
) -> None:
    """
    Stream the monthly queue from S3 and update per-day/session reservoirs.

    Parameters
    ----------
    run_context : dict
        Mutable structure holding aggregate and per-day counters as well as
        example unmatched/off-month lines.
    run_data : RunData
        Execution context including S3 bucket/key, year/month, logger, and spillover.
    data_maps : DataMaps
        Precomputed lookup maps providing per-day caps, session boundaries, and
        valid date keys.
    reservoir_manager : ReservoirManager
        Manager that owns per-day "intraday" and "overnight" reservoirs.

    Returns
    -------
    None

    Raises
    ------
    botocore.exceptions.BotoCoreError
        On low-level S3 issues (network errors, timeouts, etc.).
    botocore.exceptions.ClientError
        If the specified S3 object is missing or access is denied.
    Exception
        For unexpected decoding or stream errors encountered during iteration.

    Notes
    -----
    - Reads the S3 object as a streaming body and processes each line once,
      updating counters and routing valid lines to the correct reservoir/session.
    - Lines whose dates fall outside `data_maps.valid_date_set` are treated
      as unmatched and captured in the run context.
    - Lines that logically map to a trading day beyond the current month’s
      calendar slice are recorded into `run_data.spillover_out` for the next month.
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
                        line,
                        utc_date,
                        date_key,
                        run_context,
                        data_maps,
                        reservoir_manager,
                        run_data,
                    )


def extract_link_date(line: str, date_pattern: re.Pattern[str]) -> tuple[pd.Timestamp, str] | None:
    """
    Extract the UTC timestamp and New York trading date key from a WARC path.

    Parameters
    ----------
    line : str
        Raw line from the monthly queue representing a CC-NEWS WARC path.
    date_pattern : re.Pattern[str]
        Compiled regex capturing the yyyymmdd and hhmmss timestamp components.

    Returns
    -------
    tuple[pandas.Timestamp, str] or None
        A pair `(utc_timestamp, date_key)` where `utc_timestamp` is a tz-aware
        pandas Timestamp in UTC and `date_key` is the New York trading day
        ('YYYY-MM-DD'). Returns None if the line does not match `date_pattern`.

    Raises
    ------
    None

    Notes
    -----
    - Timestamp parsing is done with `utc=True`, and the trading date key is
      obtained by converting to `DATE_TZ` (America/New_York) and formatting with
      `DATE_FMT`.
    - This function does not validate that the resulting date key is in the
      current month; callers must check membership in `data_maps.valid_date_set`.
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
    Update run context for a line that failed parsing or fell outside the calendar.

    Parameters
    ----------
    line : str
        The raw queue line that could not be parsed into a valid trading date or
        mapped into the current calendar.
    run_context : dict
        Mutable run context holding counters and example unmatched lines.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    - Increments `lines_unmatched` and, for the first few occurrences, appends the
      stripped line into `unknown_or_offmonth_examples` for later inspection.
    - This function does not log directly; callers are expected to emit the final
      run context once scanning completes.
    """

    run_context["lines_unmatched"] += 1
    if len(run_context["unknown_or_offmonth_examples"]) < 5:
        run_context["unknown_or_offmonth_examples"].append(line.strip())


def handle_correct_line(
    line: str,
    utc_date: pd.Timestamp,
    date_key: str,
    run_context: dict[str, Any],
    data_maps: DataMaps,
    reservoir_manager: ReservoirManager,
    run_data: "RunData",
) -> None:
    """
    Route a valid line to the appropriate session reservoir and update counters.

    Parameters
    ----------
    line : str
        WARC path line that successfully parsed into a timestamp and date key.
    utc_date : pandas.Timestamp
        Tz-aware UTC timestamp extracted from the WARC filename.
    date_key : str
        New York trading-day key ('YYYY-MM-DD') corresponding to `utc_date`.
    run_context : dict[str, Any]
        Mutable run context tracking per-day intraday/overnight counts and totals.
    data_maps : DataMaps
        Lookup structure supplying per-day caps, session boundary seconds, and
        the set of valid trading-day keys for the current month.
    reservoir_manager : ReservoirManager
        Destination for sampling via `sample(line, date_key, session)`.
    run_data : RunData
        Run context, used here to access spillover mappings and the monthly
        NYSE calendar index.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If `date_key` is not present in `data_maps.cap_dict` or
        `data_maps.session_dict` despite having passed the `valid_date_set`
        check upstream.
    KeyError
        If a computed next trading-day key is not present in `data_maps.cap_dict`
        when it is expected to be within the current month’s calendar slice.

    Notes
    -----
    - Determines the session for `utc_date` by comparing its epoch seconds
      against the per-day open/close boundaries from `data_maps.session_dict`:
        * between open and close → "intraday"
        * before open → "overnight" of the same trading day
        * after close → "overnight" of the next trading day
    - The next trading day is obtained via `extract_next_trading_day(utc_date)`,
      which uses the XNYS exchange calendar and respects the global
      `LAST_SAMPLING_DAY` horizon:
        * If the returned key is in `data_maps.valid_date_set`, the line is
          routed into that day’s overnight reservoir (subject to its cap).
        * If the returned key is not in `valid_date_set`, the line is recorded
          into `run_data.spillover_out[next_date_key]` for the subsequent month.
        * If `extract_next_trading_day` returns `None`, the line is silently
          ignored as beyond the sampling horizon.
    - Per-day counters in `run_context` are bounded by the corresponding caps
      for observability only; the reservoirs themselves enforce capacity and
      uniformity of the final sample.
    """

    intraday_cap, overnight_cap = data_maps.cap_dict[date_key]
    current_session_open, current_session_close = data_maps.session_dict[date_key]
    current_date_seconds: int = to_seconds_int(utc_date)
    if current_session_open and current_session_close:
        intraday_condition: bool = (
            current_session_open <= current_date_seconds < current_session_close
        )
        if intraday_condition:
            if run_context["per_day_intraday_count"][date_key] < intraday_cap:
                run_context["per_day_intraday_count"][date_key] += 1
            reservoir_manager.sample(line, date_key, "intraday")
        elif current_date_seconds < current_session_open:
            if run_context["per_day_overnight_count"][date_key] < overnight_cap:
                run_context["per_day_overnight_count"][date_key] += 1
            reservoir_manager.sample(line, date_key, "overnight")
        else:
            next_date_key_str: str | None = extract_next_trading_day(utc_date)
            if not next_date_key_str:
                return
            if next_date_key_str in data_maps.valid_date_set:
                next_day_overnight_cap: int = data_maps.cap_dict[next_date_key_str][1]
                if (
                    run_context["per_day_overnight_count"][next_date_key_str]
                    < next_day_overnight_cap
                ):
                    run_context["per_day_overnight_count"][next_date_key_str] += 1
                reservoir_manager.sample(line, next_date_key_str, "overnight")
            else:
                run_data.spillover_out.setdefault(next_date_key_str, []).append(line)
