'''
Purpose
-------
Emit structured, one-line JSON logs for the news sampling pipeline. Centralizes
run metadata (run_id, shard_name), normalizes timestamps to UTC, and ensures
logs never crash the process.

Key behaviors
-------------
- Produces a single JSON object per line (JSONL) to STDOUT.
- Timestamps are UTC ISO-8601 with a trailing "Z".
- Includes `run_meta` (constant per run) and a free-form `context` object.
- Serialization is resilient: falls back to `default=str` on non-JSONable values.
- Flushes immediately to avoid buffering delays.

Conventions
-----------
- `event` names use snake_case (e.g., `month_scan_finished`, `parse_warning`).
- Keep `context` small and JSON-serializable; prefer strings, numbers, lists, dicts.
- `run_meta` is immutable per run; pass only stable config like year, cap, strict flag.

Downstream usage
----------------
Instantiate once per process and call `emit(event, level, context=...)` at key
checkpoints. Later filter by `event` or `level`, or group by `run_id`
to analyze an entire execution.
'''
import datetime as dt
import json

from aws.ccnews_sampler.monthly_uniform_sampling import FinalLogData
from aws.ccnews_sampler.run_data import RunData


class RunLogger:
    """
    Lightweight structured logger that writes JSONL events to STDOUT with run-scoped
    metadata (run_id, shard_name, run_meta).

    Parameters
    ----------
    run_id : str
        Unique identifier for this process/run (e.g., UUIDv4).
    shard_name : str
        Human-readable shard label (e.g., "2019").
    run_meta : dict
        Immutable run configuration to attach to every event (e.g., year, caps, flags).

    Attributes
    ----------
    run_id : str
        Propagated on every event for correlation.
    shard_name : str
        Propagated on every event to identify the shard.
    run_meta : dict
        Serialized into each log entry under "run_meta".

    Notes
    -----
    - This class never raises on serialization: it prints a best-effort fallback
    line if JSON encoding fails, then retries with `default=str`.
    - Designed for machine parsing (one JSON object per line).
    """

    def __init__(
            self,
            run_id: str,
            shard_name: str,
            run_meta: dict
            ) -> None:
        """
        Create a run-scoped logger with fixed metadata fields.

        Parameters
        ----------
        run_id : str
            Unique identifier for the current execution.
        shard_name : str
            Label for the shard this process is handling.
        run_meta : dict
            Immutable configuration to include in every log entry.

        Returns
        -------
        None

        Raises
        ------
        None

        Notes
        -----
        - `run_meta` should be JSON-serializable or convertible to strings by the logger.
        """
        self.run_id: str = run_id
        self.shard_name: str = shard_name
        self.run_meta: dict = run_meta


    def emit(
            self,
            event: str,
            level: str,
            context: dict | None = None,
            ) -> None:
        """
        Write one structured log event as a single JSON line to STDOUT.

        Parameters
        ----------
        event : str
            Snake_case event name describing what happened (e.g., "month_manifest_loaded").
        level : str
            Log severity (e.g., "INFO", "WARN", "ERROR"); case-insensitive, coerced to upper.
        context : dict | None
            Optional event-specific payload; must be JSON-serializable or convertible to str.

        Returns
        -------
        None

        Raises
        ------
        None  (serialization errors are caught; a fallback line is emitted)

        Notes
        -----
        - Timestamp is UTC ISO-8601 with a "Z" suffix.
        - On serialization failure, the logger prints a minimal error line and retries
        with `default=str` to guarantee progress.
        """
        time_stamp: str = dt.datetime.now(dt.timezone.utc).isoformat()
        log_entry: dict = {
            "run_id": self.run_id,
            "shard_name": self.shard_name,
            "run_meta": self.run_meta,
            "event": event,
            "level": level,
            "timestamp": time_stamp.replace("+00:00", "Z"),
            "context": context or {},
        }
        try:
            print(json.dumps(log_entry), flush=True)
        except (TypeError, ValueError) as e:
            print(f"Logging error: {e}", flush=True)
            print(json.dumps(log_entry, default=str), flush=True)


    def initial_emission(
        self,
        run_data: RunData,
    ) -> None:
        """
        Emit a start-of-month manifest event for observability.

        Parameters
        ----------
        logger : RunLogger
            Structured logger used for run-scoped events.
        run_data : RunData
            Execution context; fields consumed are (year, month, bucket, key).

        Returns
        -------
        None

        Notes
        -----
        - Produces the "month_manifest_loaded" INFO event with year/month and the
        S3
        """
        self.emit(
            "month_manifest_loaded",
            "INFO",
            {
                "year": run_data.year,
                "month": run_data.month,
                "s3_bucket": run_data.bucket,
                "s3_key": run_data.key
            }
        )


    def check_line_count(
            self,
            run_context: dict,
            year: str,
            month: str
        ) -> None:
        """
        Emit a warning summary if unmatched lines were observed.

        Parameters
        ----------
        run_context : dict
            Mutable counters accumulated during the scan. Must contain
            "lines_unmatched" and "unknown_or_offmonth_examples".
        logger : RunLogger
            Structured logger used for run-scoped events.
        year : str
            Four-digit year for context in the log record.
        month : str
            Two-digit month for context in the log record.

        Returns
        -------
        None

        Notes
        -----
        - If lines_unmatched > 0, emits "month_manifest_warnings" with counts and
        up to 5 example lines to aid diagnosis.
        """
        if run_context["lines_unmatched"] > 0:
            self.emit(
                "month_manifest_warnings",
                "WARN",
                {
                    "year": year,
                    "month": month,
                    "lines_unmatched": run_context["lines_unmatched"],
                    "unknown_or_offmonth_examples": run_context["unknown_or_offmonth_examples"]
                }
            )


    def samples_emitted(
            self,
            final_log_dict: FinalLogData,
        ) -> None:
        """
        Emit a terminal summary event after all per-day/session sample files are written.

        Parameters
        ----------
        final_log_dict : FinalLogData
            Payload to record with the event. Expected keys:
            - 'year' : str
            - 'month' : str
            - 'output_prefix' : str  (S3 prefix under which files were written)
            - 'days_processed' : int
            - 'files_written' : int

        Returns
        -------
        None

        Notes
        -----
        - Event name: "samples_written" (INFO).
        - Thin wrapper around `emit(...)` to standardize the final summary.
        """
        self.emit(
            "samples_written",
            "INFO",
            final_log_dict
        )
