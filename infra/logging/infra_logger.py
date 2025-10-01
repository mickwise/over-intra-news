"""
Purpose
-------
Provide a structured, configurable logging utility for the infra module.
Centralizes handling of log levels, formats, destinations, and run metadata.

Key behaviors
-------------
- Emits one structured log entry per call (`emit`).
- Supports log level thresholding (DEBUG, INFO, WARNING, ERROR).
- Serializes entries as JSON (default) or human-readable text.
- Handles invalid environment variables by falling back to defaults.
- Ensures logging never interrupts program execution.

Conventions
-----------
- Default log level is INFO.
- Default format is JSON; text output is line-based with key=value context.
- Default destination is STDERR; file destinations are opened in append mode.
- Timestamps are UTC ISO-8601 with a trailing "Z".

Downstream usage
----------------
Call `initialize_logger` at process start to configure and obtain a logger.
Use `logger.debug`, `logger.info`, `logger.warning`, or `logger.error` in code.
"""

import datetime as dt
import json
import os
import sys
from typing import TypedDict

LOG_LEVELS: set[str] = {"DEBUG", "INFO", "WARNING", "ERROR"}
LEVEL_MAPPING: dict[str, int] = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
LOG_FORMATS: set[str] = {"json", "text"}


class LogEntry(TypedDict):
    """
    Purpose
    -------
    Typed dictionary describing the structure of a single log entry.

    Fields
    ------
    timestamp : str
        UTC ISO-8601 timestamp string with a "Z" suffix.
    level : str
        Log severity level ("DEBUG", "INFO", "WARNING", "ERROR").
    run_id : str
        Unique identifier for the run that emitted this entry.
    component : str
        Name of the component producing the log.
    event : str
        Short machine-readable event name (snake_case).
    message : str
        Human-readable message string.
    run_meta : dict
        Immutable run metadata attached at logger initialization.
    context : dict
        Event-specific context payload (small, JSON-serializable).
    """

    timestamp: str
    level: str
    run_id: str
    component: str
    event: str
    message: str
    run_meta: dict
    context: dict


class InfraLogger:
    """
    Purpose
    -------
    A structured logger for infra modules that enforces level thresholds,
    normalizes output format, and writes to configurable destinations.

    Key behaviors
    -------------
    - Emits structured log entries with `emit`.
    - Provides convenience methods for each level (`debug`, `info`, `warning`, `error`).
    - Supports JSON and text formats.
    - Prevents invalid logging configuration by falling back to defaults.

    Parameters
    ----------
    component_name : str
        Human-readable name of the component or module using this logger.
    run_id : str
        Unique identifier for this execution (auto-generated if not provided).
    run_meta : dict
        Immutable run-scoped metadata.
    log_level : str, default="INFO"
        Minimum log level threshold ("DEBUG", "INFO", "WARNING", "ERROR").
    log_format : str, default="json"
        Output format ("json" or "text").
    log_dest : str, default="stderr"
        Destination for logs ("stderr" or a valid file path).

    Attributes
    ----------
    component_name : str
        Component label propagated into every log entry.
    run_id : str
        Identifier used to correlate entries from the same run.
    run_meta : dict
        Serialized into each log entry.
    level : str
        Effective log level threshold.
    format : str
        Effective output format.
    dest : str
        Effective output destination.

    Notes
    -----
    - Logging never raises exceptions; serialization falls back to `default=str`.
    - File destinations are opened in append mode with UTF-8 encoding.
    """

    def __init__(
        self,
        component_name: str,
        run_id: str,
        run_meta: dict,
        log_level: str = "INFO",
        log_format: str = "json",
        log_dest: str = "stderr",
    ) -> None:
        self.component_name = component_name
        self.run_id = run_id
        self.run_meta = run_meta
        self.level = log_level
        self.format = log_format
        self.dest = log_dest

    def emit(
        self, event: str, level: str = "INFO", msg: str | None = None, context: dict | None = None
    ) -> None:
        """
        Emit one structured log entry.

        Parameters
        ----------
        event : str
            Snake_case event name describing what happened.
        level : str, default="INFO"
            Log severity level.
        msg : str, optional
            Human-readable message string.
        context : dict, optional
            Event-specific payload; must be JSON-serializable or convertible to str.

        Returns
        -------
        None

        Notes
        -----
        - Respects log level threshold: entries below threshold are dropped.
        - Timestamp is recorded in UTC ISO-8601 with a "Z" suffix.
        """

        if LEVEL_MAPPING[level] < LEVEL_MAPPING[self.level]:
            return
        if context is None:
            context = {}
        if msg is None:
            msg = ""
        entry: LogEntry = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": level,
            "run_id": self.run_id,
            "component": self.component_name,
            "event": event,
            "message": msg,
            "run_meta": self.run_meta,
            "context": context,
        }
        formatted_entry = self.format_entry(entry)
        self.write_entry(formatted_entry)

    def debug(self, event: str, msg: str | None = None, context: dict | None = None) -> None:
        """
        Emit a DEBUG log entry.

        Parameters
        ----------
        event : str
            Snake_case event name.
        msg : str, optional
            Human-readable message.
        context : dict, optional
            Event-specific context payload.

        Returns
        -------
        None
        """

        self.emit(event=event, level="DEBUG", msg=msg, context=context)

    def info(self, event: str, msg: str | None = None, context: dict | None = None) -> None:
        """
        Emit a INFO log entry.

        Parameters
        ----------
        event : str
            Snake_case event name.
        msg : str, optional
            Human-readable message.
        context : dict, optional
            Event-specific context payload.

        Returns
        -------
        None
        """

        self.emit(event=event, level="INFO", msg=msg, context=context)

    def warning(self, event: str, msg: str | None = None, context: dict | None = None) -> None:
        """
        Emit a WARNING log entry.

        Parameters
        ----------
        event : str
            Snake_case event name.
        msg : str, optional
            Human-readable message.
        context : dict, optional
            Event-specific context payload.

        Returns
        -------
        None
        """

        self.emit(event=event, level="WARNING", msg=msg, context=context)

    def error(self, event: str, msg: str | None = None, context: dict | None = None) -> None:
        """
        Emit a ERROR log entry.

        Parameters
        ----------
        event : str
            Snake_case event name.
        msg : str, optional
            Human-readable message.
        context : dict, optional
            Event-specific context payload.

        Returns
        -------
        None
        """

        self.emit(event=event, level="ERROR", msg=msg, context=context)

    def format_entry(self, entry: LogEntry) -> str:
        """
        Format a log entry into the configured output format.

        Parameters
        ----------
        entry : LogEntry
            Structured log entry dictionary.

        Returns
        -------
        str
            Serialized log entry (JSON string or human-readable text).

        Notes
        -----
        - JSON output falls back to `default=str` on serialization failure.
        - Text output is line-based with key=value pairs for context.
        """

        if self.format == "json":
            try:
                return json.dumps(entry, ensure_ascii=False)

            except (TypeError, ValueError):
                return json.dumps(entry, ensure_ascii=False, default=str)
        else:
            context_str = " ".join(f"{k}={v}" for k, v in entry["context"].items())
            return (
                f"{entry['timestamp']} [{entry['level']}] "
                f"{entry['component']} {entry['event']} - {entry['message']} "
                f"{context_str}"
            )

    def write_entry(self, formatted_entry: str) -> None:
        """
        Write a formatted log entry to the configured destination.

        Parameters
        ----------
        formatted_entry : str
            Log entry string, already serialized.

        Returns
        -------
        None

        Notes
        -----
        - If destination is "stderr", writes to STDERR.
        - Otherwise, appends to the specified file with UTF-8 encoding.
        """

        if self.dest == "stderr":
            print(formatted_entry, file=sys.stderr)
        else:
            with open(self.dest, "a", encoding="utf-8") as f:
                f.write(formatted_entry + "\n")


def initialize_logger(
    component_name: str,
    run_id: str | None = None,
    run_meta: dict | None = None,
) -> InfraLogger:
    """
    Factory function to configure and return an InfraLogger.

    Parameters
    ----------
    component_name : str
        Name of the component using the logger.
    run_id : str, optional
        Unique identifier for the run; auto-generated if not provided.
    run_meta : dict, optional
        Immutable run metadata dictionary.

    Returns
    -------
    InfraLogger
        Configured logger instance with validated environment overrides.

    Notes
    -----
    - Environment variables LOG_LEVEL, LOG_FORMAT, LOG_DEST are honored.
    - Invalid values fall back to defaults with warnings emitted.
    """

    fall_backs = {
        "level": False,
        "log_format": False,
        "log_dest": False,
    }
    level, log_format, log_dest = extract_env_vars(fall_backs)

    if run_id is None:
        run_id = generate_run_id(component_name)

    if run_meta is None:
        run_meta = {}

    logger = InfraLogger(
        component_name=component_name,
        run_id=run_id,
        run_meta=run_meta,
        log_level=level,
        log_format=log_format,
        log_dest=log_dest,
    )
    handle_fallbacks(logger, fall_backs)
    return logger


def extract_env_vars(fall_backs: dict[str, bool]) -> tuple[str, str, str]:
    """
    Extract and validate logging configuration from environment variables.

    Parameters
    ----------
    fall_backs : dict[str, bool]
        Mutable dict tracking whether defaults had to be applied.

    Returns
    -------
    tuple[str, str, str]
        Normalized (level, format, destination).

    Notes
    -----
    - Invalid values are replaced with defaults and marked in fall_backs.
    - Destination must be "stderr" or a writable file path.
    """

    level: str = os.environ.get("LOG_LEVEL", "INFO")
    log_format: str = os.environ.get("LOG_FORMAT", "json")
    log_dest: str = os.environ.get("LOG_DEST", "stderr")

    up_level: str = level.upper()
    if up_level not in LOG_LEVELS:
        fall_backs["level"] = True
        level = "INFO"

    if log_format.lower() not in LOG_FORMATS:
        fall_backs["log_format"] = True
        log_format = "json"

    if log_dest.lower() != "stderr":
        try:
            with open(log_dest, "a", encoding="utf-8"):
                pass
        except OSError:
            fall_backs["log_dest"] = True
            log_dest = "stderr"

    return level.upper(), log_format.lower(), log_dest


def generate_run_id(component_name: str) -> str:
    """
    Generate a unique run identifier.

    Parameters
    ----------
    component_name : str
        Component name to prefix the run_id.

    Returns
    -------
    str
        Run identifier of the form `<component>--<UTC timestamp>--<pid>`.

    Notes
    -----
    - Timestamps are in UTC with second precision.
    - Process ID is included to distinguish concurrent runs.
    """

    run_id: str = (
        component_name
        + "--"
        + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "--"
        + str(os.getpid())
    )
    return run_id


def handle_fallbacks(logger: InfraLogger, fall_backs: dict[str, bool]) -> None:
    """
    Emit warnings if environment variable fallbacks were applied.

    Parameters
    ----------
    logger : InfraLogger
        Logger instance used to emit warnings.
    fall_backs : dict[str, bool]
        Mapping of config keys ("level", "log_format", "log_dest") to fallback flags.

    Returns
    -------
    None

    Notes
    -----
    - Emits one WARNING log per invalid environment variable.
    - Invalid values are attached in the context for diagnosis.
    """

    for key, triggered in fall_backs.items():
        if triggered:
            if key == "level":
                logger.emit(
                    event="FALLBACK_LOG_LEVEL",
                    level="WARNING",
                    msg="Invalid LOG_LEVEL env var; defaulting to INFO",
                    context={"invalid_value": os.environ.get("LOG_LEVEL", None)},
                )
            elif key == "log_format":
                logger.emit(
                    event="FALLBACK_LOG_FORMAT",
                    level="WARNING",
                    msg="Invalid LOG_FORMAT env var; defaulting to json",
                    context={"invalid_value": os.environ.get("LOG_FORMAT", None)},
                )
            elif key == "log_dest":
                logger.emit(
                    event="FALLBACK_LOG_DEST",
                    level="WARNING",
                    msg="Invalid LOG_DEST env var; defaulting to stderr",
                    context={"invalid_value": os.environ.get("LOG_DEST", None)},
                )
