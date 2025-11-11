"""
Purpose
-------
Unit tests for the S&P 500 membership extraction helpers.

Key behaviors
-------------
- Verify that `extract_historical_constituents`:
  - delegates CSV loading to `pandas.read_csv`,
  - parses the `date` column to tz-aware UTC timestamps,
  - sorts by `date` and filters rows to an inclusive `[start_date, end_date]`
    window,
  - resets the index on the filtered frame.
- Verify that `extract_snp_membership_windows`:
  - reads `START_DATE` and `END_DATE` from the environment,
  - converts them to UTC timestamps via `str_to_timestamp`,
  - logs the configured window and provenance metadata,
  - delegates to `extract_historical_constituents` with the parsed bounds.

Conventions
-----------
- Network I/O is not exercised; `pandas.read_csv` is stubbed to return a small
  in-memory DataFrame.
- Environment variables are injected via `monkeypatch.setenv` to keep tests
  hermetic and side-effect free.
- Logging is captured via a dummy logger object exposing an `.info(...)` method;
  type compatibility with `InfraLogger` is maintained via explicit casts.

Downstream usage
----------------
Run via `pytest` as part of the CI suite. These tests document the expected
shape and semantics of the membership window extraction logic while keeping
external dependencies stubbed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, cast

import pandas as pd
import pytest

from infra.logging.infra_logger import InfraLogger
from infra.seeds import seed_snp_memberships
from infra.utils.db_utils import str_to_timestamp


class _DummyLogger:
    """
    Purpose
    -------
    Minimal stand-in for `InfraLogger` used in S&P 500 membership tests.

    Key behaviors
    -------------
    - Collects all calls to `.info(msg, context=...)` in a list so tests can
      assert on message IDs and structured context payloads.

    Parameters
    ----------
    None

    Attributes
    ----------
    infos : list[tuple[str, dict[str, Any]]]
        Accumulates `(msg, context)` tuples recorded from `.info(...)` calls.

    Notes
    -----
    - This class intentionally implements only the subset of the `InfraLogger`
      interface used by the tested functions.
    - Type compatibility with production code is achieved with
      `typing.cast(InfraLogger, _DummyLogger(...))` at call sites.
    """

    def __init__(self) -> None:
        self.infos: List[Tuple[str, Dict[str, Any]]] = []

    def info(self, msg: str, context: Dict[str, Any] | None = None) -> None:
        """
        Record an info-level log message and its context.

        Parameters
        ----------
        msg : str
            Message identifier or human-readable message.
        context : dict[str, Any] or None, optional
            Structured context payload associated with the message.

        Returns
        -------
        None

        Raises
        ------
        None

        Notes
        -----
        - If `context` is None, an empty dictionary is stored instead.
        """
        self.infos.append((msg, context or {}))


def test_extract_historical_constituents_filters_and_normalizes_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Validate that `extract_historical_constituents` parses, sorts, and windows by date.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to stub `pandas.read_csv` so that no network I/O occurs.

    Returns
    -------
    None
        The test passes if:
            - `pandas.read_csv` is invoked once with the expected URL,
            - the `date` column is parsed to tz-aware UTC `Timestamp`s,
            - rows are filtered to the inclusive window
              `[start_date, end_date]`,
            - the resulting index is reset and monotonic from 0.

    Raises
    ------
    AssertionError
        If any of the invariants above are violated.
    """

    # Construct a fake CSV payload with unsorted dates.
    fake_csv = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-10", "2024-01-05"],
            "tickers": ["A,B", "C,D", "E,F"],
        }
    )

    captured_url: dict[str, str] = {}

    def fake_read_csv(url: str) -> pd.DataFrame:
        captured_url["url"] = url
        return fake_csv.copy()

    monkeypatch.setattr(seed_snp_memberships.pd, "read_csv", fake_read_csv)

    start_date = pd.Timestamp("2024-01-03", tz="UTC")
    end_date = pd.Timestamp("2024-01-08", tz="UTC")

    logger = cast(InfraLogger, _DummyLogger())

    result = seed_snp_memberships.extract_historical_constituents(
        start_date=start_date,
        end_date=end_date,
        logger=logger,
    )

    # CSV should be fetched from the pinned URL.
    assert captured_url["url"] == seed_snp_memberships.SNP_HISTORICAL_URL

    # Date column should be tz-aware and filtered to [2024-01-03, 2024-01-08].
    assert "date" in result.columns
    assert result["date"].dt.tz is not None

    dates = list(result["date"])
    assert dates == [pd.Timestamp("2024-01-05", tz="UTC")]

    # Index should be reset to 0..n-1.
    assert list(result.index) == [0]

    # tickers column should be preserved for filtered rows.
    assert list(result["tickers"]) == ["E,F"]


def test_extract_snp_membership_windows_reads_env_and_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure `extract_snp_membership_windows` reads env dates, logs, and delegates.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to inject environment variables and stub
        `extract_historical_constituents`.

    Returns
    -------
    None
        The test passes if:
            - `START_DATE` and `END_DATE` are read from the environment,
            - the parsed timestamps passed to `extract_historical_constituents`
              match `str_to_timestamp` of those env values,
            - an info log entry is emitted with the expected context, and
            - the function returns exactly what the stubbed
              `extract_historical_constituents` returns.

    Raises
    ------
    AssertionError
        If any of the env, logging, or delegation expectations fail.
    """

    # Inject window bounds via environment.
    monkeypatch.setenv("START_DATE", "2024-01-01")
    monkeypatch.setenv("END_DATE", "2024-01-31")

    logger = _DummyLogger()

    captured_args: dict[str, Any] = {}

    def fake_extract_historical_constituents(
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        _logger: InfraLogger,
    ) -> pd.DataFrame:
        captured_args["start_date"] = start_date
        captured_args["end_date"] = end_date
        # Return a recognizable dummy frame.
        return pd.DataFrame({"date": [start_date], "tickers": ["A,B"]})

    monkeypatch.setattr(
        seed_snp_memberships,
        "extract_historical_constituents",
        fake_extract_historical_constituents,
    )

    result = seed_snp_memberships.extract_snp_membership_windows(logger=cast(InfraLogger, logger))

    # Parsed timestamps should match str_to_timestamp of env values.
    expected_start = str_to_timestamp("2024-01-01")
    expected_end = str_to_timestamp("2024-01-31")

    assert captured_args["start_date"] == expected_start
    assert captured_args["end_date"] == expected_end

    # The function should return whatever the inner helper returned.
    assert isinstance(result, pd.DataFrame)
    assert list(result["tickers"]) == ["A,B"]

    # Logging should include the configured stage and window.
    assert logger.infos, "No info logs were recorded"
    msg, context = logger.infos[0]
    assert msg == "snp_memberships_extraction_start"
    assert context["stage"] == "extract_snp_membership_windows"
    assert context["start_date"] == expected_start.isoformat()
    assert context["end_date"] == expected_end.isoformat()
    assert context["sha"] == seed_snp_memberships.FJA05680_SHA
    assert context["url"] == seed_snp_memberships.SNP_HISTORICAL_URL
