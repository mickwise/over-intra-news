"""
Purpose
-------
Unit tests for `aws.ccnews_sampler.extract_sample`.

Key behaviors
-------------
- Verify that `generate_run_context` initializes metadata and per-day counters.
- Exercise `extract_link_date`, including both matching and non-matching lines.
- Confirm that `handle_erroneous_line` increments counters and buffers examples.
- Ensure `flush_spillover` seeds overnight reservoirs correctly.
- Validate that `handle_correct_line` routes items to intraday/overnight
  sessions, including next-day routing and spillover at month edges.
- Check that `fill_reservoirs` streams from S3 and delegates to helpers.
- Assert that `extract_sample.extract_sample` orchestrates the end-to-end
  sampling pipeline wiring in a minimal, controlled setup.

Conventions
-----------
- Uses small, in-memory NYSE calendar slices and deterministic stubs to keep
  tests fast and predictable.
- Stubs `boto3`, calendar/data map builders, and reservoir manager where
  appropriate to isolate behavior under test.
- Dates are expressed in `DATE_FMT` ('YYYY-MM-DD') and timestamps in UTC.

Downstream usage
----------------
Run via `pytest` as part of the CI suite. These tests serve as executable
documentation for how the high-level sampling pipeline is expected to wire up
calendar metadata, spillover handling, and per-day/session sampling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd
import pytest

from aws.ccnews_sampler import data_maps as data_maps_mod
from aws.ccnews_sampler import extract_sample
from aws.ccnews_sampler.ccnews_sampler_config import DATE_FMT, DATE_TZ
from aws.ccnews_sampler.reservoir_sampling import ReservoirManager
from aws.ccnews_sampler.run_data import RunData


@dataclass
class DummyReservoirManager:
    """
    Purpose
    -------
    Minimal stand-in for `ReservoirManager` used in `extract_sample` tests.

    Key behaviors
    -------------
    - Records all calls to `sample(candidate, date, session)` for later
      inspection by tests.
    - Returns a preconfigured result from `extract_sample_dict()` or an empty
      dictionary if none was provided.

    Parameters
    ----------
    result : dict[str, dict[str, list[str]]] or None, optional
        Pre-populated nested mapping to be returned by `extract_sample_dict`;
        defaults to `None`, in which case an empty dictionary is returned.

    Attributes
    ----------
    result : dict[str, dict[str, list[str]]] or None
        Final sample structure, if provided.
    calls : list[tuple[str, str, str]]
        Sequence of `(candidate, date, session)` tuples collected from
        `sample(...)` invocations.

    Notes
    -----
    - This stub deliberately implements only the subset of the real
      `ReservoirManager` interface used by tests.
    - No randomness or reservoir replacement logic is modeled here; tests that
      need sampling semantics should target the reservoir module directly.
    """

    result: Dict[str, Dict[str, List[str]]] | None = None

    def __post_init__(self) -> None:
        self.calls: List[Tuple[str, str, str]] = []

    def sample(self, candidate: str, date: str, session: str) -> None:
        """
        Record a sampling call instead of performing real reservoir updates.

        Parameters
        ----------
        candidate : str
            Item that would have been considered for sampling.
        date : str
            Trading-day key ('YYYY-MM-DD') associated with the candidate.
        session : str
            Session label ("intraday" or "overnight") for the candidate.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        self.calls.append((candidate, date, session))

    def extract_sample_dict(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Return the configured reservoir result or an empty mapping.

        Parameters
        ----------
        None

        Returns
        -------
        dict[str, dict[str, list[str]]]
            Preconfigured result if `result` was set at construction time;
            otherwise an empty dictionary.

        Raises
        ------
        None
        """
        return self.result if self.result is not None else {}


@dataclass
class DummyRunData:
    """
    Purpose
    -------
    Lightweight test double for `RunData` consumed by `extract_sample`.

    Key behaviors
    -------------
    - Supplies minimal calendar, date metadata, and configuration required by
      the sampling pipeline.
    - Exposes `spillover_in` and `spillover_out` dictionaries that tests can
      inspect or pre-populate.
    - Provides a no-op logger compatible with the interface expected by
      `extract_sample.extract_sample`.

    Parameters
    ----------
    year : str
        Four-digit year string (e.g., "2024") for the run.
    month : str
        Two-digit month string (e.g., "01") for the run.
    nyse_cal : pandas.DataFrame
        NYSE calendar slice for the month under test; used by data map builders
        and session routing.
    bucket : str, optional
        S3 bucket name; default is "dummy-bucket".
    key : str, optional
        S3 object key for the monthly queue; default is "dummy-key".
    daily_cap : int, optional
        Per-day total sampling cap passed through to `compute_daily_caps`;
        default is 10.
    rng : numpy.random.Generator, optional
        RNG instance used by quota/reservoir components, defaulting to a
        deterministic generator.

    Attributes
    ----------
    spillover_in : dict[str, list[str]]
        Mapping of date keys to candidate paths carried in from the previous
        month; initialized to an empty dict.
    spillover_out : dict[str, list[str]]
        Mapping of date keys to candidates that fall beyond the current month’s
        calendar slice; initialized to an empty dict.
    logger : Any
        Minimal logger object exposing a `.debug(...)` method that ignores all
        arguments.

    Notes
    -----
    - Tests are free to mutate `spillover_in` and `spillover_out` directly.
    - The logger is intentionally minimal; tests focus on behavior rather than
      log contents.
    """

    year: str
    month: str
    nyse_cal: pd.DataFrame
    bucket: str = "dummy-bucket"
    key: str = "dummy-key"
    daily_cap: int = 10
    rng: np.random.Generator = np.random.default_rng(123)

    def __post_init__(self) -> None:
        self.spillover_in: Dict[str, List[str]] = {}
        self.spillover_out: Dict[str, List[str]] = {}
        self.logger = type(
            "DummyLogger",
            (),
            {"debug": lambda *_args, **_kwargs: None},
        )()


def test_generate_run_context_initializes_counters_and_metadata() -> None:
    """
    Verify that `generate_run_context` initializes metadata and per-day counters.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If any run-context field or per-day counter is initialized incorrectly.

    Notes
    -----
    - Asserts that year/month fields are set, global counters start at zero,
      per-day intraday/overnight mappings cover the provided days, and
      `unknown_or_offmonth_examples` starts empty.
    """

    days = ["2024-01-02", "2024-01-03"]
    ctx = extract_sample.generate_run_context(days, year="2024", month="01")

    assert ctx["year"] == "2024"
    assert ctx["month"] == "01"
    assert ctx["lines_total"] == 0
    assert ctx["lines_matched"] == 0
    assert ctx["lines_unmatched"] == 0

    assert set(ctx["per_day_intraday_count"].keys()) == set(days)
    assert set(ctx["per_day_overnight_count"].keys()) == set(days)
    assert all(v == 0 for v in ctx["per_day_intraday_count"].values())
    assert all(v == 0 for v in ctx["per_day_overnight_count"].values())
    assert ctx["unknown_or_offmonth_examples"] == []


def test_extract_link_date_parses_timestamp_and_trading_key() -> None:
    """
    Ensure `extract_link_date` parses a CC-NEWS line into UTC timestamp and key.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the function fails to return a timestamp and date key for a line
        that matches the supplied regex.

    Notes
    -----
    - Uses a synthetic CC-NEWS WARC path embedding a known UTC timestamp and
      verifies that the returned `date_key` matches the converted New York
      civil date in `DATE_TZ` and `DATE_FMT`.
    """

    # Example WARC path embedding 2024-01-02 15:30:00 UTC.
    line = "s3://cc-news/CC-NEWS-20240102-153000-000000-foo.warc.gz"
    date_pattern = re.compile(r"CC-NEWS-(20240102)-(153000)")

    parsed = extract_sample.extract_link_date(line, date_pattern)
    assert parsed is not None
    utc_ts, date_key = parsed

    assert utc_ts.tzinfo is not None
    assert utc_ts.tz_convert(DATE_TZ).strftime(DATE_FMT) == date_key


def test_extract_link_date_returns_none_on_no_match() -> None:
    """
    Confirm that `extract_link_date` returns None when no regex match is found.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If a non-matching line produces a non-None result.

    Notes
    -----
    - Uses a line that does not contain a CC-NEWS timestamp and a pattern that
      expects one; `extract_link_date` should return `None`.
    """

    line = "s3://cc-news/not-a-cc-news-line"
    date_pattern = re.compile(r"CC-NEWS-(20240102)-(153000)")
    assert extract_sample.extract_link_date(line, date_pattern) is None


def test_handle_erroneous_line_increments_counters_and_buffers_examples() -> None:
    """
    Verify that `handle_erroneous_line` increments unmatched count and buffers lines.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the unmatched counter is not incremented correctly or if more than
        five examples are stored.

    Notes
    -----
    - Calls `handle_erroneous_line` multiple times and checks that
      `lines_unmatched` counts all calls while the examples buffer holds only
      the first five lines.
    """

    ctx = extract_sample.generate_run_context(["2024-01-02"], "2024", "01")

    for i in range(7):
        extract_sample.handle_erroneous_line(f"bad-{i}", ctx)

    assert ctx["lines_unmatched"] == 7
    # Only the first five should be kept.
    assert ctx["unknown_or_offmonth_examples"] == [
        "bad-0",
        "bad-1",
        "bad-2",
        "bad-3",
        "bad-4",
    ]


def test_flush_spillover_seeds_overnight_reservoirs() -> None:
    """
    Ensure `flush_spillover` routes incoming spillover to overnight reservoirs.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If spillover items are not forwarded to the expected date/session
        combinations.

    Notes
    -----
    - Populates `run.spillover_in` with a few date-keyed candidates and asserts
      that the dummy reservoir manager receives matching "overnight" samples.
    """

    mgr = DummyReservoirManager()
    run = cast(
        RunData,
        DummyRunData(
            year="2024",
            month="01",
            nyse_cal=pd.DataFrame(index=pd.to_datetime(["2024-01-02", "2024-01-03"])),
        ),
    )
    run.spillover_in = {
        "2024-01-02": ["spill-a", "spill-b"],
        "2024-01-03": ["spill-c"],
    }

    extract_sample.flush_spillover(run, cast(ReservoirManager, mgr))

    assert mgr.calls == [
        ("spill-a", "2024-01-02", "overnight"),
        ("spill-b", "2024-01-02", "overnight"),
        ("spill-c", "2024-01-03", "overnight"),
    ]


@pytest.mark.parametrize(
    "ts_str, date_key, expected_intraday, expected_overnight, expected_call",
    [
        (
            "2024-01-02 16:00:00",  # between open and close
            "2024-01-02",
            {"2024-01-02": 1, "2024-01-03": 0},
            {"2024-01-02": 0, "2024-01-03": 0},
            ("intra-line", "2024-01-02", "intraday"),
        ),
        (
            "2024-01-02 13:00:00",  # before open
            "2024-01-02",
            {"2024-01-02": 0, "2024-01-03": 0},
            {"2024-01-02": 1, "2024-01-03": 0},
            ("pre-line", "2024-01-02", "overnight"),
        ),
        (
            "2024-01-02 22:00:00",  # after close → next trading day overnight
            "2024-01-02",
            {"2024-01-02": 0, "2024-01-03": 0},
            {"2024-01-02": 0, "2024-01-03": 1},
            ("post-line", "2024-01-03", "overnight"),
        ),
    ],
)
def test_handle_correct_line_routes_sessions_and_updates_counts(
    ts_str: str,
    date_key: str,
    expected_intraday: Dict[str, int],
    expected_overnight: Dict[str, int],
    expected_call: Tuple[str, str, str],
) -> None:
    """
    Check that `handle_correct_line` routes to the right session and updates
    per-day counters for intraday, pre-open, and after-close timestamps.

    Parameters
    ----------
    case_name : str
        Descriptive name for the parametrized scenario (used only for clarity).
    ts_str : str
        UTC timestamp string for the CC-NEWS line under test.
    date_key : str
        Trading-day key ('YYYY-MM-DD') passed into `handle_correct_line`.
    expected_intraday : dict[str, int]
        Expected `per_day_intraday_count` mapping after a single call.
    expected_overnight : dict[str, int]
        Expected `per_day_overnight_count` mapping after a single call.
    expected_call : tuple[str, str, str]
        Expected `(line, date_key, session)` triple recorded by the reservoir
        manager for this scenario.

    Raises
    ------
    AssertionError
        If per-day counts or routing do not match the parametrized expectations.

    Notes
    -----
    - Builds a two-day NYSE calendar (2024-01-02, 2024-01-03) with simple
      intraday sessions and corresponding `DataMaps`.
    - Each parametrized case:
        * calls `handle_correct_line` once,
        * checks that only the relevant per-day counter is incremented, and
        * verifies that the reservoir manager received a single call with the
          expected `(line, trading_day, session)` triple.
    """

    # Two trading days with simple intraday sessions.
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    open_times = pd.to_datetime(["2024-01-02 14:30:00", "2024-01-03 14:30:00"], utc=True)
    close_times = pd.to_datetime(["2024-01-02 21:00:00", "2024-01-03 21:00:00"], utc=True)
    nyse_cal = pd.DataFrame(
        {
            "session_open_utc": open_times,
            "session_close_utc": close_times,
            "is_trading_day": [True, True],
        },
        index=idx,
    )

    cap_dict = {
        "2024-01-02": (2, 1),  # intraday=2, overnight=1
        "2024-01-03": (1, 2),
    }
    session_dict: dict[str, tuple[int | None, int | None]] = {
        "2024-01-02": (
            int(open_times[0].timestamp()),
            int(close_times[0].timestamp()),
        ),
        "2024-01-03": (
            int(open_times[1].timestamp()),
            int(close_times[1].timestamp()),
        ),
    }
    valid_dates = set(cap_dict.keys())
    dm = data_maps_mod.DataMaps(
        cap_dict=cap_dict,
        session_dict=session_dict,
        overnight_fraction_dict={"2024-01-02": 0.3, "2024-01-03": 0.4},
        valid_date_set=valid_dates,
    )

    run = cast(RunData, DummyRunData(year="2024", month="01", nyse_cal=nyse_cal))
    mgr = DummyReservoirManager()
    ctx = extract_sample.generate_run_context(sorted(valid_dates), "2024", "01")

    utc_ts = pd.Timestamp(ts_str, tz="UTC")

    extract_sample.handle_correct_line(
        line=expected_call[0],
        utc_date=utc_ts,
        date_key=date_key,
        run_context=ctx,
        data_maps=dm,
        reservoir_manager=cast(ReservoirManager, mgr),
        run_data=run,
    )

    # Per-day intraday / overnight counters.
    assert ctx["per_day_intraday_count"] == expected_intraday
    assert ctx["per_day_overnight_count"] == expected_overnight

    # Exactly one routing call with the expected triple.
    assert mgr.calls == [expected_call]


def test_handle_correct_line_after_close_on_last_trading_day_goes_to_spillover() -> None:
    """
    Ensure after-close traffic on the last trading day is written to spillover.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If per-day counters are incremented within the current month or if
        the line is not recorded into `spillover_out`.

    Notes
    -----
    - Builds a single-day calendar (2024-01-31) and associated `DataMaps`.
    - Asserts that an after-close timestamp:
        * does not change any per-day intraday/overnight counters for the
          current slice, and
        * is appended to `run.spillover_out` under some future date key,
          representing the next trading day outside the monthly calendar.
    """

    idx = pd.to_datetime(["2024-01-31"])
    open_times = pd.to_datetime(["2024-01-31 14:30:00"], utc=True)
    close_times = pd.to_datetime(["2024-01-31 21:00:00"], utc=True)
    nyse_cal = pd.DataFrame(
        {
            "session_open_utc": open_times,
            "session_close_utc": close_times,
            "is_trading_day": [True],
        },
        index=idx,
    )

    cap_dict = {"2024-01-31": (2, 3)}
    session_dict: dict[str, tuple[int | None, int | None]] = {
        "2024-01-31": (
            int(open_times[0].timestamp()),
            int(close_times[0].timestamp()),
        ),
    }
    dm = data_maps_mod.DataMaps(
        cap_dict=cap_dict,
        session_dict=session_dict,
        overnight_fraction_dict={"2024-01-31": 0.5},
        valid_date_set=set(cap_dict.keys()),
    )

    run = cast(RunData, DummyRunData(year="2024", month="01", nyse_cal=nyse_cal))
    mgr = cast(ReservoirManager, DummyReservoirManager())
    ctx = extract_sample.generate_run_context(["2024-01-31"], "2024", "01")

    post_ts = pd.Timestamp("2024-01-31 22:00:00", tz="UTC")

    extract_sample.handle_correct_line(
        line="edge-post-line",
        utc_date=post_ts,
        date_key="2024-01-31",
        run_context=ctx,
        data_maps=dm,
        reservoir_manager=mgr,
        run_data=run,
    )

    # No intraday/overnight count within the current month should be incremented
    # for a trading day that does not exist in the slice.
    assert ctx["per_day_intraday_count"]["2024-01-31"] == 0
    assert ctx["per_day_overnight_count"]["2024-01-31"] == 0

    # Spillover should contain the line keyed by some next-date string.
    assert "edge-post-line" in [v for vals in run.spillover_out.values() for v in vals]


def test_fill_reservoirs_streams_s3_and_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Validate that `fill_reservoirs` streams S3 lines and delegates to helpers.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to stub out `boto3`, `extract_link_date`, and
        `handle_correct_line` to avoid real I/O and focus on control flow.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If counters are not updated correctly or if the stubbed
        `handle_correct_line` is called an unexpected number of times.

    Notes
    -----
    - Stubs:
        * S3 client to yield exactly two lines ("GOOD-LINE" and "BAD-LINE"),
        * `extract_link_date` to parse only "GOOD-LINE",
        * `handle_correct_line` to increment a local call counter.
    - Asserts that:
        * `lines_total == 2`,
        * one line matches and one is unmatched,
        * and the good line is passed through to `handle_correct_line`.
    """

    # Stub S3 stream.
    class DummyBody:
        def __enter__(self) -> "DummyBody":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
            return None

        def iter_lines(self):
            yield b"GOOD-LINE"
            yield b"BAD-LINE"

    class DummyS3Client:
        def get_object(self, Bucket: str, Key: str) -> Dict[str, Any]:
            return {"Body": DummyBody()}

    monkeypatch.setattr(
        extract_sample,
        "boto3",
        type("Boto3Stub", (), {"client": lambda *_a, **_k: DummyS3Client()}),
    )

    # Stub extract_link_date: GOOD-LINE → valid; BAD-LINE → None.
    def fake_extract(line: str, _pattern: re.Pattern[str]):
        if "GOOD-LINE" in line:
            ts = pd.Timestamp("2024-01-02 16:00:00", tz="UTC")
            return ts, "2024-01-02"
        return None

    monkeypatch.setattr(extract_sample, "extract_link_date", fake_extract)

    # Stub handle_correct_line to count calls.
    called: Dict[str, int] = {"count": 0}

    def fake_handle_correct_line(
        _line: str,
        _utc_date: pd.Timestamp,
        _date_key: str,
        _run_context: dict,
        _data_maps: data_maps_mod.DataMaps,
        _reservoir_manager: DummyReservoirManager,
        _run_data: DummyRunData,
    ) -> None:
        called["count"] += 1

    monkeypatch.setattr(extract_sample, "handle_correct_line", fake_handle_correct_line)

    # Minimal calendar and data_maps.
    idx = pd.to_datetime(["2024-01-02"])
    nyse_cal = pd.DataFrame(index=idx)
    run = cast(RunData, DummyRunData(year="2024", month="01", nyse_cal=nyse_cal))

    dm = data_maps_mod.DataMaps(
        cap_dict={"2024-01-02": (1, 1)},
        session_dict={"2024-01-02": (0, 1_000_000_000)},
        overnight_fraction_dict={"2024-01-02": 0.5},
        valid_date_set={"2024-01-02"},
    )
    mgr = cast(ReservoirManager, DummyReservoirManager())
    ctx = extract_sample.generate_run_context(["2024-01-02"], "2024", "01")

    extract_sample.fill_reservoirs(ctx, run, dm, mgr)

    # GOOD-LINE + BAD-LINE.
    assert ctx["lines_total"] == 2
    assert ctx["lines_matched"] == 1
    assert ctx["lines_unmatched"] == 1
    assert called["count"] == 1
    assert "BAD-LINE" in ctx["unknown_or_offmonth_examples"][0]


def test_extract_sample_orchestrates_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Confirm that `extract_sample.extract_sample` orchestrates the full pipeline.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to stub quota, data map construction, reservoir
        management, and streaming functions to isolate orchestration behavior.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If any of the core steps are not invoked with the expected arguments or
        if the final result does not come from the dummy reservoir manager.

    Notes
    -----
    - Stubs:
        * `compute_daily_caps` to echo the input calendar,
        * `build_data_maps` to return a simple `DataMaps` instance,
        * `ReservoirManager` to return a dummy manager,
        * `flush_spillover` and `fill_reservoirs` to set flags.
    - Asserts that:
        * each stub is called,
        * the correct `daily_cap` and cap dictionary are passed,
        * and the function returns exactly what the dummy manager’s
          `extract_sample_dict` provides.
    """

    # Minimal NYSE calendar.
    idx = pd.to_datetime(["2024-01-02"])
    nyse_cal = pd.DataFrame(
        {
            "overnight_fraction": [0.5],
            "session_open_utc": [pd.Timestamp("2024-01-02 14:30:00", tz="UTC")],
            "session_close_utc": [pd.Timestamp("2024-01-02 21:00:00", tz="UTC")],
            "is_trading_day": [True],
        },
        index=idx,
    )
    run = cast(RunData, DummyRunData(year="2024", month="01", nyse_cal=nyse_cal, daily_cap=10))

    # Capture flags/args.
    calls: Dict[str, Any] = {}

    # Stub compute_daily_caps to just return the frame.
    def fake_compute_daily_caps(
        daily_cap: int, cal: pd.DataFrame, _rng: np.random.Generator
    ) -> pd.DataFrame:
        calls["compute_daily_caps"] = {"daily_cap": daily_cap, "cal_index": cal.index.copy()}
        return cal

    # Stub build_data_maps to return a minimal DataMaps.
    def fake_build_data_maps(_cal: pd.DataFrame) -> data_maps_mod.DataMaps:
        calls["build_data_maps"] = True
        return data_maps_mod.DataMaps(
            cap_dict={"2024-01-02": (5, 5)},
            session_dict={"2024-01-02": (0, 1_000_000_000)},
            overnight_fraction_dict={"2024-01-02": 0.5},
            valid_date_set={"2024-01-02"},
        )

    # Dummy reservoir manager instance to be returned by the constructor stub.
    dummy_mgr = DummyReservoirManager(
        result={"2024-01-02": {"intraday": ["a"], "overnight": ["b"]}}
    )

    def fake_reservoir_manager(
        cap_dict: dict[str, Any], _rng: np.random.Generator
    ) -> DummyReservoirManager:
        calls["ReservoirManager"] = cap_dict
        return dummy_mgr

    # Stub flush_spillover and fill_reservoirs.
    def fake_flush_spillover(
        _run_data: DummyRunData, _reservoir_manager: DummyReservoirManager
    ) -> None:
        calls["flush_spillover"] = True

    def fake_fill_reservoirs(
        _run_context: dict,
        _run_data: DummyRunData,
        _data_maps: data_maps_mod.DataMaps,
        _reservoir_manager: DummyReservoirManager,
    ) -> None:
        calls["fill_reservoirs"] = True

    monkeypatch.setattr(extract_sample, "compute_daily_caps", fake_compute_daily_caps)
    monkeypatch.setattr(extract_sample, "build_data_maps", fake_build_data_maps)
    monkeypatch.setattr(extract_sample, "ReservoirManager", fake_reservoir_manager)
    monkeypatch.setattr(extract_sample, "flush_spillover", fake_flush_spillover)
    monkeypatch.setattr(extract_sample, "fill_reservoirs", fake_fill_reservoirs)

    result = extract_sample.extract_sample(run)

    assert calls["compute_daily_caps"]["daily_cap"] == 10
    assert "build_data_maps" in calls
    assert calls["ReservoirManager"] == {"2024-01-02": (5, 5)}
    assert calls["flush_spillover"] is True
    assert calls["fill_reservoirs"] is True

    # Final result is whatever dummy_mgr.extract_sample_dict returns.
    assert result == {"2024-01-02": {"intraday": ["a"], "overnight": ["b"]}}
