"""
Purpose
-------
Unit tests for seed_evidence.

Key behaviors
-------------
- Validate ticker normalization and extraction helpers used in window plumbing.
- Check correctness of validity window construction from daily membership sets.
- Ensure alias rewriting respects date-bounded mappings and logs rewrites.
- Verify extraction of per-ticker validity windows wires helpers correctly.
- Confirm Wayback seeding flattens ticker-window maps and dispatches to
  the batch Wayback seed_evidence and loader.
- Exercise run setup logic for exclusion set construction from DB rows.
- Sanity-check `run_edgar_harvest` orchestration and its handling of
  optional Wayback seeding and EDGAR scraping flags.

Conventions
-----------
- External I/O (DB, HTTP, filesystem) is stubbed via `monkeypatch` so tests
  stay in-memory and deterministic.
- Only control-flow and data-shaping behaviors are tested; actual EDGAR or
  Wayback semantics are delegated to their own modules.

Downstream usage
----------------
Run with `pytest` as part of the CI suite. These tests are intended to be
lightweight but precise, documenting how the seed_evidence composes helpers
and passes arguments downstream.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Tuple, cast

import pandas as pd
import pytest

from infra.logging.infra_logger import InfraLogger
from infra.seeds.seed_evidence import seed_evidence
from infra.seeds.seed_evidence.seed_evidence_types import ValidityWindows


class _DummyLogger:
    """
    Purpose
    -------
    Minimal stand-in for `InfraLogger` in seed_evidence tests.

    Key behaviors
    -------------
    - Captures `.info(...)` and `.debug(...)` calls with their contexts.
    - Exposes the captured messages for assertions on logging side effects.

    Parameters
    ----------
    None

    Attributes
    ----------
    infos : list[tuple[str, dict[str, Any]]]
        Recorded info-level messages.
    debugs : list[tuple[str, dict[str, Any]]]
        Recorded debug-level messages.

    Notes
    -----
    - Only the subset of the real logger API used in this module is
      implemented. Additional methods can be added as needed.
    """

    def __init__(self) -> None:
        self.infos: List[Tuple[str, Dict[str, Any]]] = []
        self.debugs: List[Tuple[str, Dict[str, Any]]] = []

    def info(self, msg: str, context: Dict[str, Any] | None = None) -> None:
        self.infos.append((msg, context or {}))

    def debug(self, msg: str, context: Dict[str, Any] | None = None) -> None:
        self.debugs.append((msg, context or {}))


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("aapl", "AAPL"),
        (" brk.b ", "BRK-B"),
        ("ms/ft", "MS-FT"),
        (" foo bar ", "FOOBAR"),
    ],
)
def test_normalize_tickers_applies_all_transformations(raw: str, expected: str) -> None:
    """
    Check that `normalize_tickers` applies uppercasing, trimming, whitespace
    removal, and class-separator mapping.

    Parameters
    ----------
    raw : str
        Raw ticker string input to the normalization helper.
    expected : str
        The canonical EDGAR-style ticker expected after normalization.

    Returns
    -------
    None
        The test passes if the resulting single-element Series equals the
        expected normalized ticker.

    Raises
    ------
    AssertionError
        If the normalized ticker does not match the expected value.

    Notes
    -----
    - This parametrized test exercises multiple transformation combinations
      without duplicating test bodies.
    """

    s = pd.Series([raw])
    result = seed_evidence.normalize_tickers(s)
    assert result.iloc[0] == expected


def test_extract_valid_tickers_unions_all_daily_members() -> None:
    """
    Ensure `extract_valid_tickers` returns the union of all tickers across days.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if the resulting set contains exactly the tickers
        present in any of the input day-level lists.

    Raises
    ------
    AssertionError
        If any ticker is missing from the result or an unexpected ticker
        is included.

    Notes
    -----
    - This helper is used to initialize per-ticker validity window storage,
      so correctness of the union is important.
    """

    tickers = pd.Series(
        [
            ["AAA", "BBB"],
            ["BBB", "CCC"],
        ]
    )

    result = seed_evidence.extract_valid_tickers(tickers)

    assert result == {"AAA", "BBB", "CCC"}


def test_construct_validity_windows_builds_join_leave_episodes() -> None:
    """
    Verify `construct_validity_windows` detects joins, leaves, and episode ends.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if:
            - when a ticker appears then disappears, its window ends on
              the first date it is absent, and
            - when a ticker disappears, its last window entry is updated
              rather than a new one being created.

    Raises
    ------
    AssertionError
        If episode boundaries do not align with membership joins/leaves.

    Notes
    -----
    - This test covers the "closed" episode case where a ticker appears and
      later leaves before the run end date.
    """

    dates = pd.to_datetime(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
        utc=True,
    )
    snp_membership_windows = pd.DataFrame(
        {
            "date": dates,
            "tickers": [
                ["AAA"],  # day 1: AAA
                ["AAA", "BBB"],  # day 2: AAA, BBB (B joins)
                ["BBB"],  # day 3: BBB (AAA leaves)
                [],  # day 4: none (BBB leaves)
            ],
        }
    )

    valid_tickers = {"AAA", "BBB"}
    end_date = pd.Timestamp("2020-01-04", tz="UTC")

    windows = seed_evidence.construct_validity_windows(
        valid_tickers=valid_tickers,
        snp_membership_windows=snp_membership_windows,
        end_date=end_date,
    )

    # AAA: [2020-01-01, 2020-01-03)
    aaa_windows: ValidityWindows = windows["AAA"]
    assert len(aaa_windows) == 1
    aaa_start, aaa_end = aaa_windows[0]
    assert aaa_start == pd.Timestamp("2020-01-01", tz="UTC")
    assert aaa_end == pd.Timestamp("2020-01-03", tz="UTC")

    # BBB: [2020-01-02, 2020-01-04)
    bbb_windows: ValidityWindows = windows["BBB"]
    assert len(bbb_windows) == 1
    bbb_start, bbb_end = bbb_windows[0]
    assert bbb_start == pd.Timestamp("2020-01-02", tz="UTC")
    assert bbb_end == pd.Timestamp("2020-01-04", tz="UTC")


def test_construct_validity_windows_closes_open_windows_at_end_date_plus_one() -> None:
    """
    Confirm `construct_validity_windows` closes open episodes at end_date + 1 day.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if tickers that remain members on the final day
        have their validity windows closed at end_date + 1 day, preserving
        half-open [start, end) semantics.

    Raises
    ------
    AssertionError
        If open windows remain unclosed or are closed at the wrong boundary.

    Notes
    -----
    - This test covers the "tail" behavior for tickers that are still in
      the index at the final membership date.
    """

    dates = pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True)
    snp_membership_windows = pd.DataFrame(
        {
            "date": dates,
            "tickers": [
                ["ZZZ"],
                ["ZZZ"],  # never leaves
            ],
        }
    )

    valid_tickers = {"ZZZ"}
    end_date = pd.Timestamp("2020-01-02", tz="UTC")

    windows = seed_evidence.construct_validity_windows(
        valid_tickers=valid_tickers,
        snp_membership_windows=snp_membership_windows,
        end_date=end_date,
    )

    zzz_windows = windows["ZZZ"]
    assert len(zzz_windows) == 1
    start, end = zzz_windows[0]
    assert start == pd.Timestamp("2020-01-01", tz="UTC")
    assert end == pd.Timestamp("2020-01-03", tz="UTC")  # end_date + 1 day


def test_rewrite_aliased_tickers_applies_alias_within_date_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure `rewrite_aliased_tickers` uses alias rows only within the configured
    [start_date, end_date) interval and logs rewrites.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `pd.read_csv` with an in-memory alias mapping.

    Returns
    -------
    None
        The test passes if tickers within the alias date window are rewritten
        to the canonical symbol, while out-of-window rows retain the original
        ticker, and a debug log is emitted for applied rewrites.

    Raises
    ------
    AssertionError
        If alias application does not respect date bounds or no debug log is
        recorded when rewrites occur.

    Notes
    -----
    - This test does not touch the real filesystem; it replaces the alias CSV
      with an in-memory DataFrame via monkeypatch.
    """

    alias_df = pd.DataFrame(
        {
            "alias": ["AABA"],
            "canonical": ["YHOO"],
            "start_date": ["2020-01-01"],
            "end_date": ["2020-02-01"],
        }
    )

    def fake_read_csv(path: str, comment: str = "#") -> pd.DataFrame:  # type: ignore[override]
        assert "symbol_aliases" in path
        return alias_df

    monkeypatch.setattr(seed_evidence.pd, "read_csv", fake_read_csv)

    logger = _DummyLogger()

    tickers_and_dates = pd.DataFrame(
        {
            "tickers": ["AABA", "AABA", "MSFT"],
            "date": pd.to_datetime(
                ["2019-12-31", "2020-01-15", "2020-01-15"],
                utc=True,
            ),
        }
    )

    result = seed_evidence.rewrite_aliased_tickers(
        tickers_and_dates.copy(), cast(InfraLogger, logger)
    )

    assert list(result) == ["AABA", "YHOO", "MSFT"]

    # One row rewritten, one distinct alias.
    assert logger.debugs, "Expected a debug log for alias rewrites"
    msg, ctx = logger.debugs[0]
    assert msg == "alias_rewrites_applied"
    assert ctx["rewritten_rows"] == 1
    assert ctx["distinct_aliases"] == 1


def test_extract_ticker_validity_windows_normalizes_and_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify `extract_ticker_validity_windows` normalizes, explodes, filters
    tickers, and wires through to `construct_validity_windows`.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `rewrite_aliased_tickers` and `construct_validity_windows`
        so the test can focus on argument shaping and logging.

    Returns
    -------
    None
        The test passes if:
            - raw comma-separated tickers are normalized and empty entries
              are dropped,
            - `extract_valid_tickers` yields the expected ticker set, and
            - `construct_validity_windows` is called with that set and the
              grouped-by-date frame.

    Raises
    ------
    AssertionError
        If the derived valid ticker set or the grouped frame passed to
        `construct_validity_windows` does not match expectations.

    Notes
    -----
    - `rewrite_aliased_tickers` is stubbed to be an identity transformation
      so alias behavior does not affect this test.
    """

    logger = _DummyLogger()

    # One row with two tickers and a dangling comma; one completely empty.
    snp_membership_windows = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
            "tickers": [" aapl , brk.b , ", ""],
        }
    )

    def fake_rewrite_aliased_tickers(df: pd.DataFrame, _logger: InfraLogger) -> pd.Series:
        return df["tickers"]

    captured_args: Dict[str, Any] = {}

    def fake_construct_validity_windows(
        valid_tickers: seed_evidence.ValidTickers,
        grouped: pd.DataFrame,
        end_date: pd.Timestamp,
    ) -> dict[str, ValidityWindows]:
        captured_args["valid_tickers"] = set(valid_tickers)
        captured_args["grouped"] = grouped.copy()
        captured_args["end_date"] = end_date
        return {
            "SENTINEL": [  # sentinel result
                (pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-01-02", tz="UTC"))
            ]
        }

    monkeypatch.setattr(seed_evidence, "rewrite_aliased_tickers", fake_rewrite_aliased_tickers)
    monkeypatch.setattr(
        seed_evidence, "construct_validity_windows", fake_construct_validity_windows
    )

    end_date = pd.Timestamp("2020-01-31", tz="UTC")

    result = seed_evidence.extract_ticker_validity_windows(
        snp_membership_windows=snp_membership_windows,
        end_date=end_date,
        logger=cast(InfraLogger, logger),
    )

    # Returned value should be whatever construct_validity_windows returned.
    assert "SENTINEL" in result

    # Valid tickers: {AAPL, BRK-B}; empty entries dropped.
    assert captured_args["valid_tickers"] == {"AAPL", "BRK-B"}

    grouped = captured_args["grouped"]
    assert list(grouped["date"]) == [pd.Timestamp("2020-01-01", tz="UTC")]
    assert grouped["tickers"].iloc[0] == ["AAPL", "BRK-B"]

    # Logger should have emitted valid_tickers_extracted with count=2.
    info_msgs = {m for m, _ctx in logger.infos}
    assert "valid_tickers_extracted" in info_msgs


def test_seed_wayback_table_flattens_and_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify `seed_wayback_table` flattens per-ticker windows into a batch and
    dispatches to Wayback seed_evidence and loader.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `requests.Session`, `batch_extract_candidates_wayback`,
        and `load_wayback_candidates`.

    Returns
    -------
    None
        The test passes if:
            - the batch passed to `batch_extract_candidates_wayback` contains
              all (ticker, window) pairs from the map, and
            - `load_wayback_candidates` receives the candidates returned by
              the batch extractor.

    Raises
    ------
    AssertionError
        If any window is missing from the batch or the loader is not called
        with the expected candidate list.

    Notes
    -----
    - HTTP I/O is fully stubbed via a dummy Session and a fake batch extractor.
    """

    ticker_validity_windows: dict[str, ValidityWindows] = {
        "AAA": [
            (
                pd.Timestamp("2020-01-01", tz="UTC"),
                pd.Timestamp("2020-02-01", tz="UTC"),
            )
        ],
        "BBB": [
            (
                pd.Timestamp("2020-03-01", tz="UTC"),
                pd.Timestamp("2020-04-01", tz="UTC"),
            )
        ],
    }

    captured_batch: List[Tuple[str, Tuple[pd.Timestamp, pd.Timestamp]]] = []
    fake_candidates = ["CANDIDATE1", "CANDIDATE2"]

    class DummySession:
        def __enter__(self) -> "DummySession":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def fake_session_ctor() -> DummySession:
        return DummySession()

    def fake_batch_extract(batch, logger, session):
        captured_batch.extend(batch)
        return list(fake_candidates)

    loaded: Dict[str, List[Any]] = {}

    def fake_load_wayback_candidates(candidates: List[Any]) -> None:
        loaded["candidates"] = list(candidates)

    monkeypatch.setattr(seed_evidence.requests, "Session", fake_session_ctor)  # type: ignore[attr-defined]
    monkeypatch.setattr(seed_evidence, "batch_extract_candidates_wayback", fake_batch_extract)
    monkeypatch.setattr(seed_evidence, "load_wayback_candidates", fake_load_wayback_candidates)

    logger = _DummyLogger()

    seed_evidence.seed_wayback_table(
        ticker_validity_windows=ticker_validity_windows,
        logger=cast(InfraLogger, logger),
    )

    # Batch should have one entry per validity window.
    assert set(captured_batch) == {
        (
            "AAA",
            (
                pd.Timestamp("2020-01-01", tz="UTC"),
                pd.Timestamp("2020-02-01", tz="UTC"),
            ),
        ),
        (
            "BBB",
            (
                pd.Timestamp("2020-03-01", tz="UTC"),
                pd.Timestamp("2020-04-01", tz="UTC"),
            ),
        ),
    }

    assert loaded["candidates"] == fake_candidates


def test_set_up_run_builds_exclusion_set_from_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Ensure `set_up_run` composes the exclusion set from registry and manual
    override rows.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `connect_to_db` and `validity_window_to_str` so the test
        can supply fake DateRange-like objects and control the window key.

    Returns
    -------
    None
        The test passes if the exclusion set contains the expected tuples
        derived from both edgar_run_registry and manual adjudication rows.

    Raises
    ------
    AssertionError
        If any expected triple is missing from the exclusion set or an
        unexpected triple is present.

    Notes
    -----
    - A minimal DateRange stand-in is used; only `.lower` and `.upper`
      attributes are required by the implementation.
    """

    class DummyDateRange:
        def __init__(self, lower: dt.date | None, upper: dt.date | None) -> None:
            self.lower = lower
            self.upper = upper

    # Stand-in results: one from registry, one from manual adjudication.
    registry_rows = [
        ("AAA", DummyDateRange(dt.date(2020, 1, 1), dt.date(2020, 2, 1)), "0000000001"),
    ]
    manual_rows = [
        ("BBB", DummyDateRange(dt.date(2020, 3, 1), dt.date(2020, 4, 1)), "0000000002"),
    ]

    class DummyCursor:
        def __init__(self) -> None:
            self._step = 0

        def execute(self, _query: str) -> None:
            self._step += 1

        def fetchall(self) -> list[tuple[str, DummyDateRange, str]]:
            if self._step == 1:
                return registry_rows
            return manual_rows

        def __enter__(self) -> "DummyCursor":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    class DummyConn:
        def cursor(self) -> DummyCursor:
            return DummyCursor()

        def __enter__(self) -> "DummyConn":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def fake_connect_to_db() -> DummyConn:
        return DummyConn()

    def fake_validity_window_to_str(window: Tuple[pd.Timestamp, pd.Timestamp]) -> str:
        start, end = window
        return f"{start.date()} to {end.date()}"

    monkeypatch.setattr(seed_evidence, "connect_to_db", fake_connect_to_db)
    monkeypatch.setattr(seed_evidence, "validity_window_to_str", fake_validity_window_to_str)

    run_id, exclusion_set = seed_evidence.set_up_run()

    assert isinstance(run_id, str) and run_id

    expected = {
        ("AAA", "2020-01-01 to 2020-02-01", "0000000001"),
        ("BBB", "2020-03-01 to 2020-04-01", "0000000002"),
    }
    assert exclusion_set == expected


def test_run_edgar_harvest_honors_flags_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Check that `run_edgar_harvest` reads END_DATE, sets up the run, and
    conditionally invokes Wayback seeding and EDGAR collection.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub environment access, logger initialization, and all
        side-effectful helpers (run setup, membership extraction, etc.).

    Returns
    -------
    None
        The test passes if:
            - a missing END_DATE raises ValueError, and
            - when END_DATE is provided and both flags are True, the function
              calls `seed_wayback_table` and `collect_evidence` with the
              expected arguments.

    Raises
    ------
    AssertionError
        If flags are not respected or downstream calls receive incorrect
        parameters.

    Notes
    -----
    - This test focuses on control flow and wiring; EDGAR and Wayback
      internals are fully stubbed.
    """

    # ---- Case 1: force the "missing/bad END_DATE" path safely ----
    # Make load_dotenv a no-op so it can’t re-inject END_DATE from .env
    monkeypatch.setattr(seed_evidence, "load_dotenv", lambda: None)

    # Either delete END_DATE or set an invalid one; both will cause str_to_timestamp() to fail.
    # Deleting is fine now that load_dotenv is neutered:
    monkeypatch.delenv("END_DATE", raising=False)

    with pytest.raises(ValueError):
        seed_evidence.run_edgar_harvest(logger_level="INFO", seed_wayback=True, scrape_edgar=True)

    # ---- Case 2: happy path with both flags enabled ----
    monkeypatch.setenv("END_DATE", "2020-01-31")

    def fake_initialize_logger(component_name: str, level: str, run_meta: dict[str, Any]):
        return cast(InfraLogger, _DummyLogger())

    monkeypatch.setattr(seed_evidence, "initialize_logger", fake_initialize_logger)

    def fake_set_up_run() -> tuple[str, set[tuple[str, str, str]]]:
        return "RUN-ID-123", {("AAA", "WINDOW", "0000000001")}

    monkeypatch.setattr(seed_evidence, "set_up_run", fake_set_up_run)

    def fake_extract_snp_membership_windows(logger: InfraLogger) -> pd.DataFrame:
        return pd.DataFrame({"date": [], "tickers": []})

    monkeypatch.setattr(
        seed_evidence, "extract_snp_membership_windows", fake_extract_snp_membership_windows
    )

    ticker_windows = {
        "AAA": [(pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-02-01", tz="UTC"))]
    }

    def fake_extract_ticker_validity_windows(_df, _end, _logger):
        return ticker_windows

    monkeypatch.setattr(
        seed_evidence, "extract_ticker_validity_windows", fake_extract_ticker_validity_windows
    )

    seed_calls: Dict[str, Any] = {}
    collect_calls: Dict[str, Any] = {}

    def fake_seed_wayback_table(tvw, logger):  # no network
        seed_calls["tvw"] = tvw
        seed_calls["logger"] = logger

    monkeypatch.setattr(seed_evidence, "seed_wayback_table", fake_seed_wayback_table)

    def fake_collect_evidence(exclusion_set, run_id, logger):  # no EDGAR scraping
        collect_calls["exclusion_set"] = exclusion_set
        collect_calls["run_id"] = run_id
        collect_calls["logger"] = logger

    monkeypatch.setattr(seed_evidence, "collect_evidence", fake_collect_evidence)

    seed_evidence.run_edgar_harvest(logger_level="DEBUG", seed_wayback=True, scrape_edgar=True)

    assert seed_calls["tvw"] == ticker_windows
    assert collect_calls["exclusion_set"] == {("AAA", "WINDOW", "0000000001")}
    assert collect_calls["run_id"] == "RUN-ID-123"
