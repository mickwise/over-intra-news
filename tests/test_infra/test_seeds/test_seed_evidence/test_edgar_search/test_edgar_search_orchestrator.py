"""
Purpose
-------
Unit tests for the EDGAR search orchestrator (`edgar_search_orchestrator`).

Key behaviors
-------------
- Verify that `extract_wayback_candidates`:
  - queries Postgres via `connect_to_db`,
  - reconstructs `(ticker, ValidityWindow) -> [candidate_cik]` mappings, and
  - applies `SHORT_WINDOW_TICKERS_MAP` overrides.
- Check that `collect_evidence`:
  - iterates over the candidate dictionary,
  - skips pairs in `NO_ELIGIBLE_FORMS_PAIRS` and triples in `exclusion_set`,
  - and routes surviving work through `handle_evidence_collection(...)` using
    a single `requests.Session` instance.
- Ensure that `handle_evidence_collection`:
  - creates a fresh mapping-evidence buffer,
  - passes it into `fetch_edgar_evidence(...)`, and
  - delegates to `handle_collected_evidence(...)` with the captured start time.
- Confirm that `handle_collected_evidence`:
  - persists evidence and logs success when the buffer is non-empty, and
  - only emits a warning (and does not persist) when the buffer is empty.

Conventions
-----------
- Real network, EDGAR, and database calls are stubbed via `monkeypatch`.
- Logging is captured through a minimal `_DummyLogger` stand-in that mimics the
  subset of the `InfraLogger` API used by the orchestrator.

Downstream usage
----------------
Run with `pytest` as part of the CI suite. These tests act as executable
documentation for how the orchestrator interacts with Wayback candidates,
the exclusion set, and the evidence loader.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, cast

import pandas as pd
import pytest
import requests

from infra.logging.infra_logger import InfraLogger
from infra.seeds.seed_evidence.edgar_search import edgar_search_orchestrator as eso
from infra.seeds.seed_evidence.records.evidence_record import MappingEvidence
from infra.seeds.seed_evidence.seed_evidence_types import ValidityWindow


class _DummyLogger:
    """
    Purpose
    -------
    Minimal logger stand-in for EDGAR orchestrator tests.

    Key behaviors
    -------------
    - Collects `.info(...)`, `.warning(...)`, and `.debug(...)` calls.
    - Stores the message key and structured context for assertions.

    Parameters
    ----------
    None

    Attributes
    ----------
    infos : list[tuple[str, dict[str, Any]]]
        Recorded info-level messages.
    warnings : list[tuple[str, dict[str, Any]]]
        Recorded warning-level messages.
    debugs : list[tuple[str, dict[str, Any]]]
        Recorded debug-level messages.

    Notes
    -----
    - This intentionally mimics only the subset of `InfraLogger` used
      by `edgar_search_orchestrator`.
    """

    def __init__(self) -> None:
        self.infos: List[Tuple[str, Dict[str, Any]]] = []
        self.warnings: List[Tuple[str, Dict[str, Any]]] = []
        self.debugs: List[Tuple[str, Dict[str, Any]]] = []

    def info(self, msg: str, context: Dict[str, Any] | None = None) -> None:
        self.infos.append((msg, context or {}))

    def warning(self, msg: str, context: Dict[str, Any] | None = None) -> None:
        self.warnings.append((msg, context or {}))

    def debug(self, msg: str, context: Dict[str, Any] | None = None) -> None:
        self.debugs.append((msg, context or {}))


@dataclass
class _DummyDateRange:
    """
    Purpose
    -------
    Lightweight stand-in for `psycopg2.extras.DateRange` in unit tests.

    Key behaviors
    -------------
    - Exposes `.lower` and `.upper` attributes as `datetime.date` instances.
    - Behaves like an immutable container for range bounds.

    Parameters
    ----------
    lower : datetime.date
        Inclusive lower bound of the date range.
    upper : datetime.date
        Exclusive upper bound of the date range.

    Attributes
    ----------
    lower : datetime.date
        Stored start date.
    upper : datetime.date
        Stored end date.
    """

    lower: dt.date
    upper: dt.date


def test_extract_wayback_candidates_builds_dict_and_applies_short_window_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure `extract_wayback_candidates` builds the candidate dictionary and
    applies `SHORT_WINDOW_TICKERS_MAP` overrides.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub the database connection, cursor, and `SHORT_WINDOW_TICKERS_MAP`.

    Returns
    -------
    None
        The test passes if:
            - rows from the fake cursor are grouped into a `CandidateDict`
              keyed by `(ticker, ValidityWindow)`, and
            - entries listed in `SHORT_WINDOW_TICKERS_MAP` overwrite the
              discovered candidates for those keys.

    Raises
    ------
    AssertionError
        If grouping is incorrect or if the override map is not honored.
    """

    # Fake DB rows: two candidates for AAA in the same window, one for BBB.
    start_a = dt.date(2020, 1, 1)
    end_a = dt.date(2020, 2, 1)
    start_b = dt.date(2020, 3, 1)
    end_b = dt.date(2020, 4, 1)

    rows = [
        ("AAA", _DummyDateRange(start_a, end_a), "0000000001"),
        ("AAA", _DummyDateRange(start_a, end_a), "0000000002"),
        ("BBB", _DummyDateRange(start_b, end_b), "0000000003"),
    ]

    class DummyCursor:
        def __enter__(self) -> "DummyCursor":  # type: ignore[override]
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
            return None

        def execute(self, _query: str) -> None:
            return None

        def fetchall(self) -> List[Tuple[str, _DummyDateRange, str]]:
            return rows

    class DummyConn:
        def __enter__(self) -> "DummyConn":  # type: ignore[override]
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
            return None

        def cursor(self) -> DummyCursor:
            return DummyCursor()

    def fake_connect_to_db() -> DummyConn:
        return DummyConn()

    monkeypatch.setattr(eso, "connect_to_db", fake_connect_to_db)

    # SHORT_WINDOW_TICKERS_MAP should override the BBB window entirely.
    window_a: ValidityWindow = (
        pd.Timestamp(start_a, tz="UTC"),
        pd.Timestamp(end_a, tz="UTC"),
    )
    window_b: ValidityWindow = (
        pd.Timestamp(start_b, tz="UTC"),
        pd.Timestamp(end_b, tz="UTC"),
    )

    override_map = {
        ("BBB", window_b): ["OVERRIDE_CIK"],
    }
    monkeypatch.setattr(eso, "SHORT_WINDOW_TICKERS_MAP", override_map)

    result = eso.extract_wayback_candidates()

    # AAA: both discovered candidates retained.
    assert result[("AAA", window_a)] == ["0000000001", "0000000002"]

    # BBB: overridden by SHORT_WINDOW_TICKERS_MAP.
    assert result[("BBB", window_b)] == ["OVERRIDE_CIK"]


def test_collect_evidence_respects_skip_pairs_and_exclusion_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that `collect_evidence` honors `NO_ELIGIBLE_FORMS_PAIRS` and the
    caller-provided `exclusion_set`.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `extract_wayback_candidates`, `validity_window_to_str`,
        `requests.Session`, and `handle_evidence_collection`.

    Returns
    -------
    None
        The test passes if:
            - any `(ticker, window_key)` pair listed in `NO_ELIGIBLE_FORMS_PAIRS`
              is skipped entirely, and
            - for remaining pairs, only candidates *not* in `exclusion_set`
              are routed to `handle_evidence_collection(...)`.

    Raises
    ------
    AssertionError
        If a skipped pair or excluded triple is incorrectly processed.
    """

    # Two windows: AAA will be processed, BBB should be skipped via NO_ELIGIBLE_FORMS_PAIRS.
    window_a: ValidityWindow = (
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2020-02-01", tz="UTC"),
    )
    window_b: ValidityWindow = (
        pd.Timestamp("2020-03-01", tz="UTC"),
        pd.Timestamp("2020-04-01", tz="UTC"),
    )

    candidate_dict = {
        ("AAA", window_a): ["0000000001", "0000000002"],
        ("BBB", window_b): ["0000000003"],
    }

    def fake_extract_wayback_candidates() -> Dict[Tuple[str, ValidityWindow], List[str]]:
        return candidate_dict

    monkeypatch.setattr(eso, "extract_wayback_candidates", fake_extract_wayback_candidates)

    # Deterministic window keys.
    def fake_window_to_str(window: ValidityWindow) -> str:
        return f"{window[0].date()} to {window[1].date()}"

    monkeypatch.setattr(eso, "validity_window_to_str", fake_window_to_str)

    key_a = fake_window_to_str(window_a)
    key_b = fake_window_to_str(window_b)

    # BBB window should be globally skipped.
    monkeypatch.setattr(eso, "NO_ELIGIBLE_FORMS_PAIRS", {("BBB", key_b)})

    # Exclusion set skips AAA + candidate "0000000002" only.
    exclusion_set: set[tuple[str, str, str]] = {
        ("AAA", key_a, "0000000002"),
    }

    # Dummy Session so we do not hit the network and we can verify it is only created once.
    class DummySession:
        def __init__(self) -> None:
            self.get_calls: List[Dict[str, Any]] = []

        def __enter__(self) -> "DummySession":  # type: ignore[override]
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
            return None

    session_instances: List[DummySession] = []

    def fake_session_ctor() -> DummySession:
        instance = DummySession()
        session_instances.append(instance)
        return instance

    monkeypatch.setattr(eso.requests, "Session", fake_session_ctor)

    # Capture calls into handle_evidence_collection.
    handled: List[Dict[str, Any]] = []

    def fake_handle_evidence_collection(
        ticker: str,
        window: ValidityWindow,
        candidate: str,
        logger: InfraLogger,
        run_id: str,
        session: Any,
    ) -> None:
        handled.append(
            {
                "ticker": ticker,
                "window": window,
                "candidate": candidate,
                "run_id": run_id,
                "session": session,
            }
        )

    monkeypatch.setattr(eso, "handle_evidence_collection", fake_handle_evidence_collection)

    logger = cast(InfraLogger, _DummyLogger())

    eso.collect_evidence(
        exclusion_set=exclusion_set,
        run_id="RUN-XYZ",
        logger=logger,
    )

    # One Session per run.
    assert len(session_instances) == 1

    # BBB pair is skipped entirely; only AAA with candidate "0000000001" is processed.
    assert len(handled) == 1
    call = handled[0]
    assert call["ticker"] == "AAA"
    assert call["window"] == window_a
    assert call["candidate"] == "0000000001"
    assert call["run_id"] == "RUN-XYZ"


def test_handle_evidence_collection_populates_buffer_and_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Check that `handle_evidence_collection` populates the mapping-evidence buffer
    and delegates to `handle_collected_evidence(...)` with the captured start time.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub `pd.Timestamp.now`, `fetch_edgar_evidence`, and
        `handle_collected_evidence`.

    Returns
    -------
    None
        The test passes if:
            - `fetch_edgar_evidence(...)` receives the same buffer instance
              `handle_evidence_collection` created, and
            - `handle_collected_evidence(...)` is called once with that buffer
              and the expected start timestamp.

    Raises
    ------
    AssertionError
        If the buffer is not passed through correctly or if delegation does
        not occur as expected.
    """

    fixed_now = pd.Timestamp("2020-01-15T12:00:00Z")

    def fake_now(*args: Any, **kwargs: Any) -> pd.Timestamp:
        """
        Return a fixed Timestamp for tests regardless of arguments.

        Parameters
        ----------
        *args : Any
            Positional arguments (ignored).
        **kwargs : Any
            Keyword arguments (ignored).

        Returns
        -------
        pandas.Timestamp
            The fixed timestamp used for asserting `start_time`.
        """
        return fixed_now

    # Patch Timestamp.now used inside the module.
    monkeypatch.setattr(eso.pd.Timestamp, "now", staticmethod(fake_now))

    window: ValidityWindow = (
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2020-02-01", tz="UTC"),
    )

    logger = cast(InfraLogger, _DummyLogger())

    # Capture the buffer passed into fetch_edgar_evidence.
    received_buffers: List[Dict[str, MappingEvidence]] = []

    def fake_fetch_edgar_evidence(
        ticker: str,
        candidate: str,
        win: ValidityWindow,
        logger_param: InfraLogger,
        buffer: Dict[str, MappingEvidence],
        session: Any,
    ) -> None:
        # Simulate adding a single MappingEvidence record.
        received_buffers.append(buffer)
        buffer["evid-1"] = cast(MappingEvidence, object())

    monkeypatch.setattr(eso, "fetch_edgar_evidence", fake_fetch_edgar_evidence)

    handled_calls: List[Dict[str, Any]] = []

    def fake_handle_collected_evidence(
        ticker: str,
        win: ValidityWindow,
        candidate: str,
        run_id: str,
        start_time: pd.Timestamp,
        logger_param: InfraLogger,
        mapping_evidence_buffer: Dict[str, MappingEvidence],
    ) -> None:
        handled_calls.append(
            {
                "ticker": ticker,
                "window": win,
                "candidate": candidate,
                "run_id": run_id,
                "start_time": start_time,
                "buffer": mapping_evidence_buffer,
            }
        )

    monkeypatch.setattr(eso, "handle_collected_evidence", fake_handle_collected_evidence)

    session = object()

    eso.handle_evidence_collection(
        ticker="AAA",
        window=window,
        candidate="0000000001",
        logger=logger,
        run_id="RUN-123",
        session=cast(requests.Session, session),
    )

    # fetch_edgar_evidence should have seen exactly one buffer instance.
    assert len(received_buffers) == 1

    assert len(handled_calls) == 1
    call = handled_calls[0]
    assert call["ticker"] == "AAA"
    assert call["window"] == window
    assert call["candidate"] == "0000000001"
    assert call["run_id"] == "RUN-123"
    assert call["start_time"] == fixed_now

    # Same buffer object is passed through fetch → handle_collected_evidence.
    assert call["buffer"] is received_buffers[0]
    assert "evid-1" in call["buffer"]


@pytest.mark.parametrize("has_data", [True, False])
def test_handle_collected_evidence_persists_or_warns_based_on_buffer(
    has_data: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Validate that `handle_collected_evidence` either persists evidence or only
    logs a warning depending on whether the buffer is empty.

    Parameters
    ----------
    has_data : bool
        Parametrized flag indicating whether `mapping_evidence_buffer` should
        contain at least one `MappingEvidence` record.
    monkeypatch : pytest.MonkeyPatch
        Used to stub `persist_collected_data` so no real database work occurs.

    Returns
    -------
    None
        The test passes if:
            - when `has_data` is True, `persist_collected_data(...)` is called
              once with a list of evidence and INFO logs are recorded, and
            - when `has_data` is False, only a WARNING log is emitted and
              `persist_collected_data(...)` is never called.

    Raises
    ------
    AssertionError
        If persistence is triggered incorrectly or logging does not match the
        expected behavior for each case.
    """

    logger = _DummyLogger()
    window: ValidityWindow = (
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2020-02-01", tz="UTC"),
    )
    start_time = pd.Timestamp("2020-01-10T12:00:00Z")

    calls: List[Dict[str, Any]] = []

    def fake_persist_collected_data(
        run_id: str,
        ticker: str,
        win: ValidityWindow,
        candidate: str,
        start: pd.Timestamp,
        logger_param: InfraLogger,
        evidence_list: List[MappingEvidence],
    ) -> None:
        calls.append(
            {
                "run_id": run_id,
                "ticker": ticker,
                "window": win,
                "candidate": candidate,
                "start": start,
                "evidence_list": evidence_list,
            }
        )

    monkeypatch.setattr(eso, "persist_collected_data", fake_persist_collected_data)

    if has_data:
        mapping_evidence_buffer: Dict[str, MappingEvidence] = {
            "evid-1": cast(MappingEvidence, object())
        }
    else:
        mapping_evidence_buffer = {}

    eso.handle_collected_evidence(
        ticker="AAA",
        window=window,
        candidate="0000000001",
        run_id="RUN-XYZ",
        start_time=start_time,
        logger=cast(InfraLogger, logger),
        mapping_evidence_buffer=mapping_evidence_buffer,
    )

    if has_data:
        # One persistence call with a list derived from the buffer.
        assert len(calls) == 1
        call = calls[0]
        assert call["ticker"] == "AAA"
        assert call["window"] == window
        assert call["candidate"] == "0000000001"
        assert call["run_id"] == "RUN-XYZ"
        assert call["start"] == start_time
        assert len(call["evidence_list"]) == 1

        # Info logs should be present; warning should be empty.
        info_keys = {msg for msg, _ in logger.infos}
        assert "finished_collection_for_window" in info_keys
        assert "persisted_evidence_for_window" in info_keys
        assert not logger.warnings
    else:
        # No persistence call in the empty-buffer case.
        assert not calls

        # One warning about missing evidence.
        warning_keys = {msg for msg, _ in logger.warnings}
        assert "no_evidence_persisted_for_window" in warning_keys
        # No success-style info logs expected.
        assert not logger.infos
