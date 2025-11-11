"""
Purpose
-------
Unit tests for the Wayback candidate extraction helpers.

Key behaviors
-------------
- Verify that `build_wayback_url` formats a CDX query URL from a validity window.
- Validate that `collect_archive_urls_from_cdx_data`:
  - builds archive URLs using the CDX timestamp and original URL, and
  - parses timestamps into tz-aware UTC pandas.Timestamp objects.
- Ensure that `extract_valid_archive_urls`:
  - delegates to `make_request`,
  - returns an empty list and logs a warning when no data rows exist, and
  - returns sorted (url, timestamp) pairs when rows are present.
- Verify that `batch_extract_candidates_wayback`:
  - short-circuits on an empty batch with a warning,
  - computes the minimal covering window across batch validity_windows,
  - calls helpers for URL construction, CDX extraction, and snapshot extraction.
- Check that `extract_candidate_from_archive_batch`:
  - calls `make_request` with the expected arguments,
  - parses HTML via `etree.HTML`, and
  - delegates candidate extraction to `scrape_snapshot` with the correct
    arguments.

Conventions
-----------
- HTTP/network calls are never made; `make_request` is stubbed to return
  small in-memory responses with context-manager behavior.
- Logging is captured via a dummy logger object that mimics the `InfraLogger`
  interface for the methods used in this module.
- Tests focus on control flow, parameter threading, and transformation logic
  rather than HTML parsing, which is delegated to the snapshot scraper.

Downstream usage
----------------
Run with `pytest` as part of the CI suite. These tests serve as executable
documentation for the orchestration and transformation behavior of the
Wayback candidate extraction pipeline.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Tuple, cast

import pandas as pd
import pytest
import requests

from infra.logging.infra_logger import InfraLogger
from infra.seeds.seed_evidence.seed_evidence_types import ValidityWindow
from infra.seeds.seed_evidence.wayback import wayback_orchestrator


class _DummyLogger:
    """
    Purpose
    -------
    Minimal stand-in for `InfraLogger` used in Wayback extraction tests.

    Key behaviors
    -------------
    - Collects log messages for `.info(...)`, `.warning(...)`, and `.debug(...)`
      calls into dedicated lists.
    - Stores the structured `context` payloads for later assertions.

    Parameters
    ----------
    None

    Attributes
    ----------
    infos : list[tuple[str, dict[str, Any]]]
        Accumulates `(msg, context)` tuples for info-level logs.
    warnings : list[tuple[str, dict[str, Any]]]
        Accumulates `(msg, context)` tuples for warning-level logs.
    debugs : list[tuple[str, dict[str, Any]]]
        Accumulates `(msg, context)` tuples for debug-level logs.

    Notes
    -----
    - This class intentionally implements only the subset of the `InfraLogger`
      interface used in the Wayback module.
    """

    def __init__(self) -> None:
        self.infos: List[Tuple[str, Dict[str, Any]]] = []
        self.warnings: List[Tuple[str, Dict[str, Any]]] = []
        self.debugs: List[Tuple[str, Dict[str, Any]]] = []

    def info(self, msg: str, context: Dict[str, Any] | None = None) -> None:
        """
        Record an info-level message and context.

        Parameters
        ----------
        msg : str
            Message identifier or human-readable description.
        context : dict[str, Any] or None, optional
            Structured context associated with the message.

        Returns
        -------
        None
        """
        self.infos.append((msg, context or {}))

    def warning(self, msg: str, context: Dict[str, Any] | None = None) -> None:
        """
        Record a warning-level message and context.

        Parameters
        ----------
        msg : str
            Message identifier or description.
        context : dict[str, Any] or None, optional
            Structured context associated with the message.

        Returns
        -------
        None
        """
        self.warnings.append((msg, context or {}))

    def debug(self, msg: str, context: Dict[str, Any] | None = None) -> None:
        """
        Record a debug-level message and context.

        Parameters
        ----------
        msg : str
            Message identifier or description.
        context : dict[str, Any] or None, optional
            Structured context associated with the message.

        Returns
        -------
        None
        """
        self.debugs.append((msg, context or {}))


def test_build_wayback_url_formats_cdx_query() -> None:
    """
    Verify that `build_wayback_url` formats a CDX URL for the given window.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if:
            - the URL contains the expected base CDX endpoint,
            - includes the SNP_LISTS_WAYBACK_URL,
            - and encodes the from/to bounds as YYYYMMDDhhmmss.

    Raises
    ------
    AssertionError
        If the constructed URL is missing any expected component.
    """

    start = pd.Timestamp("2020-01-02 00:00:00", tz="UTC")
    end = pd.Timestamp("2020-01-10 12:34:56", tz="UTC")
    window: ValidityWindow = (start, end)

    url = wayback_orchestrator.build_wayback_url(window)

    assert "https://web.archive.org/cdx/search/cdx" in url
    assert f"url={wayback_orchestrator.SNP_LISTS_WAYBACK_URL}" in url
    assert f"from={start.strftime('%Y%m%d%H%M%S')}" in url
    assert f"to={end.strftime('%Y%m%d%H%M%S')}" in url
    assert "filter=statuscode:200" in url
    assert "filter=mimetype:text/html" in url
    assert "collapse=digest" in url


def test_collect_archive_urls_from_cdx_data_maps_rows() -> None:
    """
    Ensure `collect_archive_urls_from_cdx_data` builds URLs and parses timestamps.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if:
            - archive URLs are built from `timestamp` and `original`,
            - timestamps are tz-aware UTC pandas.Timestamp instances.

    Raises
    ------
    AssertionError
        If URLs or timestamps do not match expectations.
    """

    header = ["timestamp", "original", "mimetype", "statuscode"]
    row1 = ["20240102010101", "https://example.com/a", "text/html", "200"]
    row2 = ["20240103020202", "https://example.com/b", "text/html", "200"]
    data = [header, row1, row2]

    result = wayback_orchestrator.collect_archive_urls_from_cdx_data(data)

    assert len(result) == 2

    url1, ts1 = result[0]
    url2, ts2 = result[1]

    assert url1 == "https://web.archive.org/web/20240102010101id_/https://example.com/a"
    assert url2 == "https://web.archive.org/web/20240103020202id_/https://example.com/b"
    assert isinstance(ts1, pd.Timestamp) and ts1.tz is not None
    assert isinstance(ts2, pd.Timestamp) and ts2.tz is not None
    assert ts1 == pd.Timestamp(dt.datetime(2024, 1, 2, 1, 1, 1), tz="UTC")
    assert ts2 == pd.Timestamp(dt.datetime(2024, 1, 3, 2, 2, 2), tz="UTC")


def test_extract_valid_archive_urls_warns_on_no_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Validate that `extract_valid_archive_urls` warns and returns [] on empty payload.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to stub `make_request` and control the CDX response.

    Returns
    -------
    None
        The test passes if:
            - the returned list is empty, and
            - a warning is logged about missing Wayback archives.

    Raises
    ------
    AssertionError
        If the output is non-empty or no warning is recorded.
    """

    class DummyResponse:
        def __init__(self, data: Any) -> None:
            self._data = data

        def json(self) -> Any:
            return self._data

        def __enter__(self) -> "DummyResponse":  # type: ignore[override]
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
            return None

    def fake_make_request(url: str, *args: Any, **kwargs: Any) -> DummyResponse:
        # Only a header row, no data rows.
        data = [["timestamp", "original"]]
        return DummyResponse(data)

    monkeypatch.setattr(wayback_orchestrator, "make_request", fake_make_request)

    logger = _DummyLogger()
    session = requests.Session()

    result = wayback_orchestrator.extract_valid_archive_urls(
        wayback_url="https://fake.example/cdx",
        session=session,
        logger=cast(InfraLogger, logger),
    )

    assert not result
    assert logger.warnings, "Expected at least one warning log"
    msg, ctx = logger.warnings[0]
    assert msg == "no_wayback_archives_found"
    assert ctx["wayback_url"] == "https://fake.example/cdx"


def test_extract_valid_archive_urls_sorts_and_returns_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Check that `extract_valid_archive_urls` returns sorted (url, timestamp) pairs.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to stub `make_request` and feed a small CDX payload.

    Returns
    -------
    None
        The test passes if:
            - the output list is non-empty,
            - entries are sorted by capture timestamp ascending,
            - and no "no_valid_wayback_archives_found" warning is emitted.

    Raises
    ------
    AssertionError
        If sorting or logging behavior does not match expectations.
    """

    class DummyResponse:
        def __init__(self, data: Any) -> None:
            self._data = data

        def json(self) -> Any:
            return self._data

        def __enter__(self) -> "DummyResponse":  # type: ignore[override]
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
            return None

    header = ["timestamp", "original"]
    # Intentionally unsorted timestamps.
    row1 = ["20240103020202", "https://example.com/later"]
    row2 = ["20240102010101", "https://example.com/earlier"]
    data = [header, row1, row2]

    def fake_make_request(url: str, *args: Any, **kwargs: Any) -> DummyResponse:
        return DummyResponse(data)

    monkeypatch.setattr(wayback_orchestrator, "make_request", fake_make_request)

    logger = _DummyLogger()
    session = requests.Session()

    result = wayback_orchestrator.extract_valid_archive_urls(
        wayback_url="https://fake.example/cdx",
        session=session,
        logger=cast(InfraLogger, logger),
    )

    assert len(result) == 2

    # Should be sorted by timestamp ascending.
    (url_earlier, ts_earlier), (url_later, ts_later) = result
    assert ts_earlier < ts_later
    assert "earlier" in url_earlier
    assert "later" in url_later

    # No warning about "no_valid_wayback_archives_found" should be present.
    warning_ids = {msg for msg, _ctx in logger.warnings}
    assert "no_valid_wayback_archives_found" not in warning_ids


def test_batch_extract_candidates_wayback_returns_early_on_empty_batch() -> None:
    """
    Ensure `batch_extract_candidates_wayback` short-circuits cleanly on empty batch.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if:
            - an empty list is returned,
            - and a warning is logged about the empty batch.

    Raises
    ------
    AssertionError
        If the result is non-empty or the warning is not logged.
    """

    logger = _DummyLogger()
    session = requests.Session()

    result = wayback_orchestrator.batch_extract_candidates_wayback(
        batch=[],
        logger=cast(InfraLogger, logger),
        session=session,
    )

    assert not result
    assert logger.warnings, "Expected a warning for empty batch"
    msg, _ctx = logger.warnings[0]
    assert msg == "empty_batch_provided_to_wayback_extraction"


def test_batch_extract_candidates_wayback_processes_archive_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that `batch_extract_candidates_wayback` computes window and delegates.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to stub `build_wayback_url`, `extract_valid_archive_urls`,
        and `extract_candidate_from_archive_batch`.

    Returns
    -------
    None
        The test passes if:
            - the minimal covering validity window is passed to
              `build_wayback_url`,
            - `extract_valid_archive_urls` is called once,
            - `extract_candidate_from_archive_batch` is called once per archive,
            - and the returned candidate list reflects the mutations applied
              by the stubbed snapshot extractor.

    Raises
    ------
    AssertionError
        If any of the delegation or window computations are incorrect.
    """

    logger = _DummyLogger()
    session = requests.Session()

    # Two overlapping windows; min start + max end defines the covering window.
    window1: ValidityWindow = (
        pd.Timestamp("2020-01-02", tz="UTC"),
        pd.Timestamp("2020-01-10", tz="UTC"),
    )
    window2: ValidityWindow = (
        pd.Timestamp("2020-01-05", tz="UTC"),
        pd.Timestamp("2020-01-20", tz="UTC"),
    )

    batch = [("AAA", window1), ("BBB", window2)]

    recorded_window: dict[str, Any] = {}

    def fake_build_wayback_url(window: ValidityWindow) -> str:
        recorded_window["window"] = window
        return "https://fake.example/cdx"

    monkeypatch.setattr(wayback_orchestrator, "build_wayback_url", fake_build_wayback_url)

    archive_list: List[Tuple[str, pd.Timestamp]] = [
        (
            "https://web.archive.org/web/20240102010101id_/page",
            pd.Timestamp("2024-01-02", tz="UTC"),
        ),
        (
            "https://web.archive.org/web/20240103020202id_/page",
            pd.Timestamp("2024-01-03", tz="UTC"),
        ),
    ]

    def fake_extract_valid_archive_urls(
        wayback_url: str,
        session_param: requests.Session,
        logger_param: InfraLogger,
    ) -> List[Tuple[str, pd.Timestamp]]:
        assert wayback_url == "https://fake.example/cdx"
        assert session_param is session
        return archive_list

    monkeypatch.setattr(
        wayback_orchestrator,
        "extract_valid_archive_urls",
        fake_extract_valid_archive_urls,
    )

    calls: List[Dict[str, Any]] = []

    def fake_extract_candidate_from_archive_batch(
        archive_url: str,
        snapshot_date: pd.Timestamp,
        candidate_list: List[Any],
        seen_candidates: Any,
        batch_param: Any,
        session_param: Any,
        logger_param: Any,
    ) -> None:
        calls.append(
            {
                "archive_url": archive_url,
                "snapshot_date": snapshot_date,
                "candidate_list": candidate_list,
                "batch": batch_param,
            }
        )
        # Simulate adding one candidate per archive.
        candidate_list.append(f"candidate-from-{archive_url}")

    monkeypatch.setattr(
        wayback_orchestrator,
        "extract_candidate_from_archive_batch",
        fake_extract_candidate_from_archive_batch,
    )

    result = cast(
        List[str],
        wayback_orchestrator.batch_extract_candidates_wayback(
            batch=batch, logger=cast(InfraLogger, logger), session=session
        ),
    )

    # Covering window: min(start1, start2), max(end1, end2).
    expected_window: ValidityWindow = (window1[0], window2[1])
    assert recorded_window["window"] == expected_window

    # One call per archive URL.
    assert len(calls) == len(archive_list)
    assert len(result) == len(archive_list)
    assert all(r.startswith("candidate-from-") for r in result)

    called_urls = {c["archive_url"] for c in calls}
    expected_urls = {u for u, _ts in archive_list}
    assert called_urls == expected_urls


def test_extract_candidate_from_archive_batch_fetches_html_and_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure `extract_candidate_from_archive_batch` fetches HTML and calls scraper.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to stub `make_request`, `etree.HTML`, and `scrape_snapshot`.

    Returns
    -------
    None
        The test passes if:
            - `make_request` is invoked with `expect_json=False`,
            - the HTML root passed to `scrape_snapshot` comes from `etree.HTML`,
            - and the batch, candidate_list, and other parameters are threaded
              through unchanged.

    Raises
    ------
    AssertionError
        If any of the delegation or wiring expectations are violated.
    """

    logger = _DummyLogger()
    session = requests.Session()

    archive_url = "https://web.archive.org/web/20240102010101id_/page"
    snapshot_date = pd.Timestamp("2024-01-02", tz="UTC")
    candidate_list: List[Any] = []
    seen_candidates: Dict[Any, Any] = {}
    batch: List[Any] = [("AAA", (snapshot_date, snapshot_date + pd.Timedelta(days=1)))]

    recorded_request: Dict[str, Any] = {}

    class DummyResponse:
        def __init__(self, text: str) -> None:
            self.text = text

        def __enter__(self) -> "DummyResponse":  # type: ignore[override]
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
            return None

    def fake_make_request(
        url: str,
        expect_json: bool,
        sleep_time: float,
        session: requests.Session,
    ) -> DummyResponse:
        recorded_request["url"] = url
        recorded_request["expect_json"] = expect_json
        recorded_request["sleep_time"] = sleep_time
        recorded_request["session"] = session
        return DummyResponse("<html><body>dummy</body></html>")

    monkeypatch.setattr(wayback_orchestrator, "make_request", fake_make_request)

    recorded_html: Dict[str, Any] = {}

    def fake_html_parser(html_text: str) -> Any:
        recorded_html["html_text"] = html_text
        return {"parsed": True}

    monkeypatch.setattr(wayback_orchestrator.etree, "HTML", fake_html_parser)

    recorded_scrape: Dict[str, Any] = {}

    def fake_scrape_snapshot(
        batch_param: Any,
        html_root: Any,
        seen_candidates_param: Any,
        candidate_list_param: Any,
        archive_url_param: str,
        snapshot_date_param: pd.Timestamp,
        logger_param: Any,
    ) -> None:
        recorded_scrape["batch"] = batch_param
        recorded_scrape["html_root"] = html_root
        recorded_scrape["seen_candidates"] = seen_candidates_param
        recorded_scrape["candidate_list"] = candidate_list_param
        recorded_scrape["archive_url"] = archive_url_param
        recorded_scrape["snapshot_date"] = snapshot_date_param

    monkeypatch.setattr(wayback_orchestrator, "scrape_snapshot", fake_scrape_snapshot)

    wayback_orchestrator.extract_candidate_from_archive_batch(
        archive_url=archive_url,
        snapshot_date=snapshot_date,
        candidate_list=candidate_list,
        seen_candidates=seen_candidates,
        batch=batch,
        session=session,
        logger=cast(InfraLogger, logger),
    )

    # Request parameters should reflect HTML fetch behavior.
    assert recorded_request["url"] == archive_url
    assert recorded_request["expect_json"] is False
    assert recorded_request["session"] is session

    # HTML parser should see the response text.
    assert recorded_html["html_text"] == "<html><body>dummy</body></html>"

    # Scraper should receive the parsed root and original parameters.
    assert recorded_scrape["batch"] is batch
    assert recorded_scrape["html_root"] == {"parsed": True}
    assert recorded_scrape["seen_candidates"] is seen_candidates
    assert recorded_scrape["candidate_list"] is candidate_list
    assert recorded_scrape["archive_url"] == archive_url
    assert recorded_scrape["snapshot_date"] == snapshot_date
