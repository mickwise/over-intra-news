"""
Purpose
-------
Unit tests for `infra.utils.db_utils`.

Key behaviors
-------------
- Verify that `load_into_table`:
  - collects rows into fixed-size batches using `BATCH_SIZE`,
  - calls `flush_values_batch` for each full batch and one final partial batch.
- Verify that `flush_values_batch`:
  - acquires a cursor from the connection,
  - calls `execute_values` with the given query and batch.
- Verify that `str_to_timestamp`:
  - parses ISO "YYYY-MM-DD" strings into UTC `Timestamp`s,
  - raises `ValueError` on malformed input.
- Verify that `process_chunk`:
  - builds the expected URL,
  - selects the requested columns,
  - applies an optional `processor` function.

Conventions
-----------
- All database interactions are mocked; no real connections or network calls.
- `pandas.read_json` is monkeypatched in `process_chunk` tests to avoid HTTP.
- Tests are written with pytest and use the `monkeypatch` fixture for stubbing.

Downstream usage
----------------
- Run via `pytest` as part of the CI suite.
- Serves as executable documentation for how db_utils helpers are expected
  to behave with respect to batching, timestamp parsing, and JSON fetching.
"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
import psycopg2
import pytest

from infra.utils import db_utils


def test_load_into_table_batches_and_flushes(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that `load_into_table` batches rows and flushes full and partial batches.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to replace `flush_values_batch` with a stub that
        records calls instead of hitting a real database.

    Returns
    -------
    None
        The test passes if:
            - `flush_values_batch` is called three times for 2 * BATCH_SIZE + 5 rows,
            - the first two batches have size `BATCH_SIZE`,
            - the final batch has size 5,
            - and the concatenated flushed rows equal the original input sequence.

    Raises
    ------
    AssertionError
        If the number of flushes, batch sizes, or flattened rows do not match
        the expected values.

    Notes
    -----
    - Uses a simple row sequence `[(0,), (1,), ...]` to make boundary checks
      straightforward.
    - Ensures that `load_into_table` does not drop or duplicate rows when
      transitioning between full and partial batches.
    """

    calls: list[dict[str, Any]] = []

    def fake_flush_values_batch(conn: Any, batch: list[tuple], query: str) -> None:
        # Record a shallow copy so later mutations don't affect our assertion.
        calls.append({"conn": conn, "batch": list(batch), "query": query})

    monkeypatch.setattr(db_utils, "flush_values_batch", fake_flush_values_batch)

    conn = object()
    batch_size = db_utils.BATCH_SIZE
    total_rows = batch_size * 2 + 5  # 2 full batches + 1 partial

    rows = [(i,) for i in range(total_rows)]

    db_utils.load_into_table(
        conn=cast(psycopg2.extensions.connection, conn),
        row_generator=rows,
        input_query="INSERT ... VALUES %s",
    )

    # We expect three flushes: [0..999], [1000..1999], [2000..2004].
    assert len(calls) == 3

    lengths = [len(c["batch"]) for c in calls]
    assert lengths == [batch_size, batch_size, 5]

    # Check that the content boundaries are correct.
    all_flushed = [row for c in calls for row in c["batch"]]
    assert all_flushed == rows


def test_flush_values_batch_uses_execute_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Ensure `flush_values_batch` delegates to `execute_values` with the correct arguments.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to replace `execute_values` with a stub that
        records the cursor, query, and batch.

    Returns
    -------
    None
        The test passes if `execute_values` is called exactly once with:
            - a cursor object obtained from `conn.cursor()`,
            - the same SQL query string passed into `flush_values_batch`,
            - and the same batch list.

    Raises
    ------
    AssertionError
        If `execute_values` is not called, is called with the wrong query or
        batch, or the recorded cursor is not of the expected type.

    Notes
    -----
    - Uses a minimal `DummyConn`/`DummyCursor` pair to exercise the context
      manager behavior of `conn.cursor()`.
    - Does not commit or roll back; transaction management is left to callers.
    """

    recorded: dict[str, Any] = {}

    def fake_execute_values(cur: Any, query: str, batch: list[tuple]) -> None:
        recorded["cur"] = cur
        recorded["query"] = query
        recorded["batch"] = list(batch)

    monkeypatch.setattr(db_utils, "execute_values", fake_execute_values)

    class DummyCursor:
        def __enter__(self) -> "DummyCursor":  # type: ignore[override]
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
            return None

    class DummyConn:
        def cursor(self) -> DummyCursor:
            return DummyCursor()

    conn = DummyConn()
    batch = [("a", 1), ("b", 2)]
    query = "INSERT INTO t(col1, col2) VALUES %s"

    db_utils.flush_values_batch(
        conn=cast(psycopg2.extensions.connection, conn), batch=batch, input_query=query
    )

    assert recorded["query"] == query
    assert recorded["batch"] == batch
    assert isinstance(recorded["cur"], DummyCursor)


def test_str_to_timestamp_parses_iso_and_is_utc() -> None:
    """
    Verify that `str_to_timestamp` parses ISO dates into UTC timestamps at midnight.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if the returned object is a tz-aware `Timestamp` in UTC
        corresponding to the given "YYYY-MM-DD" and with time components set
        to 00:00:00.

    Raises
    ------
    AssertionError
        If the result is not a `Timestamp`, not timezone-aware, or does not
        match the expected date at midnight.

    Notes
    -----
    - This test only checks a single representative date; the function delegates
      parsing to `datetime.strptime` under the hood with a fixed format.
    """

    ts = db_utils.str_to_timestamp("2024-01-02")

    assert isinstance(ts, pd.Timestamp)
    assert ts.tz is not None
    assert ts.tz.tzname(None) == "UTC"
    assert ts.year == 2024
    assert ts.month == 1
    assert ts.day == 2
    assert ts.hour == 0
    assert ts.minute == 0
    assert ts.second == 0


def test_str_to_timestamp_raises_on_bad_format() -> None:
    """
    Ensure `str_to_timestamp` raises ValueError on malformed date strings.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if calling `str_to_timestamp` with an invalid format
        results in a `ValueError`.

    Raises
    ------
    AssertionError
        If no exception is raised, or an unexpected exception type is raised,
        for an invalid date string.

    Notes
    -----
    - Uses "2024/01/02" (wrong separator) to trigger the parsing failure.
    - Mirrors the underlying `datetime.strptime` behavior.
    """

    with pytest.raises(ValueError):
        db_utils.str_to_timestamp("2024/01/02")  # wrong separator


def test_process_chunk_builds_url_selects_columns_and_applies_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Check that `process_chunk` builds the request URL, selects columns, and applies a processor.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to stub `pandas.read_json` so that no real HTTP
        request is made and the returned DataFrame is under test control.

    Returns
    -------
    None
        The test passes if:
            - `read_json` is called with the expected URL,
            - only the requested `column_names` are present in the result, and
            - the `processor` is applied to the selected DataFrame.

    Raises
    ------
    AssertionError
        If the captured URL does not match the expected pattern, if the
        resulting columns differ from ["a", "c"], or if the transformed
        values do not match the processor’s behavior.

    Notes
    -----
    - Builds a fake DataFrame with columns "a", "b", and "c"; the test
      requests only ["a", "c"].
    - Uses a simple processor that multiplies column "a" by 10 to make
      verification straightforward.
    - The exact URL format is:
      f"{url}{','.join(chunk)}?apikey={api_key}".
    """

    captured_url: dict[str, str] = {}

    def fake_read_json(url: str) -> pd.DataFrame:
        captured_url["url"] = url
        # Include more columns than we select to prove selection works.
        return pd.DataFrame(
            {
                "a": [1, 2],
                "b": [10, 20],
                "c": [100, 200],
            }
        )

    monkeypatch.setattr(db_utils.pd, "read_json", fake_read_json)

    chunk = pd.Series(["TICK1", "TICK2"])
    api_key = "SECRET"
    base_url = "https://example.test/profiles?symbols="

    def processor(df: pd.DataFrame) -> pd.DataFrame:
        # Simple, visible transformation: multiply column "a" by 10.
        df = df.copy()
        df["a"] = df["a"] * 10
        return df

    result = db_utils.process_chunk(
        chunk=chunk,
        api_key=api_key,
        url=base_url,
        column_names=["a", "c"],
        processor=processor,
    )

    # URL should be base + comma-joined chunk + "?apikey=...".
    assert captured_url["url"] == "https://example.test/profiles?symbols=TICK1,TICK2?apikey=SECRET"

    # Columns should be selected and processor applied.
    assert list(result.columns) == ["a", "c"]
    assert list(result["a"]) == [10, 20]
    assert list(result["c"]) == [100, 200]
