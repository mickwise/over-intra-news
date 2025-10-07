"""
Purpose
-------
Exercise the control flow of the CC-NEWS sampler where logic actually lives:
1) S3 stream ingestion and branching in `fill_reservoirs`.
2) Session routing in `handle_correct_line`.

Strategy
--------
- Patch `boto3.client` to return a synthetic streaming body.
- Feed three lines: (a) unparsable, (b) parsable but off-month, (c) valid in-month.
- Patch helpers (`handle_erroneous_line`, `handle_correct_line`) and assert call counts.
- Parametrize `handle_correct_line` to cover both intraday and overnight paths.

Conventions
-----------
- Dates use 'YYYY-MM-DD' keys.
- UTC timestamps chosen so New York bucketing is unambiguous.

Downstream usage
----------------
Run with PyTest as part of the CC-NEWS sampler suite:
`python -m pytest -q tests/test_aws/test_ccnews_sampler/test_extract_sample.py`
"""

from types import TracebackType
from typing import Iterator, List, Type
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from aws.ccnews_sampler.data_maps import DataMaps
from aws.ccnews_sampler.extract_sample import fill_reservoirs, handle_correct_line
from aws.ccnews_sampler.reservoir_sampling import ReservoirManager
from aws.ccnews_sampler.run_data import RunData
from aws.ccnews_sampler.run_logger import RunLogger

TEST_QUEUE_STREAM: Iterator[bytes] = iter(
    [
        b"not-a-cc-news-line\n",
        b"warc/path/with/invalid/timestamp/CC-NEWS-20220102-060000-000000.warc.gz\n",
        b"warc/path/with/valid/timestamp/CC-NEWS-20220101-160000-000000.warc.gz\n",
    ]
)

# Dummies for typing
TEST_NYSE_CAL: pd.DataFrame = pd.DataFrame()
TEST_RUN_LOGGER: RunLogger = RunLogger("", "", {})
TEST_RNG: np.random.Generator = np.random.default_rng(0)

TEST_RUN_DATA: RunData = RunData(
    bucket="bucket",
    key="key",
    year="2022",
    month="01",
    daily_cap=0,
    nyse_cal=TEST_NYSE_CAL,
    logger=TEST_RUN_LOGGER,
    rng=TEST_RNG,
)
TEST_DATA_MAPS = DataMaps(
    cap_dict={
        "2022-01-01": (0, 5),  # Saturday (non-trading)
        "2022-01-05": (3, 2),  # Wednesday
    },
    session_dict={
        # date: (session_open_seconds_UTC, session_close_seconds_UTC)
        "2022-01-01": (None, None),  # non-trading day → no session
        "2022-01-05": (1641393000, 1641416400),  # 14:30–21:00 UTC (9:30–16:00 ET)
    },
    valid_date_set={"2022-01-01", "2022-01-05"},
)
TEST_RESERVOIR_MANAGER: ReservoirManager = ReservoirManager({}, TEST_RNG)
TEST_TUPLES: List[tuple[str, pd.Timestamp, str, bool]] = [
    (
        "warc/path/with/valid/timestamp/CC-NEWS-20220101-160000-000000.warc.gz",
        pd.Timestamp("2022-01-01 16:00:00+0000", tz="UTC"),
        "2022-01-01",
        False,
    ),
    (
        "warc/path/with/invalid/timestamp/CC-NEWS-20220105-160000-000000.warc.gz",
        pd.Timestamp("2022-01-05 16:00:00+0000", tz="UTC"),
        "2022-01-05",
        True,
    ),
]


class TestQueueStream:
    """
    Minimal stand-in for the S3 `Body` stream returned by `get_object`.

    Purpose
    -------
    Provide the context manager + `.iter_lines()` surface that `fill_reservoirs`
    expects, without hitting the network.

    Behavior
    --------
    - `__enter__/__exit__` make it usable in a `with` block.
    - `.iter_lines()` yields the predefined byte lines in `TEST_QUEUE_STREAM`.
    """

    def __enter__(self) -> "TestQueueStream":
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    def iter_lines(self) -> Iterator[bytes]:
        return TEST_QUEUE_STREAM


def test_fill_reservoir(mocker: MockerFixture) -> None:
    """
    Drive the S3 → parse → route loop and assert helper calls + counters.

    Setup
    -----
    - Patch `boto3.client().get_object(...).Body` to yield three lines:
        1) Unparsable         → `handle_erroneous_line`
        2) Parsable off-month → `handle_erroneous_line`
        3) Valid in-month     → `handle_correct_line`
    - Provide a `DataMaps` where only '2022-01-01' is a valid date key.

    Expectations
    -----------
    - `get_object` called once with the bucket/key from `RunData`.
    - `handle_erroneous_line` called twice (unparsable + off-month).
    - `handle_correct_line` called once (valid in-month).
    - Counters: `lines_total == 3`, `lines_matched == 2`.
    """

    run_context: dict[str, int] = {
        "lines_total": 0,
        "lines_matched": 0,
    }
    mock_s3_client, mock_handle_erroneous_line, mock_handle_correct_line = mock_helpers_and_s3(
        mocker
    )
    fill_reservoirs(run_context, TEST_RUN_DATA, TEST_DATA_MAPS, TEST_RESERVOIR_MANAGER)
    mock_s3_client.get_object.assert_called_once_with(Bucket="bucket", Key="key")
    assert mock_handle_erroneous_line.call_count == 2
    mock_handle_correct_line.assert_called_once()
    assert run_context["lines_total"] == 3
    assert run_context["lines_matched"] == 2


@pytest.mark.parametrize("line,date,date_key,is_intra_day", TEST_TUPLES)
def test_handle_correct_line(
    mocker: MockerFixture, line: str, date: pd.Timestamp, date_key: str, is_intra_day: bool
) -> None:
    """
    Route a parsed item into the correct session and bump per-day counters.

    Parameters (parametrized)
    -------------------------
    line : str
        Synthetic WARC path (kept as-is by the sampler).
    date : pandas.Timestamp
        Parsed UTC timestamp for the item.
    date_key : str
        Trading-day key ('YYYY-MM-DD').
    is_intra_day : bool
        Expected routing: True → intraday, False → overnight.

    Behavior validated
    ------------------
    - When the UTC instant falls inside [open, close), increments
      `per_day_intraday_count[date_key]` and calls `sample(..., "intraday")`.
    - Otherwise increments `per_day_overnight_count[date_key]` and calls
      `sample(..., "overnight")`.
    - Uses the fixed `DataMaps` fixture: '2022-01-01' (no session) and
      '2022-01-05' (14:30–21:00 UTC).
    """

    run_context: dict[str, dict[str, int]] = {
        "per_day_intraday_count": {date_key: 0},
        "per_day_overnight_count": {date_key: 0},
    }
    mock_reservoir_manager_sample: MagicMock = mocker.patch.object(TEST_RESERVOIR_MANAGER, "sample")
    handle_correct_line(line, date, date_key, run_context, TEST_DATA_MAPS, TEST_RESERVOIR_MANAGER)
    if is_intra_day:
        assert run_context["per_day_intraday_count"][date_key] == 1
        assert run_context["per_day_overnight_count"][date_key] == 0
        mock_reservoir_manager_sample.assert_called_once_with(line, date_key, "intraday")
    else:
        assert run_context["per_day_intraday_count"][date_key] == 0
        assert run_context["per_day_overnight_count"][date_key] == 1
        mock_reservoir_manager_sample.assert_called_once_with(line, date_key, "overnight")


def mock_helpers_and_s3(mocker: MockerFixture) -> tuple[MagicMock, MagicMock, MagicMock]:
    """
    Patch external collaborators so `fill_reservoirs` can be exercised offline.

    Returns
    -------
    (mock_s3_client, mock_handle_erroneous_line, mock_handle_correct_line)

    Details
    -------
    - `aws.ccnews_sampler.extract_sample.boto3.client` → returns a fake client whose
      `get_object(...)"Body"` is a `TestQueueStream`.
    - `handle_erroneous_line` / `handle_correct_line` → patched to count invocations.
    """

    mock_s3_client: MagicMock = MagicMock()
    mocker.patch("aws.ccnews_sampler.extract_sample.boto3.client", return_value=mock_s3_client)
    mock_s3_client.get_object.return_value = {"Body": TestQueueStream()}
    mock_handle_erroneous_line: MagicMock = mocker.patch(
        "aws.ccnews_sampler.extract_sample.handle_erroneous_line"
    )
    mock_handle_correct_line: MagicMock = mocker.patch(
        "aws.ccnews_sampler.extract_sample.handle_correct_line"
    )
    return mock_s3_client, mock_handle_erroneous_line, mock_handle_correct_line
