"""
Purpose
-------
Parametrized tests for small, critical helpers used by the EDGAR search flow:
`evaluate_updated_timestamp` (window gating on <updated>) and
`evaluate_page_break_conditions` (pagination stop conditions).

Key behaviors
-------------
- Validates half-open window semantics for `<updated>`:
  older-than-window stops paging, at/after-window proceeds, missing `<updated>` skips entry.
- Verifies page-break decisions for empty, short, and full pages using `REQUEST_COUNT`.

Conventions
-----------
- All timestamps are UTC.
- Window semantics are left-closed, right-open (`start <= ts < end`).
- Only public helpers are exercised; collaborators are patched via `pytest-mock`.

Downstream usage
----------------
- Run with: `python -m pytest -q tests`
- Part of CI/CD pipelines to ensure seed logic correctness.
"""

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from infra.seeds.seed_firm_characteristics.seed_evidence.edgar_search.edgar_search_utils import (
    REQUEST_COUNT,
    RunData,
    evaluate_page_break_conditions,
    evaluate_updated_timestamp,
)

# fmt: off
from tests.test_infra.test_seeds.test_seed_firm_characteristics.test_edgar_search.\
    edgar_search_testing_utils import (
    TEST_TICKER,
    TEST_VALIDITY_WINDOW,
    UPDATED_AT_START,
)
from tests.test_infra.test_seeds.test_seed_firm_characteristics.test_edgar_search.\
    test_edgar_search_core.edgar_search_core_testing_utils import (
    make_run_data,
)

# fmt: on


@pytest.mark.parametrize(
    "updated_node_text,expected_ts,expected_missing",
    [
        # 1) Updated older than window start -> stop paging: (None, False)
        ("2024-01-01T00:00:00Z", None, False),
        # 2) Updated at or after window start -> proceed with timestamp: (ts, False)
        ("2024-01-02T00:00:00Z", UPDATED_AT_START, False),
        # 3) Missing <updated> -> skip entry, continue paging: (None, True)
        (None, None, True),
    ],
)
def test_evaluate_updated_timestamp(
    mocker: MockerFixture,
    updated_node_text: str | None,
    expected_ts: pd.Timestamp | None,
    expected_missing: bool,
) -> None:
    """
    Verify `(timestamp, missing_flag)` outcomes for `<updated>` across key scenarios.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Used to patch `find_element` to simulate presence/absence of `<updated>`.
    updated_node_text : str | None
        The `<updated>` text to inject, or `None` to simulate a missing node.
    expected_ts : pandas.Timestamp | None
        Expected timestamp result (or `None` when paging should stop or entry should be skipped).
    expected_missing : bool
        Expected "missing `<updated>`" flag.

    Returns
    -------
    None
        Asserts that the returned `(ts, missing)` pair matches expectations.

    Raises
    ------
    AssertionError
        If the function’s output deviates from the expected `(ts, missing)`.

    Notes
    -----
    - Patch targets only the XML read (`find_element`); no logging is asserted.
    - Encodes three cases:
        (1) older-than-window → `(None, False)`,
        (2) at/after-window → `(ts, False)`,
        (3) missing `<updated>` → `(None, True)`.
    """

    run_data: RunData = make_run_data(TEST_TICKER, TEST_VALIDITY_WINDOW)

    def find_element_side_effect(_parent: Any, tag: str, _ns: Any) -> MagicMock | None:
        if tag == "updated":
            if updated_node_text is None:
                return None
            elem: MagicMock = MagicMock()
            elem.text = updated_node_text
            return elem
        return None

    mocker.patch(
        "infra.seeds.seed_firm_characteristics.seed_evidence."
        "edgar_search.edgar_search_utils.find_element",
        side_effect=find_element_side_effect,
    )

    result_ts, result_missing = evaluate_updated_timestamp(run_data)

    assert result_ts == expected_ts
    assert result_missing is expected_missing


@pytest.mark.parametrize(
    "entries_count,expected_break",
    [
        # 1) No entries -> break
        (0, True),
        # 2) Short page -> break
        (REQUEST_COUNT - 1, True),
        # 3) Full page -> keep going
        (REQUEST_COUNT, False),
    ],
)
def test_evaluate_page_break_conditions(
    entries_count: int,
    expected_break: bool,
) -> None:
    """
    Check pagination stop decisions for empty, short, and full pages.

    Parameters
    ----------
    entries_count : int
        Number of mocked `<entry>` elements on the page.
    expected_break : bool
        Whether a page break (stop paging) is expected.

    Returns
    -------
    None
        Asserts the boolean returned by `evaluate_page_break_conditions`.

    Raises
    ------
    AssertionError
        If the page-break decision does not match the expected outcome.

    Notes
    -----
    - Covers boundary conditions at `0`, `REQUEST_COUNT - 1`, and `REQUEST_COUNT`.
    - A dummy logger is passed to satisfy the signature; no logger behavior is asserted.
    - Keeps the test isolated from network/IO and XML parsing.
    """

    logger: MagicMock = MagicMock()
    entries: list[MagicMock] = [MagicMock() for _ in range(entries_count)]

    result: bool = evaluate_page_break_conditions(entries, logger, TEST_TICKER, 0)

    assert result is expected_break
