"""
Purpose
-------
Validate the inclusive start / exclusive end computation performed by
`month_bounds` for month slices used in NYSE calendar queries.

Behavior validated
------------------
- Mid-year month (February): returns (YYYY-02-01, YYYY-03-01).
- December rollover: returns (YYYY-12-01, (YYYY+1)-01-01).

Conventions
-----------
- Pure date arithmetic; no database or timezone concerns.
- Fixed test year for determinism.

Downstream usage
----------------
Run with PyTest as part of the calendar utilities test suite:
`python -m pytest -q tests/.../test_calendar_utils.py`
"""

import datetime as dt

from aws.ccnews_sampler.calendar_utils import month_bounds

TEST_YEAR: int = 2024


def test_month_bounds() -> None:
    """
    Verify inclusive start / exclusive end bounds for February and December.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the returned tuples differ from the expected (start_inclusive, end_exclusive).

    Notes
    -----
    - Covers a representative mid-year month and the year-rollover edge case.
    """

    expected_february_bounds: tuple[dt.date, dt.date] = (
        dt.date(TEST_YEAR, 2, 1),
        dt.date(TEST_YEAR, 3, 1),
    )
    expected_december_bounds: tuple[dt.date, dt.date] = (
        dt.date(TEST_YEAR, 12, 1),
        dt.date(TEST_YEAR + 1, 1, 1),
    )
    assert month_bounds(TEST_YEAR, 2) == expected_february_bounds
    assert month_bounds(TEST_YEAR, 12) == expected_december_bounds
