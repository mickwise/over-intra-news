"""
Purpose
-------
Unit tests for `aws.ccnews_sampler.quota`.

Key behaviors
-------------
- Validate that `compute_daily_caps`:
  - produces integer intraday and overnight caps whose sum equals `daily_cap`
    for every day,
  - respects edge cases where `overnight_fraction` is 0.0 or 1.0,
  - implements unbiased randomized rounding via a Bernoulli step on the
    fractional remainder,
  - mutates the input DataFrame in place and returns the same object.

Conventions
-----------
- All tests operate on small, fully in-memory DataFrames.
- Randomness is controlled via a deterministic dummy RNG so outcomes are
  reproducible and not tied to NumPy’s generator internals.

Downstream usage
----------------
- Run via `pytest` as part of the CI suite.
- Serves as executable documentation of how per-day intraday and overnight
  caps are derived from `overnight_fraction` and `daily_cap`.
"""

from __future__ import annotations

from typing import Iterable, List, cast

import numpy as np
import pandas as pd

from aws.ccnews_sampler import quota


class DummyRNG:
    """
    Purpose
    -------
    Minimal stand-in for `numpy.random.Generator` used to drive deterministic
    randomized rounding in tests.

    Key behaviors
    -------------
    - Returns a fixed sequence of pre-specified uniform(0, 1) values via
      a `.random(n)` method, matching the interface expected by
      `compute_daily_caps`.
    - Asserts that the requested number of draws matches the configured
      length, catching accidental size mismatches in tests.

    Parameters
    ----------
    values : Iterable[float]
        Sequence of values in [0.0, 1.0] to return on the next `.random(n)`
        call, in order.

    Attributes
    ----------
    _values : list[float]
        Backing store for the deterministic sequence of random draws.

    Notes
    -----
    - This is deliberately minimal and only implements the subset of the
      `numpy.random.Generator` interface used by the production code.
    - Using a dummy RNG instead of a real Generator keeps tests robust to
      implementation changes in NumPy’s RNG algorithms.
    """

    def __init__(self, values: Iterable[float]) -> None:
        self._values: List[float] = list(values)

    def random(self, n: int) -> np.ndarray:
        """
        Return the next `n` deterministic uniform(0, 1) draws as a NumPy array.

        Parameters
        ----------
        n : int
            Number of draws requested by `compute_daily_caps`.

        Returns
        -------
        numpy.ndarray
            Array of shape `(n,)` containing the configured values in order.

        Raises
        ------
        AssertionError
            If `n` does not match the number of configured values, indicating
            that the test setup is inconsistent with the production call.
        """
        assert n == len(self._values), f"DummyRNG expected {len(self._values)} draws, got {n}"
        return np.asarray(self._values, dtype=float)


def test_compute_daily_caps_respects_edge_cases_and_invariants() -> None:
    """
    Validate caps for a mix of overnight fractions, including 0.0 and 1.0.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if, for each row:
            - `intraday_cap` and `overnight_cap` are integers,
            - their sum equals `daily_cap`,
            - `overnight_fraction = 0.0` ⇒ full intraday, zero overnight,
            - `overnight_fraction = 1.0` ⇒ zero intraday, full overnight.

    Raises
    ------
    AssertionError
        If any of the invariants fail for the constructed example.

    Notes
    -----
    - Uses a small mixed set of fractions [0.0, 0.25, 0.5, 1.0] with a fixed
      `daily_cap` to exercise both interior and edge behavior in one place.
    - Random draws matter only for the 0.25 row (fractional intraday
      expectation); the others have integer expectations and no fractional
      remainder.
    """
    daily_cap = 10
    df = pd.DataFrame(
        {
            "overnight_fraction": [0.0, 0.25, 0.5, 1.0],
        },
        index=pd.date_range("2024-01-01", periods=4),
    )

    # Chosen so that only the 0.25 row has a non-trivial Bernoulli decision.
    rng = cast(np.random.Generator, DummyRNG([0.42, 0.9, 0.1, 0.7]))

    result = quota.compute_daily_caps(daily_cap, df, rng)

    # Invariants: integer caps and per-row sum = daily_cap.
    assert pd.api.types.is_integer_dtype(result["intraday_cap"])
    assert pd.api.types.is_integer_dtype(result["overnight_cap"])
    assert (result["intraday_cap"] + result["overnight_cap"] == daily_cap).all()

    # overnight_fraction == 0.0 → full intraday.
    assert result.loc["2024-01-01", "intraday_cap"] == daily_cap
    assert result.loc["2024-01-01", "overnight_cap"] == 0

    # overnight_fraction == 1.0 → full overnight.
    assert result.loc["2024-01-04", "intraday_cap"] == 0
    assert result.loc["2024-01-04", "overnight_cap"] == daily_cap


def test_compute_daily_caps_randomized_rounding_matches_floor_plus_bernoulli() -> None:
    """
    Verify that randomized rounding follows floor + Bernoulli(fractional part).

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if, for a fixed `overnight_fraction` and `daily_cap`,
        the resulting `intraday_cap` values:
            - equal either `floor(intraday_expected)` or
              `floor(intraday_expected) + 1` for each row, and
            - match the expected Bernoulli outcomes given a deterministic sequence
              of uniforms supplied by `DummyRNG`.

    Raises
    ------
    AssertionError
        If any intraday cap falls outside the allowed two-point support or
        disagrees with the Bernoulli comparison against the dummy draws.

    Notes
    -----
    - Uses `overnight_fraction = 0.25` and `daily_cap = 5`, so
      `intraday_expected = 3.75`, floor = 3, fractional = 0.75.
    - Chooses dummy uniforms `[0.0, 0.5, 0.74, 0.75]`, so the first three rows
      should round up to 4, and the last row should remain at 3.
    """
    daily_cap = 5
    df = pd.DataFrame(
        {"overnight_fraction": [0.25, 0.25, 0.25, 0.25]},
        index=pd.date_range("2024-03-01", periods=4),
    )
    rng = cast(np.random.Generator, DummyRNG([0.0, 0.5, 0.74, 0.75]))

    result = quota.compute_daily_caps(daily_cap, df, rng)

    intraday_expected = daily_cap * (1.0 - 0.25)  # 3.75
    floor_val = int(np.floor(intraday_expected))

    expected_intraday = pd.Series(
        [floor_val + 1, floor_val + 1, floor_val + 1, floor_val],
        index=result.index,
    )

    assert result["intraday_cap"].equals(expected_intraday)
    assert (result["overnight_cap"] == daily_cap - expected_intraday).all()
    assert (result["intraday_cap"] + result["overnight_cap"] == daily_cap).all()


def test_compute_daily_caps_mutates_and_returns_same_dataframe() -> None:
    """
    Ensure `compute_daily_caps` mutates the input DataFrame and returns it.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if:
            - The object identity of the returned DataFrame matches the input.
            - The new columns `intraday_cap` and `overnight_cap` exist on the
              original frame after the call.

    Raises
    ------
    AssertionError
        If a new DataFrame object is allocated or required columns are missing.

    Notes
    -----
    - This test documents the in-place mutation contract so that callers can
      rely on shared references to the same DataFrame being updated.
    """
    df = pd.DataFrame(
        {"overnight_fraction": [0.1, 0.9]},
        index=pd.date_range("2024-04-01", periods=2),
    )
    rng = cast(np.random.Generator, DummyRNG([0.3, 0.7]))

    result = quota.compute_daily_caps(10, df, rng)

    assert result is df
    assert "intraday_cap" in df.columns
    assert "overnight_cap" in df.columns
