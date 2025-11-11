"""
Purpose
-------
Unit tests for `aws.ccnews_sampler.reservoir_sampling`.

Key behaviors
-------------
- Verify that `Reservoir.consider`:
  - Fills up to its configured capacity.
  - Uses the classic Algorithm R replacement rule once full, based on
    controlled RNG outputs.
  - Tracks `seen_count` correctly and never exceeds capacity.
- Verify that `ReservoirManager`:
  - Instantiates one intraday and one overnight reservoir per date key
    using capacities from `cap_dict`.
  - Routes samples to the correct underlying reservoir via `sample(...)`.
  - Exposes the final samples via `extract_sample_dict()` with the
    expected nested dictionary structure.

Conventions
-----------
- Tests use a small deterministic RNG stub where exact replacement
  behavior must be asserted.
- For manager tests, a real `numpy.random.Generator` is used; we only
  assert structural and capacity invariants, not stochastic properties.

Downstream usage
----------------
- Run via `pytest` as part of the CI suite.
- Serves as executable documentation of how reservoir sampling and
  reservoir management are expected to behave within the CC-NEWS
  sampling pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from aws.ccnews_sampler import reservoir_sampling


@dataclass
class _DeterministicRNG:
    """
    Purpose
    -------
    Minimal RNG stub that returns a pre-defined sequence of integers from
    its `integers(...)` method, used to make reservoir behavior fully
    deterministic in tests.

    Key behaviors
    -------------
    - On each call to `integers(...)`, returns the next value from an
      internal sequence.
    - Ignores all parameters except for consuming calls in order; the
      caller is responsible for providing a sequence compatible with the
      algorithm under test.

    Parameters
    ----------
    values : list[int]
        The finite sequence of integer values to yield from successive
        `integers(...)` calls.

    Attributes
    ----------
    _iter : iterator[int]
        Internal iterator over the provided integer sequence.

    Notes
    -----
    - This stub is cast to `numpy.random.Generator` at call sites to
      satisfy type checking; only the `integers` method is relied upon.
    - If more integers are requested than provided, `StopIteration` will
      be raised, surfacing an error in the test.
    """

    values: list[int]

    def __post_init__(self) -> None:
        self._iter = iter(self.values)

    def integers(
        self,
        _low: int,
        _high: int | None = None,
        _size: None = None,
        _dtype: Any = None,
        _endpoint: bool = False,
    ) -> np.int64:
        """
        Return the next predetermined integer in the sequence.

        Parameters
        ----------
        low : int
            Lower bound of the uniform range (ignored by this stub).
        high : int or None, optional
            Upper bound of the uniform range (ignored by this stub).
        size : None, optional
            Unused; present to match the `Generator.integers` signature.
        dtype : Any, optional
            Unused; present to match the signature.
        endpoint : bool, optional
            Unused; present to match the signature.

        Returns
        -------
        numpy.int64
            The next integer from the internal sequence, wrapped as
            `numpy.int64`.

        Raises
        ------
        StopIteration
            If called more times than the length of `values`.

        Notes
        -----
        - The parameters are accepted for signature compatibility only;
          the test controls correctness by choosing appropriate `values`.
        """
        val = next(self._iter)
        return np.int64(val)


def test_reservoir_fills_then_replaces_according_to_rng_sequence() -> None:
    """
    Verify that `Reservoir.consider` fills to capacity, then replaces
    items according to RNG outputs once full.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if, given a deterministic RNG sequence, the
        reservoir contents after several insertions match the expected
        replacements implied by Algorithm R.

    Raises
    ------
    AssertionError
        If the reservoir length exceeds the capacity, or if the final
        `samples` list does not reflect the expected replacements given
        the deterministic RNG outputs.

    Notes
    -----
    - Setup:
      - Capacity = 2.
      - Candidates: "a", "b", "c", "d" (in that order).
      - RNG sequence for `integers(0, seen_count)` calls: [0, 1].
    - Expected behavior:
      - "a" and "b" fill the reservoir.
      - On "c" (seen_count=3), RNG=0 → replace index 0 with "c".
      - On "d" (seen_count=4), RNG=1 → replace index 1 with "d".
      - Final reservoir: ["c", "d"].
    """

    rng_stub = _DeterministicRNG(values=[0, 1])
    rng = cast(np.random.Generator, rng_stub)

    reservoir = reservoir_sampling.Reservoir(cap=2, samples=[], rng=rng)

    reservoir.consider("a")
    reservoir.consider("b")
    reservoir.consider("c")
    reservoir.consider("d")

    assert reservoir.cap == 2
    assert reservoir.seen_count == 4
    assert len(reservoir.samples) == 2
    assert reservoir.samples == ["c", "d"]


def test_reservoir_respects_zero_capacity_and_only_counts_seen() -> None:
    """
    Ensure that a zero-capacity `Reservoir` never stores samples but still
    tracks the number of seen items.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if no items are stored in `samples` when
        `cap == 0`, and `seen_count` reflects the number of calls to
        `consider(...)`.

    Raises
    ------
    AssertionError
        If any sample is stored in a zero-capacity reservoir or if
        `seen_count` does not match the number of inputs.

    Notes
    -----
    - This test exercises the edge case where `cap = 0`, which should
      behave as a pure counter with no stored items.
    """

    rng = np.random.default_rng(123)
    reservoir = reservoir_sampling.Reservoir(cap=0, samples=[], rng=rng)

    for candidate in ["x", "y", "z"]:
        reservoir.consider(candidate)

    assert reservoir.cap == 0
    assert reservoir.seen_count == 3
    assert not reservoir.samples


def test_reservoir_manager_initializes_and_routes_samples_correctly() -> None:
    """
    Verify that `ReservoirManager`:
    - Builds one intraday and one overnight reservoir per date key using
      capacities from `cap_dict`.
    - Routes samples to the appropriate per-date, per-session reservoir.
    - Exposes the final samples via `extract_sample_dict()` with a
      consistent nested structure.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if:
            - Reservoir capacities match the input `cap_dict`.
            - Calls to `sample(...)` only affect the targeted session
              reservoir.
            - `extract_sample_dict()` returns a nested dict with the same
              date keys and session keys, and its lists match the underlying
              reservoir `samples`.

    Raises
    ------
    AssertionError
        If any reservoir capacity is incorrect, if routing sends items to
        the wrong reservoir, or if the extracted sample dictionary does
        not mirror the internal reservoir contents.

    Notes
    -----
    - Uses a small cap_dict:
        {
            "2024-01-02": (2, 1),
            "2024-01-03": (1, 0),
        }
      to exercise both non-zero and zero session caps.
    """

    cap_dict: dict[str, reservoir_sampling.SessionCaps] = {
        "2024-01-02": (2, 1),
        "2024-01-03": (1, 0),
    }
    rng = np.random.default_rng(42)

    manager = reservoir_sampling.ReservoirManager(cap_dict=cap_dict, rng=rng)

    # Check initialization capacities and keys.
    assert set(manager.reservoir_dict.keys()) == set(cap_dict.keys())

    for date, (intra_cap, over_cap) in cap_dict.items():
        intraday_res = manager.reservoir_dict[date]["intraday"]
        overnight_res = manager.reservoir_dict[date]["overnight"]
        assert intraday_res.cap == intra_cap
        assert overnight_res.cap == over_cap

    # Route some samples.
    manager.sample("i1", "2024-01-02", "intraday")
    manager.sample("i2", "2024-01-02", "intraday")
    manager.sample("o1", "2024-01-02", "overnight")
    manager.sample("i3", "2024-01-03", "intraday")

    # Capacities should be respected.
    assert len(manager.reservoir_dict["2024-01-02"]["intraday"].samples) <= 2
    assert len(manager.reservoir_dict["2024-01-02"]["overnight"].samples) <= 1
    # Date with overnight cap 0 should never hold overnight samples.
    assert manager.reservoir_dict["2024-01-03"]["overnight"].samples == []

    # Extract nested sample dictionary and compare with internal state.
    extracted = manager.extract_sample_dict()

    assert set(extracted.keys()) == set(cap_dict.keys())
    for date in cap_dict:
        assert set(extracted[date].keys()) == {"intraday", "overnight"}
        assert extracted[date]["intraday"] is manager.reservoir_dict[date]["intraday"].samples
        assert extracted[date]["overnight"] is manager.reservoir_dict[date]["overnight"].samples
