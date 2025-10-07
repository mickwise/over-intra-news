"""
Purpose
-------
Unit tests for the reservoir sampler’s core admission logic (`Reservoir.consider`).

Key behaviors
-------------
- Verifies append-only behavior while the reservoir size is <= cap (RNG unused).
- Verifies post-cap behavior: RNG is called exactly once on the first overflow
  and the candidate either replaces an existing item (j < cap) or is discarded (j >= cap).
- Checks per-item invariants: `seen_count` increments by 1 and final size equals `min(n, cap)`.

Conventions
-----------
- `TEST_CAP` is fixed at 10 to exercise the boundary at n = cap and n = cap + 1.
- Replacement index is forced via a mocked RNG (`MagicMock.integers`) set to return
  `TEST_REPLACEMENT_INDEX` for the replacement-path case.
- Candidate items are synthetic strings (`"sample_{i}"`) with monotonically
  increasing indices for easy membership checks.

Downstream usage
----------------
- Run under PyTest; serves as a specification for `Reservoir.consider` behavior.
"""

from typing import List, cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from aws.ccnews_sampler.reservoir_sampling import Reservoir

TEST_CAP: int = 10
TEST_REPLACEMENT_INDEX: int = 3
TEST_RNG_SAMPLE_SIZE: List[tuple[int | None, int]] = [
    (11, 5),  # Sample smaller then cap, no replacement
    (TEST_REPLACEMENT_INDEX, 11),  # Sample larger than cap, with replacement
    (10, 11),  # Sample larger than cap, no replacement
]


@pytest.mark.parametrize("rng_return_val,sample_size", TEST_RNG_SAMPLE_SIZE)
def test_consider(rng_return_val: int, sample_size: int) -> None:
    """
    Drive `Reservoir.consider` through boundary and post-cap paths in a single, parameterized test.

    Parameters
    ----------
    rng_return_val : int | None
        Fixed value returned by `rng.integers(0, seen_count)` once the reservoir is over capacity.
        `None` denotes the pre-cap scenario, where the RNG must not be called.
    sample_size : int
        Number of candidates to feed into the reservoir in order.

    Returns
    -------
    None
        Assertions validate behavior; no value is returned.

    Notes
    -----
    - For `sample_size <= TEST_CAP`, the test asserts that the RNG is **never** invoked.
    - For `sample_size == TEST_CAP + 1`, the test asserts a **single** RNG call with
    arguments `(0, TEST_CAP + 1)` and delegates step-wise checks to `run_time_assertions(...)`.
    """

    test_reservoir, mock_rng = generate_test_reservoir(rng_return_val)
    test_samples: list[str] = [f"sample_{i}" for i in range(sample_size)]
    run_time_assertions(test_reservoir, test_samples, rng_return_val)
    assert len(test_reservoir.samples) == min(sample_size, TEST_CAP)

    # Assertions based on replacement policy
    if sample_size <= TEST_CAP:
        mock_rng.integers.assert_not_called()
    else:
        mock_rng.integers.assert_called_once_with(0, TEST_CAP + 1)


def run_time_assertions(
    test_reservoir: Reservoir, test_samples: List[str], rng_return_val: int
) -> None:
    """
    Feed a sequence into a reservoir and assert per-step invariants.

    Parameters
    ----------
    test_reservoir : Reservoir
        Reservoir under test; expected to start with an empty `samples` list.
    test_samples : List[str]
        Ordered stream of candidate strings to submit to `Reservoir.consider`.
    rng_return_val : int | None
        Mocked return value used by `rng.integers(...)` after capacity is reached.
        When less than `TEST_CAP`, the current item must replace the element at
        `TEST_REPLACEMENT_INDEX`; otherwise the current item must be absent.

    Returns
    -------
    None
        Performs assertions on each step.

    Notes
    -----
    - Validates that `seen_count` increases by exactly 1 per item.
    - While `len(samples) <= TEST_CAP`, every new candidate must be present (append path).
    - On the first item past capacity:
        * if `rng_return_val < TEST_CAP` → the new candidate must appear at
        `samples[TEST_REPLACEMENT_INDEX]`;
        * else → the new candidate must **not** be present (no replacement).
    """

    prev_seen_count: int = test_reservoir.seen_count
    for sample in test_samples:
        test_reservoir.consider(sample)
        assert test_reservoir.seen_count == prev_seen_count + 1
        if test_reservoir.seen_count <= TEST_CAP:
            assert sample in test_reservoir.samples
        else:
            if rng_return_val < TEST_CAP:
                assert sample == test_reservoir.samples[3]
            else:
                assert sample not in test_reservoir.samples
        prev_seen_count = test_reservoir.seen_count


def generate_test_reservoir(rng_return_val: int) -> tuple[Reservoir, MagicMock]:
    """
    Build a `Reservoir` configured for deterministic tests by injecting a mocked RNG.

    Parameters
    ----------
    rng_return_val : int
        Value that the mocked RNG should return from `integers(low, high)` when
        `Reservoir.consider(...)` first needs a replacement index (i.e., once the
        reservoir is at capacity). This forces either replacement or no-op behavior
        depending on the test scenario.

    Returns
    -------
    tuple[Reservoir, MagicMock]
        A pair `(reservoir, mock_rng)` where:
        - `reservoir` has `cap=TEST_CAP`, empty `samples`, and uses the injected RNG.
        - `mock_rng` is the `MagicMock` whose `.integers(...)` method returns
          `rng_return_val`, allowing assertions about RNG usage (called/not called,
          arguments, etc.).

    Notes
    -----
    - The RNG is injected (not patched globally), keeping tests isolated and explicit.
    - The `MagicMock` is cast to `numpy.random.Generator` to satisfy static type
      checkers while preserving full mock behavior at runtime.
    """

    mock_rng: MagicMock = MagicMock()
    mock_rng.integers.return_value = rng_return_val
    return Reservoir(cap=TEST_CAP, samples=[], rng=cast(np.random.Generator, mock_rng)), mock_rng
