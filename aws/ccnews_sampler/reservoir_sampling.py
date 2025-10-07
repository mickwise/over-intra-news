"""
Purpose
-------
Streaming, fixed-capacity sampling of WARC links via reservoir sampling.
Provides a per-session (intraday/overnight) sampler per trading day and a thin
manager that pre-allocates those reservoirs from per-day caps.

Key behaviors
-------------
- Reservoir.consider(candidate): single-pass, O(1) expected-time admission test.
- Bounded memory: each reservoir keeps at most `cap` samples.
- Deterministic runs when given a seeded numpy.random.Generator.

Conventions
-----------
- "intraday" and "overnight" session keys are used per date (string YYYY-MM-DD).
- Caps come from `SessionCaps` (tuple[int, int] = (intraday_cap, overnight_cap)).
- RNG may be shared across all reservoirs for single-process runs.

Downstream usage
----------------
- Construct `ReservoirManager(cap_dict, rng)` once per month (or run).
- For each parsed link, call ReservoirManager.sample(candidate, date, session).
- After the scan, read `manager.reservoir_dict[date]["intraday"].samples`
  (and "overnight") to get the uniformly-sampled results.
"""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from aws.ccnews_sampler.data_maps import SessionCaps

OverIntraSamples: TypeAlias = dict[str, list[str]]


@dataclass
class Reservoir:
    """
    Purpose
    -------
    Maintain a fixed-size uniform sample from a streaming sequence of items
    using classic reservoir sampling.

    Key behaviors
    -------------
    - O(1) expected-time admission per item.
    - Keeps at most `cap` items in `samples`.
    - Deterministic when driven by a seeded numpy Generator.

    Construction
    ------------
    Parameters
    ----------
    cap : int
        Maximum number of samples to retain.
    samples : list[str]
        Initial sample list. Will be mutated in place.
    rng : numpy.random.Generator
        RNG used for admission draws.

    Attributes
    ----------
    cap : int
        Reservoir capacity.
    samples : list[str]
        The current reservoir contents.
    rng : numpy.random.Generator
        Random generator used for selection.
    seen_count : int
        Count of total items considered so far.

    Notes
    -----
    - Not thread-safe. If using concurrency, shard reservoirs per worker or
      protect access externally.
    """

    cap: int
    samples: list[str]
    rng: np.random.Generator
    seen_count: int = 0

    def consider(self, candidate: str) -> None:
        """
        Consider a candidate for inclusion in the reservoir.

        Parameters
        ----------
        candidate : str
            The link (or any token) to consider for sampling.

        Returns
        -------
        None

        Notes
        -----
        - Fills the reservoir until `cap`, then replaces a random existing item
          with probability `cap / seen_count`.
        - Uses `rng.integers(0, seen_count)` (high is exclusive).
        """
        current_sample_amount: int = len(self.samples)
        self.seen_count += 1
        if current_sample_amount < self.cap:
            self.samples.append(candidate)
        else:
            j: np.int64 = self.rng.integers(0, self.seen_count)
            if j < self.cap:
                self.samples[j] = candidate


class ReservoirManager:
    """
    Purpose
    -------
    Allocate and hold per-day reservoirs for "intraday" and "overnight" sessions,
    using caps derived from trading-calendar logic.

    Key behaviors
    -------------
    - Builds a nested `reservoir_dict[date][session] -> Reservoir`.
    - Centralizes RNG injection across all child reservoirs.

    Parameters
    ----------
    cap_dict : dict[str, SessionCaps]
        Mapping from YYYY-MM-DD date string to `(intraday_cap, overnight_cap)`.
    rng : numpy.random.Generator
        RNG to be shared across reservoirs.

    Attributes
    ----------
    reservoir_dict : dict[str, dict[str, Reservoir]]
        Nested mapping: date → {"intraday": Reservoir, "overnight": Reservoir}.

    Notes
    -----
    - If you plan to parallelize by date/session, consider creating independent
      child RNGs per reservoir.
    """

    def __init__(self, cap_dict: dict[str, SessionCaps], rng: np.random.Generator) -> None:
        self.reservoir_dict: dict[str, dict[str, Reservoir]] = {
            date: {
                "intraday": Reservoir(cap=cap[0], samples=[], rng=rng),
                "overnight": Reservoir(cap=cap[1], samples=[], rng=rng),
            }
            for date, cap in cap_dict.items()
        }

    def sample(self, candidate: str, date: str, session: str) -> None:
        """
        Route a candidate item to the appropriate per-day/session reservoir.

        Parameters
        ----------
        candidate : str
            The link (or token) to consider for sampling.
        date : str
            Trading day key in 'YYYY-MM-DD' (must exist in `reservoir_dict`).
        session : str
            Session bucket for that day. Expected values: "intraday" or "overnight".

        Returns
        -------
        None

        Raises
        ------
        None

        Notes
        -----
        - Delegates to the underlying `Reservoir.consider(...)`.
        - Uniform reservoir semantics apply: fills up to capacity, then admits with
        probability `cap / seen_count` by replacing a random existing item.
        """

        self.reservoir_dict[date][session].consider(candidate)

    def extract_sample_dict(self) -> dict[str, OverIntraSamples]:
        """
        Materialize sampled links per date and session.

        Returns
        -------
        dict[str, OverIntraSamples]
            A nested mapping of the form:
            {
                "<YYYY-MM-DD>": {
                    "intraday":  [<str>, ...],
                    "overnight": [<str>, ...]
                },
                ...
            }
            where each list contains the final uniform sample for that date/session.

        Notes
        -----
        - The returned lists are the live `samples` lists from each reservoir
        (i.e., not deep-copied). If you need immutability, copy them before
        exposing externally.
        - Capacities are enforced per reservoir; lists may be shorter than caps
        if fewer candidates were seen.
        """

        return {
            date: {
                "intraday": reservoir_info["intraday"].samples,
                "overnight": reservoir_info["overnight"].samples,
            }
            for date, reservoir_info in self.reservoir_dict.items()
        }
