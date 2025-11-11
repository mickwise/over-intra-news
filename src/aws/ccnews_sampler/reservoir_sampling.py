"""
Purpose
-------
Provide streaming, fixed-capacity reservoir sampling primitives for CC-NEWS
WARC links. This module implements per-trading-day, per-session uniform
sampling and a manager that wires reservoirs to trading-calendar-derived caps.

Key behaviors
-------------
- Implements a classic reservoir sampler (`Reservoir`) with O(1) expected
  admission time per item and bounded memory.
- Provides `ReservoirManager`, which pre-allocates one intraday and one
  overnight reservoir per trading day based on a cap dictionary.
- Exposes a typed alias (`OverIntraSamples`) for the final per-date,
  per-session sample structure.

Conventions
-----------
- Date keys are New York trading dates formatted as 'YYYY-MM-DD'.
- Session keys are the strings "intraday" and "overnight".
- Caps come from `SessionCaps` (tuple[int, int] = (intraday_cap, overnight_cap)).
- A single numpy.random.Generator instance may be shared across all reservoirs
  for deterministic single-process runs.

Downstream usage
----------------
Construct a `ReservoirManager(cap_dict, rng)` once per run or per month, then
route each parsed WARC link via `manager.sample(candidate, date_key, session)`.
After the scan finishes, call `manager.extract_sample_dict()` to obtain a
nested mapping of trading_date → { "intraday": [...], "overnight": [...] }
that can be written out or further processed.
"""

from dataclasses import dataclass

import numpy as np

from aws.ccnews_sampler.ccnews_sampler_types import OverIntraSamples
from aws.ccnews_sampler.data_maps import SessionCaps


@dataclass
class Reservoir:
    """
    Purpose
    -------
    Maintain a fixed-size, uniform random sample from a streaming sequence of
    items using the classic reservoir sampling algorithm (What is called
    algorithm R in the literature).

    Key behaviors
    -------------
    - Accepts items one at a time via `consider(...)` and either keeps or discards
      them in O(1) expected time.
    - Ensures that, after processing N items, each has equal probability
      `min(1, cap / N)` of being present in `samples`.

    Parameters
    ----------
    cap : int
        Maximum number of items the reservoir is allowed to retain.
    samples : list[str]
        Initial contents of the reservoir; mutated in place as new items are
        considered.
    rng : numpy.random.Generator
        Random number generator used to perform admission draws and index
        selection.

    Attributes
    ----------
    cap : int
        Fixed capacity of the reservoir; never changes after construction.
    samples : list[str]
        Current reservoir contents; grows until `cap` and is then updated
        in place as items are replaced.
    rng : numpy.random.Generator
        RNG instance shared across admission decisions for this reservoir.
    seen_count : int
        Number of items that have been passed to `consider(...)` so far,
        including those not currently stored.

    Notes
    -----
    - Implements the standard algorithm: the first `cap` items are always kept;
      for each subsequent item i (1-based), a uniform index j ∼ U{0, i - 1} is
      drawn and the item replaces `samples[j]` if j < cap.
    - The class is not thread-safe. If used from multiple threads, synchronize
      calls to `consider(...)` externally or shard reservoirs per worker.
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
    Coordinate per-day, per-session reservoir samplers for CC-NEWS sampling,
    using a precomputed dictionary of intraday and overnight caps.

    Key behaviors
    -------------
    - Pre-allocates a pair of reservoirs ("intraday", "overnight") for each
      date key present in the cap dictionary.
    - Provides a simple routing API (`sample(candidate, date, session)`) that
      forwards items to the correct underlying reservoir.
    - Extracts final per-date, per-session samples as plain Python dictionaries
      suitable for serialization.

    Parameters
    ----------
    cap_dict : dict[str, SessionCaps]
        Mapping from 'YYYY-MM-DD' date keys to `(intraday_cap, overnight_cap)`
        integer tuples specifying reservoir capacities for each session.
    rng : numpy.random.Generator
        Random number generator to be shared across all child reservoirs,
        ensuring deterministic behavior for a fixed seed and input order.

    Attributes
    ----------
    reservoir_dict : dict[str, dict[str, Reservoir]]
        Nested mapping from date key to session key ("intraday"/"overnight")
        to the corresponding `Reservoir` instance.

    Notes
    -----
    - It is assumed that `date` and `session` arguments passed to `sample(...)`
      correspond to keys that exist in `reservoir_dict`; callers are responsible
      for enforcing this invariant.
    - Sharing a single RNG across reservoirs is sufficient for uniformity; if
      stronger independence guarantees or parallelization are needed, consider
      constructing child generators per reservoir upstream.
    - `extract_sample_dict()` returns live references to the `samples` lists;
      copy them if you require immutability or isolation from further mutation.
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
