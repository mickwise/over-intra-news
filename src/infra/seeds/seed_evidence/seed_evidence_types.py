"""
Purpose
-------
Type aliases and helpers shared across the EDGAR harvesting pipeline. Provides
canonical type names for XML/entry collections and validity-window pairs, plus
string↔window conversion utilities used for keys, logs, and registry lookups.

Key behaviors
-------------
- Defines `Entries`, `NameSpaceBindings`, `CaptureDicts`, and window aliases
  for clarity and static tooling.
- Converts half-open [start, end) validity windows to/from canonical string keys.
- Preserves UTC and half-open semantics in both directions.

Conventions
-----------
- All timestamps are tz-aware UTC.
- Validity windows use half-open semantics: `start <= ts < end`.
- Canonical string format: `'YYYY-MM-DD to YYYY-MM-DD'` with day precision.

Downstream usage
----------------
- Import the aliases for readable type hints across modules.
- Use `validity_window_to_str(window)` for stable keys/logging.
- Use `str_to_validity_window(key)` to reconstruct the window from stored keys.
"""

from typing import Any, List, TypeAlias

import pandas as pd

Entries: TypeAlias = List[Any]
NameSpaceBindings: TypeAlias = dict[str, Any]
ValidityWindow: TypeAlias = tuple[pd.Timestamp, pd.Timestamp]
ValidityWindows: TypeAlias = List[ValidityWindow]
CandidateDict: TypeAlias = dict[tuple[str, ValidityWindow], list[str]]


def validity_window_to_str(window: ValidityWindow) -> str:
    """
    Return a canonical string key for a half-open [start, end) validity window.

    Parameters
    ----------
    window : tuple[pandas.Timestamp, pandas.Timestamp]
        UTC-aware [start, end) pair.

    Returns
    -------
    str
        'YYYY-MM-DD to YYYY-MM-DD'

    Raises
    ------
    None

    Notes
    -----
    - Preserves half-open semantics; the right bound is **not** decremented.
    - Intended for keys, grouping, and logs—not for constructing SQL range literals.
    """

    return f"{window[0].strftime('%Y-%m-%d')} to {window[1].strftime('%Y-%m-%d')}"


def str_to_validity_window(window_str: str) -> ValidityWindow:
    """
    Parse a canonical string key back into a half-open [start, end) validity window.

    Parameters
    ----------
    window_str : str
        Canonical key in the form 'YYYY-MM-DD to YYYY-MM-DD'.

    Returns
    -------
    tuple[pandas.Timestamp, pandas.Timestamp]
        UTC-aware [start, end) pair.

    Raises
    ------
    ValueError
        If the input string does not match the expected 'YYYY-MM-DD to YYYY-MM-DD' pattern
        or yields invalid dates.

    Notes
    -----
    - Sets tz='UTC' for both bounds.
    - Mirrors `validity_window_to_str` and preserves half-open semantics.
    """

    parts: List[str] = window_str.split(" to ")
    start: pd.Timestamp = pd.Timestamp(parts[0], tz="UTC")
    end: pd.Timestamp = pd.Timestamp(parts[1], tz="UTC")
    return (start, end)
