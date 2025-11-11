"""
Purpose
-------
Provide shared type aliases for CC-NEWS sampling components.

Key behaviors
-------------
- Capture the shapes of per-session, per-day, and per-month sampling
  structures in a single place.
- Document the `(year, month)` key convention used across calendar, reservoir,
  and orchestration modules.

Conventions
-----------
- `YearMonth` is always a pair of string components `(year, month)` where
  `year` is `'YYYY'` and `month` is `'MM'`.
- Day strings inside `MonthlySamples` follow `DATE_FMT` (`'YYYY-MM-DD'`).
- Session keys inside `OverIntraSamples` are the strings `"intraday"` and
  `"overnight"`.

Downstream usage
----------------
- Import these aliases in modules such as `reservoir_sampling`,
  `extract_sample`, and `uniform_sampling` to keep function signatures concise
  and consistent.
- Use `SamplingDict` as the canonical type for the full `(year, month, day,
  session)` sampling structure returned by the high-level pipeline.
"""

from typing import TypeAlias

SessionCaps: TypeAlias = tuple[int, int]
SessionTimes: TypeAlias = tuple[int | None, int | None]
OverIntraSamples: TypeAlias = dict[str, list[str]]
MonthlySamples: TypeAlias = dict[str, OverIntraSamples]
YearMonth: TypeAlias = tuple[str, str]
SamplingDict: TypeAlias = dict[YearMonth, MonthlySamples]
