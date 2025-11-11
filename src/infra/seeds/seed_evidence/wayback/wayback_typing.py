"""
Purpose
-------
Centralize lightweight type aliases used across the Wayback seeding
pipeline for clarity and static checking.

Key behaviors
-------------
- Defines `Batch` as a list of (ticker, validity_window) pairs.
- Defines `SeenCandidates` as the in-memory index of WayBackCandidate
  objects keyed first by (ticker, validity_window) and then by
  candidate_cik.

Conventions
-----------
- `ValidityWindow` follows half-open [start, end) semantics using
  pandas.Timestamp objects.
- Tickers are expected to be normalized upstream to the canonical form.

Downstream usage
----------------
Import these aliases in other Wayback modules instead of repeating
the raw nested dict or list types, improving readability and tooling
support.
"""

from typing import List, TypeAlias

from infra.seeds.seed_evidence.seed_evidence_types import ValidityWindow
from infra.seeds.seed_evidence.wayback.wayback_config import WayBackCandidate

SeenCandidates: TypeAlias = dict[tuple[str, ValidityWindow], dict[str, WayBackCandidate]]
Batch: TypeAlias = List[tuple[str, ValidityWindow]]
