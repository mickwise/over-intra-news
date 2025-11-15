"""
Purpose
-------
Central configuration for the CC-NEWS parser, defining concurrency limits,
content-encoding settings, HTML tag filters, and canonicalization rules
for firm-name matching.

Key behaviors
-------------
- Derives `CPU_COUNT` from the host and computes `MAXIMAL_WORKER_COUNT`
  as `max(1, min(CPU_COUNT - 2, 12))` to leave headroom for the OS and other
  processes.
- Enumerates `COMPRESSED_CONTENT_TYPES` to decide when HTTP payloads
  should be transparently decompressed.
- Lists `NON_VISIBLE_TAGS` that are stripped from HTML before visible
  text extraction (e.g., `<script>`, `<style>`, `<head>`).

Conventions
-----------
- `CPU_COUNT` falls back to 8 when `os.cpu_count()` returns None, to
  avoid degenerate zero-worker configurations.
- `MAXIMAL_WORKER_COUNT` is capped at 12 and always at least two fewer
  than `CPU_COUNT` to balance throughput with system stability.
- The tag set is uppercase to align with upstream text
  normalization, which converts tokens to uppercase ASCII.
- Constants are treated as read-only; callers should not mutate them
  at runtime.

Downstream usage
----------------
Import this module from parser and matching components to share a single
source of truth for worker counts, content-encoding handling, HTML
visibility rules, and firm-name suffix stripping. Adjust values here to
tune performance or matching behavior across the entire pipeline.
"""

import os
from typing import List

CPU_COUNT: int = os.cpu_count() or 8
MAXIMAL_WORKER_COUNT: int = max(1, min(CPU_COUNT - 2, 12))
COMPRESSED_CONTENT_TYPES: set[str] = {"gzip", "x-gzip", "deflate"}
NON_VISIBLE_TAGS: List[str] = [
    "style",
    "script",
    "head",
    "title",
    "meta",
    "link",
    "svg",
]
