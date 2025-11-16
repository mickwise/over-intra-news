"""
Purpose
-------
Central configuration for the CC-NEWS parser, defining concurrency limits,
content-encoding settings, HTML tag filters, canonicalization rules for
firm-name matching, and structural hints for locating article bodies in HTML.

Key behaviors
-------------
- Derives `CPU_COUNT` from the host and computes `MAXIMAL_WORKER_COUNT`
  as `max(1, min(CPU_COUNT - 2, 12))` to leave headroom for the OS and other
  processes.
- Enumerates `COMPRESSED_CONTENT_TYPES` to decide when HTTP payloads
  should be transparently decompressed.
- Lists `NON_VISIBLE_TAGS` that are stripped from HTML before visible
  text extraction (e.g., `<script>`, `<style>`, `<head>`).
- Defines `NAME_SUFFIXES_SET` of common corporate suffixes used during
  firm-name canonicalization.
- Provides `ARTICLE_ROOT_XPATHS` to identify high-level article containers
  (e.g., `<article>`, main content divs, or common CMS IDs/classes).
- Provides `ARTICLE_BODY_XPATHS` to identify nested article-body regions
  within a chosen root (e.g., elements with `itemprop="articleBody"` or
  body-like classes, plus structural fallbacks such as paragraphs).

Conventions
-----------
- `CPU_COUNT` falls back to 8 when `os.cpu_count()` returns None, to
  avoid degenerate zero-worker configurations.
- `MAXIMAL_WORKER_COUNT` is capped at 12 and always at least two fewer
  than `CPU_COUNT` to balance throughput with system stability.
- All tag and suffix sets are uppercase to align with upstream text
  normalization, which converts tokens to uppercase ASCII.
- `ARTICLE_ROOT_XPATHS` are absolute XPaths evaluated against the document
  root, while `ARTICLE_BODY_XPATHS` are relative XPaths evaluated inside
  a chosen article root element.
- Constants are treated as read-only; callers should not mutate them
  at runtime.

Downstream usage
----------------
Import this module from parser and matching components to share a single
source of truth for worker counts, content-encoding handling, HTML
visibility rules, firm-name suffix stripping, and article-structure
selection. Adjust values here to tune performance, matching behavior,
and article-body detection across the entire pipeline.
"""

import os
from typing import List

CPU_COUNT: int = os.cpu_count() or 8
MAXIMAL_WORKER_COUNT: int = max(1, min(CPU_COUNT - 2, 12))
COMPRESSED_CONTENT_TYPES: set[str] = {"gzip", "x-gzip", "deflate"}
ARTICLE_ROOT_XPATHS: List[str] = [
    # Semantic HTML5 containers
    "//article",
    "//main",
    # Schema.org / microdata style
    "//*[@itemprop='articleBody']",
    "//*[@itemtype='http://schema.org/NewsArticle']",
    "//*[@itemtype='https://schema.org/NewsArticle']",
    # Common CMS patterns (id-based)
    "//*[@id='article-body']",
    "//*[@id='articleBody']",
    "//*[@id='story-body']",
    "//*[@id='storyBody']",
    "//*[@id='main-content']",
    "//*[@id='mainContent']",
    "//*[@id='content']",
    "//*[@id='article']",
    "//*[@id='news-article']",
    "//*[@id='newsArticle']",
    # Common CMS patterns (class-based)
    "//*[contains(@class, 'article-body')]",
    "//*[contains(@class, 'articleBody')]",
    "//*[contains(@class, 'story-body')]",
    "//*[contains(@class, 'storyBody')]",
    "//*[contains(@class, 'post-content')]",
    "//*[contains(@class, 'entry-content')]",
    "//*[contains(@class, 'entryContent')]",
    "//*[contains(@class, 'content-body')]",
    "//*[contains(@class, 'contentBody')]",
    "//*[contains(@class, 'news-article')]",
    "//*[contains(@class, 'newsArticle')]",
    "//*[contains(@class, 'article-content')]",
    "//*[contains(@class, 'articleContent')]",
    "//*[contains(@class, 'article-text')]",
    "//*[contains(@class, 'articleText')]",
    "//*[contains(@class, 'story-content')]",
    "//*[contains(@class, 'storyContent')]",
]
ARTICLE_BODY_XPATHS: List[str] = [
    # Obvious article body containers
    ".//*[@itemprop='articleBody']",
    ".//*[@id='article-body' or @id='articleBody' or @id='story-body' or @id='storyBody']",
    # Common CMS “content” blocks
    ".//div[contains(@class, 'article-body') or contains(@class, 'articleBody')]",
    ".//div[contains(@class, 'story-body') or contains(@class, 'storyBody')]",
    (
        ".//div[contains(@class, 'post-content') or contains(@class, 'entry-content')"
        "or contains(@class, 'entryContent')]"
    ),
    (
        ".//div[contains(@class, 'article-content') or contains(@class, 'articleContent') or"
        "contains(@class, 'article-text') or contains(@class, 'articleText')]"
    ),
    (
        ".//div[contains(@class, 'content-body') or contains(@class, 'contentBody') or"
        "contains(@class, 'story-content') or contains(@class, 'storyContent')]"
    ),
    # Structural fallbacks
    ".//section[contains(@class, 'article') or contains(@class, 'content')]",
    ".//p",
]
NON_VISIBLE_TAGS: List[str] = [
    "style",
    "script",
    "head",
    "title",
    "meta",
    "link",
    "svg",
]
NAME_SUFFIXES_SET: set[str] = {
    "INC",
    "INCDE",
    "INCMD",
    "INCORPORATED",
    "CORP",
    "CORPORATION",
    "INTERNATIONAL",
    "GROUP",
    "CO",
    "COMPANY",
    "LLC",
    "LP",
    "L.P.",
    "LLLP",
    "LLP",
    "LTD",
    "LIMITED",
    "HOLDINGS",
    "HLDGS",
    "HOLDING",
    "PLC",
    "DE",
    "MD",
    "UT",
    "MO",
}
