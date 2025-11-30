"""
Purpose
-------
Centralize configuration constants for the LDA topic modeling and coherence
measurement pipeline, so that corpus bounds, sampling windows, seeds, and
runtime knobs live in a single importable module rather than being
hard-coded in notebooks.

Key behaviors
-------------
- Exposes canonical corpus date bounds used when ingesting, cleaning, and
  training on the news corpus.
- Defines the default list of random seeds used when launching multiple
  LDA runs for robustness and sensitivity checks.
- Specifies the default number of top words per topic used in coherence
  measurement and reporting.
- Configures the default worker count for Gensim-based coherence
  computation.
- Provides a separate sampling window for drawing diagnostic sub-corpora
  used in coherence estimation.

Conventions
-----------
- Dates are expressed as timezone-agnostic `datetime.date` objects and are
  interpreted in the same trading calendar as the rest of the project.
- `CORPUS_START_DATE` / `CORPUS_END_DATE` bound the full news corpus used
  for LDA training and downstream regressions.
- `SAMPLE_START_DATE` / `SAMPLE_END_DATE` are used to construct run_ids for
  each of the LDA runs performed.
- `DEFAULT_SEED_NUMBERS` is an ordered, project-wide default seed list;
  downstream code may rely on these specific values when reproducing runs.
- `TOP_WORD_NUM` and `GENSIM_PROCESS_COUNT` are safe defaults for
  notebooks and batch jobs; callers may override them explicitly if needed.

Downstream usage
----------------
Import this module from modeling notebooks and batch jobs instead of
hard-coding dates, seeds, or process counts. Use the exported constants to
parameterize corpus sampling and coherence measurement so that configuration
stays consistent across experiments and is easy to audit.
"""

import datetime as dt
from typing import List

CORPUS_START_DATE: dt.date = dt.date(2016, 8, 1)

CORPUS_END_DATE: dt.date = dt.date(2025, 8, 1)


DEFAULT_SEED_NUMBERS: List[int] = [42, 43, 44, 45]


TOP_WORD_NUM: int = 10


GENSIM_PROCESS_COUNT: int = 6


SAMPLE_START_DATE: dt.date = dt.date(2016, 8, 1)


SAMPLE_END_DATE: dt.date = dt.date(2022, 8, 1)
