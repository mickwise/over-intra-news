"""
Purpose
-------
Extract LDA training documents from Postgres and materialize a MALLET-compatible
tab-delimited input file in the "[ID]\t[tag]\t[text]" "spreadsheet" format.

Key behaviors
-------------
- Run a parameterized SQL query to pull bag-of-words documents for a given
  `corpus_version` and trading-day window.
- Expand document–term counts into repeated tokens and aggregate them into a
  single whitespace-separated string per article.
- Write one line per article to a UTF-8 text file that can be consumed by
  `mallet import-file` and downstream topic-model training scripts.

Conventions
-----------
- `sample_start` and `sample_end` are interpreted as calendar dates and are
  applied to `parsed_news_articles.trading_day` using a `BETWEEN` predicate
  (inclusive of both endpoints).
- `instance_id` is the article's `article_id`, and the tag field is fixed to
  the literal string `"no_label"`.
- Output is written to `INPUT_FILE_PATH` (by default
  `"local_data/lda_input_documents.txt"`) as configured in `lda.lda_config`;
  each line has the form:

    <article_id>\tno_label\t<token_1 token_2 ... token_n>\n

Downstream usage
----------------
Call `export_corpus` from the modeling pipeline to materialize the LDA
training corpus as a text file, then invoke MALLET's `import-file` and
`train-topics` commands against that file to fit topic models for the
selected sample.
"""

import datetime as dt
from typing import List

from infra.utils.db_utils import connect_to_db
from lda.lda_config import INFERENCE_INPUT_FILE_PATH, TRAINING_INPUT_FILE_PATH


def export_corpus(
    sample_start: dt.date = dt.date(2016, 8, 1),
    sample_end: dt.date = dt.date(2022, 8, 1),
    corpus_version: int = 1,
    training: bool = True,
) -> None:
    """
    Materialize an LDA corpus from Postgres into a MALLET-compatible
    tab-delimited text file.

    Parameters
    ----------
    sample_start : datetime.date, default dt.date(2016, 8, 1)
        Inclusive lower bound on `parsed_news_articles.trading_day` for
        selecting articles to include in the corpus.
    sample_end : datetime.date, default dt.date(2022, 8, 1)
        Inclusive upper bound on `parsed_news_articles.trading_day` for
        selecting articles to include in the corpus.
    corpus_version : int, default 1
        Corpus construction identifier used to filter both the vocabulary and
        document–term matrix.
    training : bool, default True
        When True, write the corpus to `TRAINING_INPUT_FILE_PATH` for use by
        `mallet train-topics`. When False, write the corpus to
        `INFERENCE_INPUT_FILE_PATH` for use by `mallet infer-topics` on an
        out-of-sample window.

    Returns
    -------
    None
        Writes a file to `TRAINING_INPUT_FILE_PATH` (when ``training=True``)
        or `INFERENCE_INPUT_FILE_PATH` (when ``training=False``) as configured
        in `lda.lda_config`; does not return any in-memory representation of
        the corpus.

    Raises
    ------
    DatabaseError
        If `connect_to_db` fails or the query execution encounters an error.
        (The exact exception type depends on the DB driver used in
        `infra.utils.db_utils`.)
    OSError
        If opening or writing the output text file fails (e.g., due to
        permission or disk-space issues).

    Notes
    -----
    - The SQL query returned by `generate_input_query` expands term counts into
      repeated tokens using `generate_series` and then aggregates them with
      `STRING_AGG`, so the output `document_text` matches the tokenized,
      cleaned corpus.
    - All rows are pulled into memory via `cursor.fetchall()` before writing.
      For very large corpora, a streaming approach (fetching in batches) may be
      preferable on small machines.
    """
    input_query: str = generate_input_query()
    with connect_to_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(input_query, (corpus_version, sample_start, sample_end))
            document_list: List[tuple] = cursor.fetchall()
    if training:
        with open(TRAINING_INPUT_FILE_PATH, "w", encoding="utf-8") as f:
            for instance_id, document_text in document_list:
                f.write(f"{instance_id}\tno_label\t{document_text}\n")
    else:
        with open(INFERENCE_INPUT_FILE_PATH, "w", encoding="utf-8") as f:
            for instance_id, document_text in document_list:
                f.write(f"{instance_id}\tno_label\t{document_text}\n")


def generate_input_query() -> str:
    """
    Build the SQL query used to extract LDA training documents from Postgres.

    The query joins `lda_document_terms`, `lda_vocabulary`, `lda_documents`,
    and `parsed_news_articles` to reconstruct per-article token sequences and
    enforce corpus, activity, and date filters.

    Parameters
    ----------
    None

    Returns
    -------
    str
        A parameterized SQL string with three placeholders, to be executed with
        `(corpus_version, sample_start, sample_end)` in that order.

    Notes
    -----
    - `dt.corpus_version = %s` filters both the document–term matrix and
      vocabulary to a single corpus construction.
    - `v.is_active = TRUE` enforces vocabulary pruning (e.g., rare or boiler-
      plate tokens).
    - `d.included_in_training = TRUE` restricts to documents chosen for LDA
      training.
    - `p.trading_day BETWEEN DATE %s AND DATE %s` applies an inclusive date
      window on the NYSE trading day.
    - `generate_series(1, dt.term_count)` is used to expand term counts into
      repeated tokens, and `STRING_AGG` with an `ORDER BY v.term_id, g.n`
      produces a deterministic token ordering within each document.
    """
    return """
    SELECT
        dt.article_id AS instance_id,
        STRING_AGG(
            v.token,
            ' ' ORDER BY v.term_id, g.n
        ) AS document_text
    FROM lda_document_terms AS dt
    JOIN lda_vocabulary AS v
      ON v.term_id = dt.term_id
     AND v.corpus_version = dt.corpus_version
    JOIN lda_documents AS d
      ON d.article_id = dt.article_id
     AND d.corpus_version = dt.corpus_version
    JOIN parsed_news_articles AS p
      ON p.article_id = dt.article_id
    JOIN LATERAL generate_series(1, dt.term_count) AS g(n) ON TRUE
    WHERE
          dt.corpus_version = %s
      AND v.is_active = TRUE
      AND d.included_in_training = TRUE
      AND p.trading_day BETWEEN
            DATE %s AND
            DATE %s
    GROUP BY
        dt.article_id
    ORDER BY
        MIN(p.trading_day),
        dt.article_id;
    """
