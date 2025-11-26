# Parquet: lda_training_doc_topics

Purpose
-------
Store the per-document topic proportions for **training** documents, as
produced by `mallet train-topics --output-doc-topics`, in a tidy
(long-format) Parquet table.

Each row corresponds to a single `(document, topic)` pair.

Source
------
- Raw file: `data/lda_output_doc_topics.txt`
- Derived from MALLET’s doc-topics output by:
  - parsing each line of `<doc_index>\t<instance_id>\t<p_0>\t...\t<p_{K-1}>`
  - reshaping into long format over topics.

Columns
-------

| Column name        | Type    | Description                                                                 |
|--------------------|---------|-----------------------------------------------------------------------------|
| doc_index          | BIGINT  | MALLET’s internal document index (0-based), as printed in the first column. |
| instance_id        | TEXT    | Original document identifier (`article_id`), from the second column.        |
| topic_id           | INT     | Topic index `k` in `0, 1, ..., K-1`.                                        |
| topic_proportion   | DOUBLE  | Estimated proportion \(\theta_{d,k}\) for this `(doc, topic)` pair.         |

Conventions
-----------
- One row per `(doc_index, topic_id)`; all topics for a given document
  sum (approximately) to 1.0.
- `instance_id` can be joined back to article_id.
- The number of topics `K` is implied by the training configuration
  (e.g., `num_topics=200`) and not stored explicitly in the table.
