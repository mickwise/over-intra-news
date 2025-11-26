# Parquet: lda_inference_doc_topics

Purpose
-------
Store the per-document topic proportions for **inference** (out-of-sample)
documents, as produced by `mallet infer-topics --output-doc-topics`, in a
tidy (long-format) Parquet table.

Each row corresponds to a single `(document, topic)` pair, analogous to
`lda_training_doc_topics`.

Source
------
- Raw file: `data/lda_inference_output_doc_topics.txt`
- Parsed from MALLET’s inferred doc-topics output, which has the same
  wide format as the training doc-topics file:
  `<doc_index>\t<instance_id>\t<p_0>\t...\t<p_{K-1}>`.

Columns
-------

| Column name        | Type    | Description                                                                  |
|--------------------|---------|------------------------------------------------------------------------------|
| doc_index          | BIGINT  | MALLET’s internal document index for the inference batch.                    |
| instance_id        | TEXT    | Original document identifier for the inference corpus (`article_id`).        |
| topic_id           | INT     | Topic index `k` in `0, 1, ..., K-1`, consistent with the **training** model. |
| topic_proportion   | DOUBLE  | Inferred proportion \(\theta_{d,k}\) for this `(doc, topic)` pair.           |

Conventions
-----------
- Same semantics as `lda_training_doc_topics`, but for documents not used
  in training.
- `topic_id` indexing and the topic semantics are shared with the training
  model (same `OUTPUT_MODEL_FILE_PATH` / `INFERENCER_FILE_PATH`).
- `instance_id` will be used to join back to your out-of-sample returns,
  firm characteristics, etc.

