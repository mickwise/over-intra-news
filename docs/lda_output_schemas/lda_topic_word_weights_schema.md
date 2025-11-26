# Parquet: lda_topic_word_weights

Purpose
-------
Store the unnormalized topic–word weights for every `(topic, token)` pair,
as produced by `mallet train-topics --topic-word-weights-file`, in a tidy
Parquet table.

This allows reconstruction or analysis of the full topic–word distribution
beyond truncated top-word lists.

Source
------
- Raw file: `data/lda_topic_words.txt`
- MALLET lines have the structure:
  `<topic_id>\t<word>\t<weight>`
- Little MALLET Wrapper parses this exactly as three tab-separated fields.

Columns
-------

| Column name | Type    | Description                                                                 |
|-------------|---------|-----------------------------------------------------------------------------|
| topic_id    | INT     | Topic index `k` in `0, 1, ..., K-1`.                                        |
| token       | TEXT    | Vocabulary token (same tokenization as `lda_vocabulary`).                   |
| weight      | DOUBLE  | Unnormalized topic–word weight from MALLET (non-negative real number).      |

Conventions
-----------
- For a fixed `(topic_id)`, normalizing `weight` over all tokens yields
  the topic–word probabilities used by MALLET’s LDA model.
- Tokens are in the same vocabulary as your training corpus
  (`lda_vocabulary`) and can be joined on the token string if needed.
- This table can be large (K × |V|), so Parquet + columnar scans matter.
