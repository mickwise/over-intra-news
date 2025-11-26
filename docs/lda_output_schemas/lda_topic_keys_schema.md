# Parquet: lda_topic_keys

Purpose
-------
Store the human-readable top words for each topic, as produced by
`mallet train-topics --output-topic-keys`, in a tidy parquet file where
each row is a single `(topic, word_rank, token)`.

Source
------
- Raw file: `data/lda_output_topic_keys.txt`
- MALLET lines have the structure:
  `<topic_id>\t<topic_score>\t<word_1 word_2 ... word_M>`
- The parsing layer:
  - splits on `\t` to recover `topic_id`, `topic_score`, and the full
    top-words string,
  - then splits the words string on spaces and assigns a `word_rank`
    based on position within the list.

Columns
-------

| Column name  | Type    | Description                                                                 |
|--------------|---------|-----------------------------------------------------------------------------|
| topic_id     | INT     | Topic index `k` in `0, 1, ..., K-1`.                                        |
| topic_score  | DOUBLE  | Numeric topic-level score from the second column (often a log-likelihood).  |
| word_rank    | INT     | 1-based rank of the token within the topic’s top-word list.                 |
| token        | TEXT    | Surface form of the top token for this rank and topic.                      |

Conventions
-----------
- For a given `topic_id`, `word_rank` increases with decreasing probability
  (i.e., `word_rank = 1` is the most probable/top word).
- `topic_score` is constant across rows for a given `topic_id` and is
  simply repeated for convenience.
- The number of rows per topic equals the `--num-top-words` setting used
  in training.
