# Parquet: ccnews_articles

S3 layout (logical):

s3://<news-bucket>/ccnews_articles/year=2020/month=01/day=02/session=intraday/articles.parquet

Note: Partition columns are encoded in the S3 path; the table also carries `ny_date`
and `session` as regular columns for convenience.

## Row semantics

- One row = one filtered article associated with at least one S&P 500 firm (by CIK)
  on a given New York trading date and session.
- Only HTML responses with HTTP 200 status, ≥ 25 words of visible text, detected
  English language, and at least one firm match (≤ 3 firms) are retained.

## Columns

| Column            | Type           | Description                                                                                          |
|-------------------|----------------|------------------------------------------------------------------------------------------------------|
| warc_path         | string         | Full S3 URI of the source WARC file from which this article was extracted.                          |
| warc_date_utc     | string (ISO)   | WARC capture timestamp in UTC from `WARC-Date` header (for example, `2020-01-02T03:04:05Z`).        |
| url               | string         | Original article URL from `WARC-Target-URI`.                                                        |
| http_status       | int32 / int64  | HTTP status code for the response (only `200` rows are retained).                                   |
| http_content_type | string         | Value of `Content-Type` header as normalized lower-case string.                                     |
| payload_digest    | string         | WARC payload digest (for example, `sha1:...`) for deduplication and audit.                          |
| ny_date           | date           | New York trading date (America/New_York calendar) aligned to `trading_calendar.trading_day`.        |
| session           | string         | Logical session bucket: `"intraday"` or `"overnight"`.                                              |
| cik_list          | array<string>  | List of matched firm CIKs (length 1–3) based on canonicalized firm-name tokens.                     |
| word_count        | int32 / int64  | Count of whitespace-delimited tokens in the visible ASCII text after filtering.                     |
| language_confidence| float         | langdetect output probability of English detected.                                                  |
| full_text         | string         | Upper-cased, ASCII-only visible text extracted from HTML after stripping non-visible tags.          |

## Notes

- Table is optimized for downstream topic modeling and text featurization.
- `full_text` is intentionally denormalized for simplicity; storage cost is traded
  for easier reproducibility and inspection.
- `cik_list` enables many-to-one mapping between articles and S&P 500 constituents
  without exploding rows per firm.
