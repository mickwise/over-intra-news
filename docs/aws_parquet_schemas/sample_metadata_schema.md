# ccnews_sample_stats schema

S3 layout (logical):

s3://<news-bucket>/ccnews_sample_stats/
  year=2020/month=01/day=02/session=intraday/sample_stats.parquet

Note: Partition columns are encoded in the S3 path; the table also
carries `date` and `session` as regular columns for convenience.

## Row semantics

- One row = one WARC *sample* (one `.warc.gz` file) processed for a given
  New York trading date and session.
- Counts are aggregated over all records in that sample after applying
  parser gates.

## Columns

| Column          | Type          | Description                                                                                 |
|-----------------|---------------|---------------------------------------------------------------------------------------------|
| date            | date          | New York trading date (America/New_York calendar) aligned to `trading_calendar.trading_day`.|
| session         | string        | Logical session bucket: `"intraday"` or `"overnight"`.                                      |
| records_scanned | int32 / int64 | Total number of WARC records iterated in this sample (all record types).                    |
| html_200_count  | int32 / int64 | Count of records with `rec_type == "response"` and HTTP status `200`.                       |
| unhandled_errors| int32 / int64 | Amount of exceptions caught mid sample parsing.                                             |
| decompression_errors| int32 / int64 | Amount of decompression exceptions caught mid sample parsing.                           |
| ge_25_words     | int32 / int64 | Number of HTML 200 responses whose visible text length was ≥ 25 tokens after canonicalization.|
| too_long_articles| int32 / int64| Number of HTML 200 responses whose visible text length was ≥ MAXIMUM_ALLOWED_TOKENSafter canonicalization.|
| english_count   | int32 / int64 | Number of candidate articles where `langdetect.detect(...) == "en"`.                        |
| matched_any_firm| int32 / int64 | Number of English articles that matched at least one S&P 500 firm (by CIK).                 |
| articles_kept   | int32 / int64 | Final count of articles retained for this sample after all gates (status, length, lang, firms).|

## Notes

- Metrics are designed for monitoring parser health and selectivity over time.
- `records_scanned` provides a denominator for conversion-rate style diagnostics
  across subsequent filters.
- The schema is stable across years; new monitoring counters should be added as
  additional columns rather than repurposing existing ones.
