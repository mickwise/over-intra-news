-- =============================================================================
-- news_sample_stats.sql
--
-- Purpose
--   Store per-(trading_day, session) aggregate statistics for the CC-NEWS
--   parsing pipeline. Each row summarizes how many WARC records and articles
--   passed each gating stage for a given NYSE trading_day and session.
--
-- Row semantics
--   One row = aggregate parser stats for a single (trading_day, session)
--   pair. Counts represent totals across all WARC samples processed for
--   that trading_day and session.
--
-- Conventions
--   - trading_day is the NYSE trading date (America/New_York) corresponding to
--     trading_calendar.trading_day.
--   - session is either 'intraday' or 'overnight', matching the return
--     decomposition and parsed_news_articles.session.
--   - All statistic columns are non-negative integers; zero values are
--     allowed (e.g., no articles kept for a trading_day/session).
--
-- Keys & constraints
--   - Primary key: (trading_day, session).
--   - Checks:
--       - session must be 'intraday' or 'overnight'.
--       - All statistic fields must be >= 0.
--
-- Relationships
--   - trading_day and session align with parsed_news_articles.trading_day and
--     parsed_news_articles.session, enabling joins to compare throughput
--     and kept-article counts.
--   - trading_day should join to trading_calendar.trading_day
--     for alignment with trading sessions and return horizons.
--
-- Audit & provenance
--   - This table stores only aggregate counters, not record-level lineage.
--   - Detailed per-sample statistics live in sample-level Parquet outputs
--     and parser logs; use this table for monitoring and sanity checks
--     rather than full audit trails.
--
-- Performance
--   - The primary key on (trading_day, session) provides a btree index suitable
--     for equality lookups and range scans over trading_day.
--   - Expected access patterns are:
--       - time-sliced monitoring (e.g., WHERE trading_day BETWEEN ...),
--       - joins to parsed_news_articles on (trading_day, session).
--
-- Change management
--   - Schema evolution should prefer add-only changes (new nullable
--     statistic columns) to avoid breaking downstream consumers.
-- =============================================================================

CREATE TABLE IF NOT EXISTS news_sample_stats (
    -- ===========
    -- Identifiers
    -- ===========

    -- Date of the news sample
    trading_day DATE NOT NULL,

    -- Session of the news sample (e.g., 'intraday', 'overnight')
    session TEXT NOT NULL,

    -- =================
    -- Sample Statistics
    -- =================

    -- Total number of records scanned
    records_scanned INT NOT NULL,

    -- Total number of HTTP 200 OK records
    html_200_count INT NOT NULL,

    -- Total number of unhandled errors
    unhandled_errors INT NOT NULL,

    -- Total number of decompression errors
    decompression_errors INT NOT NULL,

    -- Total number of articles with at least 25 words
    ge_25_words INT NOT NULL,

    -- Total number of articles with greater than MAXIMUM_ALLOWED_TOKENS (3000)
    too_long_articles INT NOT NULL,

    -- Total number of English articles
    english_count INT NOT NULL,

    -- Total number of articles that matched with at least one firm
    matched_any_firm INT NOT NULL,

    -- Total number of articles kept
    articles_kept INT NOT NULL,

    -- ===========
    -- Constraints
    -- ===========

    -- Primary key on (trading_day, session)
    PRIMARY KEY (trading_day, session),

    -- Ensure session is either 'intraday' or 'overnight'
    CONSTRAINT nss_session_check CHECK (session IN ('intraday', 'overnight')),

    -- Ensure non-negative statistics
    CONSTRAINT nss_non_negative_stats CHECK (
        records_scanned >= 0
        AND
        html_200_count >= 0
        AND
        unhandled_errors >= 0
        AND
        decompression_errors >= 0
        AND
        ge_25_words >= 0
        AND
        too_long_articles >= 0
        AND
        english_count >= 0
        AND
        matched_any_firm >= 0
        AND
        articles_kept >= 0
    )

);


COMMENT ON TABLE news_sample_stats IS
'Per-(trading_day, session) aggregate statistics for the CC-NEWS parser,
capturing counts at each gating stage for monitoring and QA.';

COMMENT ON COLUMN news_sample_stats.trading_day IS
'NYSE trading date (America/New_York date) over which parser stats are
aggregated; aligns with trading_calendar.trading_day.';

COMMENT ON COLUMN news_sample_stats.session IS
'Parser session bucket: either ''intraday'' or ''overnight'', matching
the session used in parsed_news_articles.session.';

COMMENT ON COLUMN news_sample_stats.records_scanned IS
'Total number of WARC records iterated over for this (trading_day, session),
including both response and non-response records.';

COMMENT ON COLUMN news_sample_stats.html_200_count IS
'Number of response records with HTTP status 200 and an HTML content type
encountered for this (trading_day, session).';

COMMENT ON COLUMN news_sample_stats.unhandled_errors IS
'Number of WARC records that resulted in unhandled parser errors
 during processing.';

COMMENT ON COLUMN news_sample_stats.decompression_errors IS
'Number of WARC records that could not be decompressed due to
unrecognized or invalid compression formats.';

COMMENT ON COLUMN news_sample_stats.ge_25_words IS
'Number of HTML 200 responses whose cleaned visible ASCII text contains
at least 25 tokens.';

COMMENT ON COLUMN news_sample_stats.too_long_articles IS
'Number of HTML 200 articles whose cleaned visible ASCII text exceeds
MAXIMUM_ALLOWED_TOKENS (3000 tokens) and were therefore excluded from
further processing.';

COMMENT ON COLUMN news_sample_stats.english_count IS
'Number of >= 25-word articles whose language detector classified them
as English (e.g., langdetect == ''en'').';

COMMENT ON COLUMN news_sample_stats.matched_any_firm IS
'Number of English articles for which at least one firm was matched by
the name-based detection logic.';

COMMENT ON COLUMN news_sample_stats.articles_kept IS
'Number of articles that passed all gating conditions (HTML 200, >= 25
words, English language, between 1 and 3 matched firms) and were kept
for insertion into parsed_news_articles.';

COMMENT ON CONSTRAINT nss_session_check ON news_sample_stats IS
'Ensures session is constrained to the two valid buckets: intraday and
overnight.';

COMMENT ON CONSTRAINT nss_non_negative_stats ON news_sample_stats IS
'Guards against negative aggregate counts for any of the statistics
columns.';

COMMENT ON INDEX news_sample_stats_pkey IS
'Btree index backing the primary key (trading_day, session); supports equality
lookups and time-range scans for monitoring and joins.';
