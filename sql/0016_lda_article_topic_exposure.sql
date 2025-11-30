-- =============================================================================
-- lda_article_topic_exposure.sql
--
-- Purpose
--   Store per-article, per-topic exposure vectors produced by MALLET LDA for
--   a given model run, aligned to the article_id in the lda_documents table.
--
-- Row semantics
--   One row = one (run_id, article_id, corpus_version, topic_id) pair with a
--   topic_exposure in [0, 1].
--
-- Conventions
--   - article_id and corpus_version match lda_documents.article_id and
--     lda_documents.corpus_version.
--   - run_id references the trained LDA model in lda_model_run.
--   - topic_exposure is the (possibly truncated) topic weight emitted by
--     MALLET’s doc-topics output (may not sum exactly
--     to 1.0 due to thresholding).
--
-- Keys & constraints
--   - Primary key: (run_id, article_id, corpus_version, topic_id).
--   - Checks: topic_exposure between 0 and 1 inclusive.
--
-- Relationships
--   - run_id → lda_model_run.run_id (ON DELETE CASCADE).
--   - (article_id, corpus_version) → lda_documents(article_id, corpus_version)
--     (ON DELETE CASCADE).
--
-- Audit & provenance
--   - created_at records when the exposure row was materialized from MALLET
--     outputs into the relational store.
--
-- Performance
--   - Index on (run_id, topic_id) to support cross-sectional queries by topic.
--   - Primary key already covers common joins by
--     (run_id, article_id, topic_id).
-- =============================================================================
CREATE TABLE IF NOT EXISTS lda_article_topic_exposure (

    -- ===========
    -- Identifiers
    -- ===========

    -- LDA model run responsible for these exposures
    run_id TEXT NOT NULL REFERENCES lda_run_registry (run_id) ON DELETE CASCADE,

    -- Article identifier, aligned with lda_documents.article_id
    article_id TEXT NOT NULL,

    -- Corpus version associated with the article and the model
    corpus_version SMALLINT NOT NULL,

    -- Topic identifier (0-based, as used by MALLET)
    topic_id INTEGER NOT NULL,

    -- ==================
    -- Exposure magnitude
    -- ==================

    -- Topic exposure for this (article, topic) pair
    topic_exposure DOUBLE PRECISION NOT NULL,

    -- ==========
    -- Provenance
    -- ==========

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- ===========
    -- Constraints
    -- ===========

    -- Composite primary key over the logical dimension tuple
    CONSTRAINT late_pk
    PRIMARY KEY (run_id, article_id, corpus_version, topic_id),

    -- Ensure exposures lie in [0, 1]; MALLET may not sum to 1.0
    CONSTRAINT late_chk_proportion
    CHECK (topic_exposure >= 0.0 AND topic_exposure <= 1.0),

    -- Foreign key into lda_documents to align with article metadata
    CONSTRAINT late_fk_article
    FOREIGN KEY (article_id, corpus_version)
    REFERENCES lda_documents (article_id, corpus_version)
    ON DELETE CASCADE

);

-- Index to speed up queries by (run_id, topic_id)
CREATE INDEX IF NOT EXISTS idx_lda_article_topic_exposure_run_topic
ON lda_article_topic_exposure (run_id, topic_id);

COMMENT ON TABLE lda_article_topic_exposure IS
'Per-article, per-topic exposure vectors for a given LDA model run,
derived from MALLET doc-topics outputs.';

COMMENT ON COLUMN lda_article_topic_exposure.run_id IS
'Identifier of the LDA model run that produced these topic exposures.';

COMMENT ON COLUMN lda_article_topic_exposure.article_id IS
'Article identifier, matching lda_documents.article_id.';

COMMENT ON COLUMN lda_article_topic_exposure.corpus_version IS
'Corpus version shared with lda_documents.corpus_version for this article.';

COMMENT ON COLUMN lda_article_topic_exposure.topic_id IS
'Integer topic identifier (0-based) as emitted by MALLET.';

COMMENT ON COLUMN lda_article_topic_exposure.topic_exposure IS
'Topic exposure for the given (run_id, article_id, corpus_version, topic_id);
constrained to [0, 1].';

COMMENT ON COLUMN lda_article_topic_exposure.created_at IS
'UTC timestamp when this exposure row was inserted into the database.';

COMMENT ON CONSTRAINT late_pk
ON lda_article_topic_exposure IS
'Primary key enforcing uniqueness of
(run_id, article_id, corpus_version, topic_id).';

COMMENT ON INDEX idx_lda_article_topic_exposure_run_topic IS
'Index to accelerate retrieval of all articles
exposed to a given topic within an LDA run.';
