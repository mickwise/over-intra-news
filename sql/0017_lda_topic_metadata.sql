-- =============================================================================
-- lda_topic_metadata.sql
--
-- Purpose
--   Store lightweight metadata for each topic in an LDA model run, including
--   top terms, optional human-readable labels, and a junk flag.
--
-- Row semantics
--   One row = one topic within a particular LDA run.
--
-- Conventions
--   - top_terms is a small array of the highest-weight terms for inspection.
--   - human_label is a short free-text label assigned manually (nullable).
--   - is_junk marks purely mechanical / boilerplate topics that should be
--     down-weighted or ignored in regressions.
--
-- Keys & constraints
--   - Primary key: (run_id, topic_id).
--
-- Relationships
--   - run_id → lda_model_run.run_id (ON DELETE CASCADE).
--
-- Audit & provenance
--   - created_at records when metadata was first registered; subsequent edits
--     can be tracked via notes or external tooling.
-- =============================================================================
CREATE TABLE IF NOT EXISTS lda_topic_metadata (

    -- ===========
    -- Identifiers
    -- ===========

    -- LDA model run this topic belongs to
    run_id TEXT NOT NULL
    REFERENCES lda_run_registry (run_id) ON DELETE CASCADE,

    -- Topic identifier (0-based, consistent with MALLET)
    topic_id INTEGER NOT NULL,

    -- ===================
    -- Descriptive metadata
    -- ===================

    -- Highest-weight terms for this topic, in order
    top_terms TEXT [] NOT NULL,

    -- Topic coherence score (e.g., Cv)
    cv_coherence DOUBLE PRECISION NOT NULL,

    -- Optional human-readable label for the topic
    human_label TEXT,

    -- Flag to mark boilerplate / junk topics
    is_junk BOOLEAN NOT NULL DEFAULT FALSE,

    -- Free-form notes (e.g., labeling rationale)
    notes TEXT,

    -- ==========
    -- Provenance
    -- ==========

    -- Timestamp when this row was created
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- ===========
    -- Constraints
    -- ===========

    -- Primary key constraint
    CONSTRAINT ltm_pk
    PRIMARY KEY (run_id, topic_id)
);

-- Index to speed up queries by (run_id, is_junk)
CREATE INDEX IF NOT EXISTS idx_lda_topic_metadata_run_junk
ON lda_topic_metadata (run_id, is_junk);

COMMENT ON TABLE lda_topic_metadata IS
'Per-topic metadata for each LDA model run, including top terms,
optional human labels, and a junk flag.';

COMMENT ON COLUMN lda_topic_metadata.run_id IS
'LDA model run identifier this topic belongs to,
referencing lda_model_run.run_id.';

COMMENT ON COLUMN lda_topic_metadata.topic_id IS
'Topic identifier (0-based) for this run.';

COMMENT ON COLUMN lda_topic_metadata.top_terms IS
'Array of the highest-weight terms for this topic,
ordered by decreasing weight.';

COMMENT ON COLUMN lda_topic_metadata.human_label IS
'Optional concise human-readable label assigned to the topic.';

COMMENT ON COLUMN lda_topic_metadata.is_junk IS
'Boolean flag marking topics that are considered
junk or boilerplate for downstream analysis.';

COMMENT ON COLUMN lda_topic_metadata.notes IS
'Free-form notes describing labeling decisions or
other topic-level observations.';

COMMENT ON COLUMN lda_topic_metadata.created_at IS
'UTC timestamp when this topic metadata row was inserted.';

COMMENT ON INDEX idx_lda_topic_metadata_run_junk IS
'Index to accelerate retrieval of junk vs non-junk
topics within an LDA model run.';
