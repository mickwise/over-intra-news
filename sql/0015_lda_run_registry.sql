-- =============================================================================
-- lda_model_run.sql
--
-- Purpose
--   Register each trained LDA model run, including its hyperparameters,
--   training window, and S3 storage location for downstream reproducibility.
--
-- Row semantics
--   One row = one LDA model generation identified by run_id, typically tied
--   to a specific corpus_version and training window.
--
-- Conventions
--   - run_id is a human-readable identifier (e.g. over_intra_v1_training).
--   - corpus_version aligns with lda_documents.corpus_version and indicates
--     which corpus definition underlies the model.
--   - alpha_sum and beta are stored as the scalar hyperparameters passed to
--     MALLET; optimization flags record how MALLET was allowed to adapt them.
--   - lda_results_s3_bucket / lda_results_s3_prefix point to Parquet outputs
--     produced by lda_output_parse (doc-topics, topic-word weights, etc.).
--
-- Keys & constraints
--   - Primary key: run_id (synthetic logical identifier).
--   - Checks: num_topics ≥ 1; num_iterations ≥ 1; alpha_sum, beta > 0 when
--     present; training_end_date ≥ training_start_date when both are set;
--     bucket/prefix must be non-empty after TRIM.
--
-- Relationships
--   - Referenced by lda_article_topic_exposure.run_id and
--     lda_topic_metadata.run_id.
--
-- Audit & provenance
--   - created_at records when the run was registered.
--   - notes can store free-form comments (e.g. “Glasserman baseline v1”).
--
-- Performance
--   - Index on corpus_version to support fast lookup of all runs trained on
--     a given corpus.
--
-- Change management
--   - New model generations should create new run_id values rather than
--     mutating existing rows, preserving a clear audit trail.
-- =============================================================================
CREATE TABLE IF NOT EXISTS lda_run_registry (

    -- ===========
    -- Identifiers
    -- ===========

    -- Logical identifier for the LDA model generation
    run_id TEXT PRIMARY KEY,

    -- Underlying corpus definition used for training
    corpus_version SMALLINT NOT NULL,

    -- =====================
    -- Model hyperparameters
    -- =====================

    -- Number of topics in the model
    num_topics INTEGER NOT NULL,

    -- Number of Gibbs iterations used for training
    num_iterations INTEGER NOT NULL,

    -- Total Dirichlet mass over topics (alpha sum); NULL if defaulted
    alpha_sum DOUBLE PRECISION,

    -- Dirichlet hyperparameter for topic–word distributions (beta);
    -- NULL if defaulted
    beta DOUBLE PRECISION,

    -- MALLET optimize-interval (in iterations); NULL if default
    optimize_interval INTEGER,

    -- MALLET optimize-burn-in (in iterations); NULL if default
    optimize_burn_in INTEGER,

    -- Whether a symmetric alpha prior was enforced
    symmetric_alpha BOOLEAN,

    -- RNG seed used when training the model; Null if default
    random_seed INTEGER,

    -- ==========================
    -- Training window & location
    -- ==========================

    -- Inclusive start of the training period (e.g., first article date)
    training_start_date DATE NOT NULL,

    -- Inclusive end of the training period
    training_end_date DATE NOT NULL,

    -- S3 bucket holding Parquet outputs for this run
    lda_results_s3_bucket TEXT NOT NULL,

    -- S3 prefix under the bucket where outputs are stored
    lda_results_s3_prefix TEXT NOT NULL,

    -- ==========
    -- Provenance
    -- ==========

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Free-form comments / labels for this run
    notes TEXT,

    -- ===========
    -- Constraints
    -- ===========

    -- Ensure num topics is at least 1
    CONSTRAINT lmr_chk_num_topics
    CHECK (num_topics >= 1),

    -- Ensure num iterations is at least 1
    CONSTRAINT lmr_chk_num_iterations
    CHECK (num_iterations >= 1),

    -- Ensure alpha_sum is positive if set
    CONSTRAINT lmr_chk_alpha_sum
    CHECK (alpha_sum > 0),

    -- Ensure beta is positive if set
    CONSTRAINT lmr_chk_beta
    CHECK (beta IS NULL OR beta > 0),

    -- Ensure training_end_date >= training_start_date
    CONSTRAINT lmr_chk_training_window
    CHECK (training_end_date >= training_start_date),

    -- Ensure S3 bucket is not empty after trimming whitespace
    CONSTRAINT lmr_chk_bucket_not_empty
    CHECK (LENGTH(TRIM(lda_results_s3_bucket)) > 0),

    -- Ensure S3 prefix is not empty after trimming whitespace
    CONSTRAINT lmr_chk_prefix_not_empty
    CHECK (LENGTH(TRIM(lda_results_s3_prefix)) > 0)

);

-- Index to speed up queries by corpus_version
CREATE INDEX IF NOT EXISTS idx_lda_model_run_corpus_version
ON lda_model_run (corpus_version);

COMMENT ON TABLE lda_model_run IS
'Registry of LDA model generations, including hyperparameters,
training window, and S3 location for Parquet outputs.';

COMMENT ON COLUMN lda_model_run.run_id IS
'Logical identifier for the LDA model generation
(e.g., over_intra_v1_training).';

COMMENT ON COLUMN lda_model_run.corpus_version IS
'Corpus definition used for this model, aligned with
lda_documents.corpus_version.';

COMMENT ON COLUMN lda_model_run.num_topics IS
'Number of latent topics in the LDA model.';

COMMENT ON COLUMN lda_model_run.num_iterations IS
'Number of Gibbs sampling iterations used during training.';

COMMENT ON COLUMN lda_model_run.alpha_sum IS
'Total Dirichlet mass over topics (alpha sum) as passed to MALLET;
NULL if MALLET default was used.';

COMMENT ON COLUMN lda_model_run.beta IS
'Dirichlet hyperparameter for topic–word distributions (beta);
NULL if MALLET default was used.';

COMMENT ON COLUMN lda_model_run.optimize_interval IS
'MALLET optimize-interval (in iterations) controlling how often
alpha is re-estimated; NULL if Default.';

COMMENT ON COLUMN lda_model_run.optimize_burn_in IS
'MALLET optimize-burn-in; number of iterations before
alpha optimization begins; NULL if default.';

COMMENT ON COLUMN lda_model_run.symmetric_alpha IS
'Boolean flag indicating whether MALLET used a symmetric alpha prior.';

COMMENT ON COLUMN lda_model_run.random_seed IS
'Random seed supplied to MALLET when training this model.';

COMMENT ON COLUMN lda_model_run.training_start_date IS
'Inclusive start date of the article sample used for model training.';

COMMENT ON COLUMN lda_model_run.training_end_date IS
'Inclusive end date of the article sample used for model training.';

COMMENT ON COLUMN lda_model_run.lda_results_s3_bucket IS
'S3 bucket name that holds Parquet outputs for this LDA run
(doc-topics, topic-word weights, etc.).';

COMMENT ON COLUMN lda_model_run.lda_results_s3_prefix IS
'Base S3 key prefix under lda_results_s3_bucket where Parquet
outputs for this run are stored.';

COMMENT ON COLUMN lda_model_run.created_at IS
'UTC timestamp when this LDA model run was registered in the catalog.';

COMMENT ON COLUMN lda_model_run.notes IS
'Free-form notes and labels describing this LDA run.';

COMMENT ON INDEX idx_lda_model_run_corpus_version IS
'Index to accelerate lookup of all LDA runs associated
with a given corpus_version.';
