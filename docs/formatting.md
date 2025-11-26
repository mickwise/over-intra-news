# Python

## Module doc:

"""
Purpose
-------
<What this module does and why it exists.>

Key behaviors
-------------
- <List the main operations or responsibilities.>
- <Include any important side effects, like writing to DB.>

Conventions
-----------
- <Assumptions, like timezone, unique keys, batching rules.>
- <Anything that downstream code can safely rely on.>

Downstream usage
----------------
<How other parts of the project should use this module.>
"""

## Function doc:
"""
<One-sentence summary of what the function does.>

Parameters
----------
param1 : <type>
    <Description of param1>
param2 : <type>
    <Description of param2>

Returns
-------
<type>
    <What is returned and under what conditions.>

Raises
------
<ExceptionType>
    <When this error is raised and why.>

Notes
-----
- <Any important implementation details, assumptions, or constraints.>
"""

## Class doc:
"""
Purpose
-------
<What this class represents and why it exists. One or two sentences.>

Key behaviors
-------------
- <Behavior #1>
- <Behavior #2>

Parameters
----------
param1 : <type>
    <Meaning, units, constraints.>
param2 : <type>
    <Meaning, units, constraints.>

Attributes
----------
attr1 : <type>
    <What it stores and when it changes.>
attr2 : <type>
    <What it stores and when it changes.>

Notes
-----
- <Important invariants or performance characteristics.>
- <Thread/process-safety expectations.>
"""


# Bash

## Module doc: 

# =============================================================================
# Name - description
#
# Purpose
#
# Assumptions
#
# Conventions
#
# IAM required (minimum)
#
# Usage
# 
# =============================================================================

## Function doc:

# Name
# -----------------------------
# Purpose
#
# Contract
#
# Effects
#
# Fails
#
# IAM required (minimum)
#
# Notes
#

# PostGres

## Module doc:
-- =============================================================================
-- <filename>.sql
--
-- Purpose
--   <What this table is for and why it exists. One to three sentences.>
--
-- Row semantics
--   <What one row represents. Clarify entity vs. episode vs. fact.>
--
-- Conventions
--   - <Key normalization rules (e.g., UPPER tickers, zero-padded CIKs).>
--   - <Timezone or date span policy (e.g., half-open [start, end)).>
--   - <Immutability/append-only expectations, if any.>
--
-- Keys & constraints
--   - Primary key: <column(s)>
--   - Natural keys / uniqueness: <notes>
--   - Checks: <format/quality guards worth calling out>
--
-- Relationships
--   - <FKs this table owns or is expected to receive in downstream tables.>
--   - <How other tables are expected to join to this one.>
--
-- Audit & provenance
--   <What lineage is (or is not) stored here; where full provenance lives.>
--
-- Performance
--   <Indexes or partitioning choices and the query patterns they serve.>
--
-- Change management
--   <How to extend this schema without breaking downstream (e.g., add-only).>
-- =============================================================================
