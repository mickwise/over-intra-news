"""
Purpose
-------
Configuration for security_master_profiles_membership utilities, including
canonicalization rules for company names and the staging table name used for
auto-accepted profile episodes.

Key behaviors
-------------
- Define NAME_SUFFIXES_TO_REMOVE, the list of legal-entity and jurisdiction
  suffixes that should be stripped during company-name canonicalization.
- Define AUTO_PICK_TABLE, the temporary/staging table name used when loading
  auto-accepted profile names into security_profile_history.

Conventions
-----------
- NAME_SUFFIXES_TO_REMOVE entries are specified in uppercase and are matched
  as whole words in canonicalization logic (optionally followed by a period).
- AUTO_PICK_TABLE is a short, snake_case identifier suitable for creating a
  transient table in the warehouse (e.g., via pandas.to_sql).
- This module is read-only at runtime; values are treated as constants and
  should not be mutated by downstream code.

Downstream usage
----------------
- Imported by security_master_profiles_memberships_utils and related modules
  to drive deterministic canonicalization and staging behavior.
"""

from typing import List

NAME_SUFFIXES_TO_REMOVE: List[str] = [
    "INC",
    "INCDE",
    "INCMD",
    "INCORPORATED",
    "CORP",
    "CORPORATION",
    "INTERNATIONAL",
    "GROUP",
    "CO",
    "COMPANY",
    "LLC",
    "LP",
    "L.P.",
    "LLLP",
    "LLP",
    "LTD",
    "LIMITED",
    "HOLDINGS",
    "HLDGS",
    "HOLDING",
    "PLC",
    "DE",
    "MD",
    "UT",
    "MO",
]

AUTO_PICK_TABLE = "auto_profile_name_candidates"
