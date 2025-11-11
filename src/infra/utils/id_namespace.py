"""
id_namespace.py

Purpose
-------
Provide a single, fixed UUID namespace for generating deterministic IDs across
all ingestion scripts (e.g., uuid5(PROJECT_NAMESPACE, <name>)).

Key points
----------
- The namespace value is generated once (offline) and committed to the repo.
- Never rotate this value; changing it would change every derived ID.
- Import and reuse from this module wherever deterministic IDs are needed.

Downstream usage
----------------
from infra.utils.id_namespace import PROJECT_NAMESPACE
"""

import uuid

PROJECT_NAMESPACE: uuid.UUID = uuid.UUID("0df0b2a5-e0ba-426a-ab31-4ea724f356ce")
