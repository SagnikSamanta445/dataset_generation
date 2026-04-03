"""
config.py — Central configuration for the DB Normalization data generation pipeline.
Edit these values before running generate.py.
"""

import os

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "sk-YOUR_KEY_HERE")
OPENAI_MODEL:   str = "gpt-4o-mini"          # cheap + fast; swap to gpt-4o for quality
OPENAI_TIMEOUT: int = 60                      # seconds per request

# ── Data source ───────────────────────────────────────────────────────────────
HF_SOURCE_DATASET: str = "wikitablequestions" # loaded from HuggingFace Hub
HF_SOURCE_SPLIT:   str = "train"
MAX_SOURCE_TABLES: int = 500                  # how many clean tables to pull

# ── Perturbation ──────────────────────────────────────────────────────────────
ANOMALY_CATEGORIES = [
    "value_swap",             # swap values between two rows in the same column
    "arithmetic_error",       # introduce a deliberate miscalculation
    "temporal_impossibility", # break chronological ordering / impossible date
    "unit_inconsistency",     # silently mix units (km vs miles, USD vs EUR)
    "duplicate_conflict",     # duplicate a row but alter one field to conflict
]

# How many corrupted variants to generate per clean table (1–3 recommended)
VARIANTS_PER_TABLE: int = 1

# Seed base — seeds are deterministic: seed = SEED_BASE + global_index
SEED_BASE: int = 42

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR:         str = "./output"
HF_REPO_ID:         str = ""     # e.g. "your-username/db-norm-rl-dataset"; leave "" to skip push
HF_DATASET_NAME:    str = "db_normalization_rl"
PUSH_TO_HUB:        bool = False  # set True + fill HF_REPO_ID to push

# ── Prompt template ───────────────────────────────────────────────────────────
# This is the conversational prompt passed to GRPOTrainer.
# {table_csv} is replaced with the corrupted table at runtime.
AGENT_PROMPT_TEMPLATE: str = (
    "You are a database normalization expert. "
    "The following table contains one or more data anomalies. "
    "Identify every anomaly, explain why it is invalid, and return a corrected version of the full table.\n\n"
    "Table:\n{table_csv}"
)
