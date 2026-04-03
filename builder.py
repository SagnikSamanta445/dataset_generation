"""
builder.py — Assemble PerturbedTable records into a HuggingFace Dataset
             and optionally push it to the Hub.

Dataset schema (one row per perturbed table):
  seed                 int     — deterministic seed for reset(seed) in OpenEnv
  prompt               str     — conversational prompt fed to GRPOTrainer
  corrupted_table_csv  str     — the anomalous table (agent's starting state)
  original_table_csv   str     — clean ground truth (used for reward computation)
  anomaly_category     str     — one of the five categories in config.py
  anomaly_description  str     — GPT's description of what was changed
  source_table_id      str     — provenance: WikiTQ example index
  num_rows             int     — row count of the corrupted table
  num_cols             int     — column count
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset, DatasetDict

from config import (
    AGENT_PROMPT_TEMPLATE,
    HF_DATASET_NAME,
    HF_REPO_ID,
    OUTPUT_DIR,
    PUSH_TO_HUB,
)
from perturbations import PerturbedTable

logger = logging.getLogger(__name__)


# ── Row serialisation ─────────────────────────────────────────────────────────

def _perturbed_to_row(pt: PerturbedTable) -> dict:
    prompt = AGENT_PROMPT_TEMPLATE.format(table_csv=pt.corrupted_csv)
    return {
        "seed":                pt.seed,
        "prompt":              prompt,
        "corrupted_table_csv": pt.corrupted_csv,
        "original_table_csv":  pt.original_csv,
        "anomaly_category":    pt.anomaly_category,
        "anomaly_description": pt.anomaly_description,
        "source_table_id":     pt.source_table_id,
        "num_rows":            len(pt.corrupted_rows),
        "num_cols":            len(pt.corrupted_header),
    }


# ── Main builder ──────────────────────────────────────────────────────────────

def build_dataset(perturbed_tables: list[PerturbedTable]) -> Dataset:
    """Convert a list of PerturbedTable objects into a HuggingFace Dataset."""
    rows = [_perturbed_to_row(pt) for pt in perturbed_tables]
    df   = pd.DataFrame(rows)

    # Sort by seed for reproducibility
    df = df.sort_values("seed").reset_index(drop=True)

    logger.info(
        "Built dataset: %d rows | categories: %s",
        len(df),
        df["anomaly_category"].value_counts().to_dict(),
    )

    return Dataset.from_pandas(df)


def train_test_split_dataset(
    ds: Dataset,
    test_size: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """Split into train/test and return a DatasetDict."""
    split = ds.train_test_split(test_size=test_size, seed=seed)
    return DatasetDict({"train": split["train"], "test": split["test"]})


# ── Persistence ───────────────────────────────────────────────────────────────

def save_dataset(ds_dict: DatasetDict, output_dir: Optional[str] = None) -> Path:
    """
    Save dataset to disk as Parquet (per split).
    Returns the output directory Path.
    """
    out = Path(output_dir or OUTPUT_DIR) / HF_DATASET_NAME
    out.mkdir(parents=True, exist_ok=True)

    ds_dict.save_to_disk(str(out))

    # Also write a human-readable JSONL for quick inspection
    for split_name, split_ds in ds_dict.items():
        jsonl_path = out / f"{split_name}.jsonl"
        with open(jsonl_path, "w") as f:
            for row in split_ds:
                f.write(json.dumps(row) + "\n")
        logger.info("Saved %s → %s (%d rows)", split_name, jsonl_path, len(split_ds))

    # Write a dataset card
    card_path = out / "README.md"
    card_path.write_text(_make_dataset_card(ds_dict))
    logger.info("Dataset card written → %s", card_path)

    logger.info("Full dataset saved to %s", out)
    return out


def push_to_hub(ds_dict: DatasetDict) -> None:
    """Push DatasetDict to HuggingFace Hub (requires HF_REPO_ID in config)."""
    if not HF_REPO_ID:
        logger.warning("HF_REPO_ID is empty — skipping Hub push.")
        return
    logger.info("Pushing to Hub: %s …", HF_REPO_ID)
    ds_dict.push_to_hub(HF_REPO_ID, private=False)
    logger.info("Push complete.")


# ── Dataset card ──────────────────────────────────────────────────────────────

def _make_dataset_card(ds_dict: DatasetDict) -> str:
    total = sum(len(v) for v in ds_dict.values())
    splits = {k: len(v) for k, v in ds_dict.items()}
    return f"""---
language:
  - en
license: apache-2.0
task_categories:
  - table-question-answering
  - reinforcement-learning
tags:
  - openenv
  - db-normalization
  - rl
  - grpo
  - synthetic
---

# DB Normalization RL Dataset

Synthetic dataset of **anomalous tables** for training a database normalization agent
inside the [OpenEnv](https://github.com/huggingface/openenv-course) framework with GRPO.

## Dataset stats
| Split | Rows |
|-------|------|
{chr(10).join(f'| {k} | {v} |' for k, v in splits.items())}
| **Total** | **{total}** |

## Schema

| Column | Description |
|--------|-------------|
| `seed` | Deterministic seed for `reset(seed)` in OpenEnv |
| `prompt` | Conversational prompt passed to `GRPOTrainer` |
| `corrupted_table_csv` | The anomalous table (agent's starting observation) |
| `original_table_csv` | Clean ground truth (used to compute rewards) |
| `anomaly_category` | One of: `value_swap`, `arithmetic_error`, `temporal_impossibility`, `unit_inconsistency`, `duplicate_conflict` |
| `anomaly_description` | Precise description of the injected anomaly |
| `source_table_id` | Provenance — WikiTQ example index |
| `num_rows` / `num_cols` | Table dimensions |

## Generation methodology
Clean tables sourced from **WikiTableQuestions (WikiTQ)**. Anomalies injected via
targeted GPT prompting following the TABARD benchmark methodology.
"""
