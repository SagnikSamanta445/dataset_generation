"""
sources.py — Load clean ground-truth tables from WikiTableQuestions (WikiTQ).

WikiTQ is available on the HuggingFace Hub as 'wikitablequestions'.
Each example has:
  - 'table'  : {'header': [...], 'rows': [[...], ...]}
  - 'question': natural-language question over the table
  - 'answers' : list of gold answers

We extract (header, rows) pairs and convert them to CSV strings.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from typing import Iterator

import pandas as pd
from datasets import load_dataset

from config import HF_SOURCE_DATASET, HF_SOURCE_SPLIT, MAX_SOURCE_TABLES

logger = logging.getLogger(__name__)


@dataclass
class CleanTable:
    table_id:  str            # stable identifier from the source dataset
    header:    list[str]
    rows:      list[list[str]]
    question:  str            # original WikiTQ question (useful metadata)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def to_csv(self) -> str:
        """Serialise to CSV string (header + rows)."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(self.header)
        writer.writerows(self.rows)
        return buf.getvalue().strip()

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.header)

    @property
    def num_rows(self) -> int:
        return len(self.rows)

    @property
    def num_cols(self) -> int:
        return len(self.header)


# ── Filtering heuristics ──────────────────────────────────────────────────────

def _is_usable(table: CleanTable) -> bool:
    """
    Drop tables that are too small/large or have blank headers —
    perturbation prompts become nonsensical on degenerate tables.
    """
    if table.num_rows < 3 or table.num_rows > 40:
        return False
    if table.num_cols < 3 or table.num_cols > 12:
        return False
    if any(h.strip() == "" for h in table.header):
        return False
    return True


# ── Main loader ───────────────────────────────────────────────────────────────

def load_clean_tables() -> list[CleanTable]:
    """
    Stream WikiTQ from the Hub, parse each example, apply quality filters,
    and return up to MAX_SOURCE_TABLES CleanTable objects.
    """
    logger.info("Loading '%s' (split=%s) from HuggingFace Hub …",
                HF_SOURCE_DATASET, HF_SOURCE_SPLIT)

    ds = load_dataset(HF_SOURCE_DATASET, split=HF_SOURCE_SPLIT, streaming=True)

    tables: list[CleanTable] = []
    seen_ids: set[str] = set()

    for idx, example in enumerate(ds):
        if len(tables) >= MAX_SOURCE_TABLES:
            break

        raw_table = example.get("table", {})
        header: list[str] = raw_table.get("header", [])
        rows:   list[list[str]] = raw_table.get("rows", [])

        # Deduplicate by content fingerprint (same table reused across questions)
        fingerprint = str(header) + str(rows[:2])
        if fingerprint in seen_ids:
            continue
        seen_ids.add(fingerprint)

        ct = CleanTable(
            table_id=f"wikitq_{idx:05d}",
            header=header,
            rows=rows,
            question=example.get("question", ""),
        )

        if not _is_usable(ct):
            continue

        tables.append(ct)

    logger.info("Loaded %d usable clean tables.", len(tables))
    return tables


def iter_clean_tables() -> Iterator[CleanTable]:
    """Streaming version — yields one table at a time (memory-efficient)."""
    yield from load_clean_tables()
