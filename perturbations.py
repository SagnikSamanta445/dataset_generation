"""
perturbations.py — Inject category-specific anomalies into clean tables via OpenAI.

Strategy (mirrors TABARD methodology from the doc):
  - Send the clean table + anomaly category to GPT.
  - GPT returns a JSON blob with the corrupted table + a textual description
    of exactly what was changed (used later as the ground-truth reward signal).
  - We validate the response structurally before accepting it.

Supported anomaly categories (defined in config.ANOMALY_CATEGORIES):
  value_swap            — swaps values between two rows in the same column
  arithmetic_error      — introduces a deliberate miscalculation in a numeric cell
  temporal_impossibility— breaks chronological ordering or creates an impossible date
  unit_inconsistency    — silently mixes measurement/currency units
  duplicate_conflict    — duplicates a row but alters one field to create a conflict
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TIMEOUT,
    ANOMALY_CATEGORIES,
)
from sources import CleanTable

logger = logging.getLogger(__name__)

# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class PerturbedTable:
    source_table_id:    str
    anomaly_category:   str
    anomaly_description: str          # GPT's explanation of what it changed
    original_csv:       str
    corrupted_csv:      str
    corrupted_header:   list[str]
    corrupted_rows:     list[list[str]]
    seed:               int
    raw_gpt_response:   str = field(repr=False)


# ── Prompt templates per anomaly category ────────────────────────────────────

_CATEGORY_INSTRUCTIONS: dict[str, str] = {
    "value_swap": (
        "Swap the values of the SAME column between exactly TWO different rows. "
        "The swap must make the data factually inconsistent (e.g., population counts "
        "swapped between two countries so the numbers contradict known relative sizes). "
        "Do NOT swap values that happen to be identical."
    ),
    "arithmetic_error": (
        "Locate a numeric column where values in other columns imply a computable result "
        "(e.g., total = price × quantity, average = sum / count). "
        "Alter exactly ONE cell in that column so the arithmetic is visibly wrong "
        "by a meaningful but non-obvious amount (not simply doubling or halving)."
    ),
    "temporal_impossibility": (
        "Find a date or year column. Introduce exactly ONE of: "
        "(a) a date that is chronologically impossible given sibling rows (e.g., end date before start date), "
        "(b) a founding/birth year in the future, or "
        "(c) a duration that contradicts the start/end dates present in the table. "
        "Keep the cell format identical (same date format as the original)."
    ),
    "unit_inconsistency": (
        "Identify a column whose values share an implicit unit (distances in km, prices in USD, "
        "weights in kg, etc.). Change exactly ONE cell's value so it silently represents a "
        "DIFFERENT unit without modifying the column header "
        "(e.g., change '5 km' to '5 miles' by only altering the number to reflect the wrong-unit value, "
        "keeping the header unchanged). The inconsistency must be subtle."
    ),
    "duplicate_conflict": (
        "Duplicate exactly ONE row verbatim, then alter a SINGLE non-key field in the duplicate "
        "to create a factual conflict (e.g., same person/entity with two different nationalities, "
        "or same product with two different prices). Insert the duplicate adjacent to the original row."
    ),
}

_SYSTEM_PROMPT = """You are a data quality testing assistant.
Your job is to corrupt a clean CSV table by injecting a specific, realistic-looking anomaly.
The corruption must:
  1. Preserve the exact CSV structure (same headers, same number of columns per row).
  2. Be subtle — the table should still look plausible at a glance.
  3. Affect the MINIMUM number of cells needed to introduce the anomaly (usually 1-2 cells).

Respond with ONLY a valid JSON object — no markdown fences, no extra text.
Schema:
{
  "corrupted_csv": "<full CSV string with header and all rows, newlines as \\n>",
  "anomaly_description": "<1-3 sentences precisely describing what was changed and why it is an anomaly>"
}"""


# ── OpenAI client ─────────────────────────────────────────────────────────────

def _make_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT)


_client: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = _make_client()
    return _client


# ── Core perturbation logic ───────────────────────────────────────────────────

def _build_user_message(table: CleanTable, category: str) -> str:
    instructions = _CATEGORY_INSTRUCTIONS[category]
    return (
        f"Anomaly category: {category}\n"
        f"Instruction: {instructions}\n\n"
        f"Clean table (CSV):\n{table.to_csv()}"
    )


def _call_openai(user_message: str, retries: int = 3) -> str:
    client = _get_client()
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("OpenAI call failed (attempt %d/%d): %s", attempt + 1, retries, exc)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)   # exponential back-off
    raise RuntimeError("All OpenAI retries exhausted.")


def _parse_response(raw: str, original_csv: str) -> tuple[str, str]:
    """
    Parse GPT JSON → (corrupted_csv, anomaly_description).
    Falls back gracefully if the response is malformed.
    """
    try:
        blob = json.loads(raw)
        corrupted_csv       = blob["corrupted_csv"].strip()
        anomaly_description = blob["anomaly_description"].strip()
        return corrupted_csv, anomaly_description
    except (json.JSONDecodeError, KeyError) as exc:
        raise ValueError(f"GPT response parse failed: {exc}\nRaw: {raw[:300]}")


def _csv_to_header_rows(csv_str: str) -> tuple[list[str], list[list[str]]]:
    """Parse CSV string → (header, rows)."""
    import csv as _csv, io
    reader = _csv.reader(io.StringIO(csv_str))
    lines  = list(reader)
    if len(lines) < 2:
        raise ValueError("Corrupted CSV has fewer than 2 lines.")
    return lines[0], lines[1:]


def _validate_structure(
    original: CleanTable,
    corrupted_csv: str,
) -> tuple[list[str], list[list[str]]]:
    """
    Ensure the corrupted table has the same header and column count as the original.
    Raises ValueError if structural integrity is violated.
    """
    header, rows = _csv_to_header_rows(corrupted_csv)

    if [h.strip() for h in header] != [h.strip() for h in original.header]:
        raise ValueError(
            f"Header mismatch.\nOriginal: {original.header}\nCorrupted: {header}"
        )

    for i, row in enumerate(rows):
        if len(row) != original.num_cols:
            raise ValueError(
                f"Column count mismatch at row {i}: "
                f"expected {original.num_cols}, got {len(row)}."
            )

    return header, rows


# ── Public API ────────────────────────────────────────────────────────────────

def perturb_table(
    table: CleanTable,
    category: str,
    seed: int,
) -> PerturbedTable:
    """
    Inject one anomaly of `category` into `table` using OpenAI.
    Returns a PerturbedTable with the corrupted CSV and metadata.
    """
    if category not in ANOMALY_CATEGORIES:
        raise ValueError(f"Unknown category '{category}'. Valid: {ANOMALY_CATEGORIES}")

    original_csv  = table.to_csv()
    user_message  = _build_user_message(table, category)
    raw_response  = _call_openai(user_message)

    corrupted_csv, anomaly_description = _parse_response(raw_response, original_csv)
    corrupted_header, corrupted_rows   = _validate_structure(table, corrupted_csv)

    return PerturbedTable(
        source_table_id=    table.table_id,
        anomaly_category=   category,
        anomaly_description=anomaly_description,
        original_csv=       original_csv,
        corrupted_csv=      corrupted_csv,
        corrupted_header=   corrupted_header,
        corrupted_rows=     corrupted_rows,
        seed=               seed,
        raw_gpt_response=   raw_response,
    )


def perturb_table_all_variants(
    table: CleanTable,
    seed_offset: int,
    categories: Optional[list[str]] = None,
    variants_per_table: int = 1,
) -> list[PerturbedTable]:
    """
    Generate `variants_per_table` perturbed versions of `table`,
    each with a randomly sampled (without replacement) anomaly category.
    """
    cats = categories or ANOMALY_CATEGORIES
    chosen = random.sample(cats, k=min(variants_per_table, len(cats)))

    results: list[PerturbedTable] = []
    for i, cat in enumerate(chosen):
        seed = seed_offset + i
        try:
            pt = perturb_table(table, cat, seed)
            results.append(pt)
            logger.debug("  ✓ [%s] seed=%d  %s", table.table_id, seed, cat)
        except Exception as exc:
            logger.warning("  ✗ [%s] seed=%d  %s — skipped: %s",
                           table.table_id, seed, cat, exc)
    return results
