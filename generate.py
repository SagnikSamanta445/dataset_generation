"""
generate.py — Main entry point for the DB Normalization data generation pipeline.

Usage:
    # Basic run (reads OPENAI_API_KEY from env)
    python generate.py

    # Override table count and output dir
    python generate.py --max-tables 200 --output-dir ./my_output

    # Push to HuggingFace Hub after generation
    python generate.py --push-to-hub --hf-repo-id your-username/db-norm-rl

    # Dry-run (loads sources + validates, no OpenAI calls)
    python generate.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import random
from pathlib import Path

# ── Logging setup (before any local imports) ──────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Local imports ─────────────────────────────────────────────────────────────
import config as cfg
from sources      import load_clean_tables, CleanTable
from perturbations import perturb_table_all_variants, PerturbedTable
from builder      import build_dataset, train_test_split_dataset, save_dataset, push_to_hub


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DB Normalization RL data generator")
    p.add_argument("--max-tables",    type=int,  default=cfg.MAX_SOURCE_TABLES)
    p.add_argument("--variants",      type=int,  default=cfg.VARIANTS_PER_TABLE,
                   help="Corrupted variants per clean table")
    p.add_argument("--seed-base",     type=int,  default=cfg.SEED_BASE)
    p.add_argument("--output-dir",    type=str,  default=cfg.OUTPUT_DIR)
    p.add_argument("--hf-repo-id",    type=str,  default=cfg.HF_REPO_ID)
    p.add_argument("--push-to-hub",   action="store_true", default=cfg.PUSH_TO_HUB)
    p.add_argument("--dry-run",       action="store_true",
                   help="Load sources only; skip OpenAI calls")
    p.add_argument("--categories",    nargs="+", default=cfg.ANOMALY_CATEGORIES,
                   help="Subset of anomaly categories to use")
    return p.parse_args()


# ── Dry-run mode ──────────────────────────────────────────────────────────────

def dry_run(tables: list[CleanTable], args: argparse.Namespace) -> None:
    logger.info("=== DRY RUN — no OpenAI calls ===")
    for t in tables[:5]:
        logger.info(
            "  table_id=%-20s  rows=%-3d  cols=%-2d  question=%s",
            t.table_id, t.num_rows, t.num_cols, t.question[:60],
        )
    logger.info("Total usable tables: %d", len(tables))
    estimated = len(tables) * args.variants
    logger.info("Estimated dataset size: ~%d rows", estimated)
    logger.info("Sample CSV (first table):\n%s", tables[0].to_csv()[:400])


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    # Override config from CLI args
    cfg.MAX_SOURCE_TABLES  = args.max_tables
    cfg.VARIANTS_PER_TABLE = args.variants
    cfg.SEED_BASE          = args.seed_base
    cfg.OUTPUT_DIR         = args.output_dir
    cfg.HF_REPO_ID         = args.hf_repo_id
    cfg.PUSH_TO_HUB        = args.push_to_hub

    # Validate API key
    api_key = os.environ.get("OPENAI_API_KEY", cfg.OPENAI_API_KEY)
    if not args.dry_run and api_key.startswith("sk-YOUR"):
        logger.error(
            "Set your OpenAI API key via:\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "or edit OPENAI_API_KEY in config.py"
        )
        sys.exit(1)
    cfg.OPENAI_API_KEY = api_key

    # ── Step 1: Load clean tables ─────────────────────────────────────────────
    logger.info("STEP 1 — Loading clean tables from WikiTQ …")
    tables = load_clean_tables()

    if not tables:
        logger.error("No usable tables found. Aborting.")
        sys.exit(1)

    if args.dry_run:
        dry_run(tables, args)
        return

    # ── Step 2: Perturb tables ────────────────────────────────────────────────
    logger.info("STEP 2 — Injecting anomalies via OpenAI (%s) …", cfg.OPENAI_MODEL)
    logger.info("  Tables: %d | Variants per table: %d | Categories: %s",
                len(tables), args.variants, args.categories)

    all_perturbed: list[PerturbedTable] = []
    random.seed(args.seed_base)

    for i, table in enumerate(tables):
        seed_offset = args.seed_base + i * args.variants
        logger.info("[%d/%d] Perturbing %s …", i + 1, len(tables), table.table_id)

        variants = perturb_table_all_variants(
            table=table,
            seed_offset=seed_offset,
            categories=args.categories,
            variants_per_table=args.variants,
        )
        all_perturbed.extend(variants)

    logger.info("Total perturbed records generated: %d", len(all_perturbed))

    if not all_perturbed:
        logger.error("No records generated (all perturbations failed?). Aborting.")
        sys.exit(1)

    # ── Step 3: Build HuggingFace Dataset ────────────────────────────────────
    logger.info("STEP 3 — Building HuggingFace Dataset …")
    ds = build_dataset(all_perturbed)
    ds_dict = train_test_split_dataset(ds, test_size=0.1, seed=args.seed_base)

    logger.info(
        "Dataset splits → train: %d  |  test: %d",
        len(ds_dict["train"]), len(ds_dict["test"]),
    )

    # ── Step 4: Save to disk ──────────────────────────────────────────────────
    logger.info("STEP 4 — Saving dataset to disk …")
    out_path = save_dataset(ds_dict, output_dir=args.output_dir)
    logger.info("Dataset saved to: %s", out_path)

    # ── Step 5: Push to Hub (optional) ───────────────────────────────────────
    if args.push_to_hub:
        logger.info("STEP 5 — Pushing to HuggingFace Hub …")
        push_to_hub(ds_dict)

    logger.info("=" * 60)
    logger.info("Done! Dataset ready for OpenEnv / GRPOTrainer.")
    logger.info("  Rows (train):  %d", len(ds_dict["train"]))
    logger.info("  Rows (test):   %d", len(ds_dict["test"]))
    logger.info("  Output dir:    %s", out_path)
    logger.info("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(parse_args())
