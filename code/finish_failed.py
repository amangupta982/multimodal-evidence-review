#!/usr/bin/env python3
"""
finish_failed.py  —  re-run ONLY the rows that fell back (e.g. due to an API
rate limit) and merge the new predictions into an existing output.csv.

A "failed/fallback" row is detected by its signature: claim_status=
not_enough_information AND risk_flags contains 'manual_review_required' AND the
justification says it couldn't be processed automatically.

Usage (from code/):
    # finish the failed rows using whatever backend config.yaml currently says
    python finish_failed.py --output ../output.csv

    # commonly: switch config backend to "ollama" first so this runs locally/free
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from pipeline import Pipeline                       # noqa: E402
from enums import OUTPUT_COLUMNS                    # noqa: E402

FALLBACK_MARK = "could not be processed automatically"


def is_fallback(row) -> bool:
    return (str(row.get("claim_status")) == "not_enough_information"
            and "manual_review_required" in str(row.get("risk_flags"))
            and FALLBACK_MARK in str(row.get("claim_status_justification", "")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--output", default=str(HERE.parent / "output.csv"))
    ap.add_argument("--input", default=None,
                    help="Original input claims CSV (default: config test_claims).")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    input_csv = args.input or cfg["paths"]["test_claims"]
    out_df = pd.read_csv(args.output, dtype=str, keep_default_na=False)
    in_df = pd.read_csv(input_csv)

    failed_idx = [i for i, r in out_df.iterrows() if is_fallback(r)]
    if not failed_idx:
        print("No fallback rows found — output.csv is already complete. Nothing to do.")
        return
    print(f"Found {len(failed_idx)} fallback rows to redo: "
          f"{[out_df.iloc[i]['user_id'] for i in failed_idx]}")
    print(f"Backend: {cfg['ollama'].get('backend','ollama')}")

    pipe = Pipeline(cfg, prompts_dir=str(HERE / "prompts"))
    fixed = 0
    for i in failed_idx:
        # match the input row by position (output preserves input order)
        in_row = in_df.iloc[i]
        try:
            new = pipe.process_row(in_row, exclude_self=False)
            for c in OUTPUT_COLUMNS:
                out_df.at[i, c] = new[c]
            fixed += 1
            print(f"  fixed row {i+1} ({in_row['user_id']}) -> {new['claim_status']}")
            out_df.to_csv(args.output, index=False)   # save after each (crash-safe)
        except Exception as e:                          # noqa: BLE001
            print(f"  row {i+1} still failed: {e}")

    out_df.to_csv(args.output, index=False)
    remaining = sum(is_fallback(r) for _, r in out_df.iterrows())
    print(f"\nFixed {fixed}/{len(failed_idx)}. Remaining fallback rows: {remaining}")
    print(f"Updated -> {args.output}")


if __name__ == "__main__":
    main()