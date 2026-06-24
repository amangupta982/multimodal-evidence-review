#!/usr/bin/env python3
"""
main.py  —  terminal entry point (AGENTS.md §6 contract).

Runs the full two-stage pipeline on an input claims CSV and writes output.csv
with the exact 14-column schema, validated to closed enums.

Usage:
    python main.py                          # uses config test_claims -> ../output.csv
    python main.py --input ../dataset/claims.csv --output ../output.csv
    python main.py --limit 3                # quick smoke run on first 3 rows

Everything is local/offline (Ollama at localhost:11434). No cloud, no secrets.
"""
import argparse
import sys
from pathlib import Path

import yaml
try:
    from dotenv import load_dotenv      # optional: load GEMINI_API_KEY from .env
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pipeline import Pipeline   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", default=None,
                    help="Input claims CSV (default: paths.test_claims from config).")
    ap.add_argument("--output", default="../output.csv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    input_csv = args.input or cfg["paths"]["test_claims"]

    # Auto-detect: if configured test file is absent but a sibling exists, hint.
    if not Path(input_csv).exists():
        print(f"[ERROR] Input not found: {input_csv}")
        print("        Pass --input <path> (e.g. ../dataset/claims.csv or ../dataset/test.csv)")
        sys.exit(1)

    print(f"Running pipeline on {input_csv} ...")
    pipe = Pipeline(cfg, prompts_dir="prompts")
    out, acct = pipe.run(input_csv, args.output, exclude_self=False, limit=args.limit)
    print(f"\nWrote {len(out)} rows -> {args.output}")
    print("Operational tally:", acct)


if __name__ == "__main__":
    main()
