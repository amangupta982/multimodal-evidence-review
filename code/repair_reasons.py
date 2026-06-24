#!/usr/bin/env python3
"""
repair_reasons.py  —  one-time, no-API cleanup of an existing output.csv.

Fixes rows where evidence_standard_met_reason is junk (empty / "true" / "false")
by writing a meaningful sentence derived deterministically from the matched
evidence rule. Does NOT touch any other column or call any model.

Usage (from code/):
    python repair_reasons.py --output ../output.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
from evidence import EvidenceRules, parse_claim     # noqa: E402

JUNK = {"", "true", "false", "none", "null"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--output", default=str(HERE.parent / "output.csv"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    rules = EvidenceRules(cfg["paths"]["evidence_requirements"])
    df = pd.read_csv(args.output, dtype=str, keep_default_na=False)

    fixed = 0
    for i, row in df.iterrows():
        reason = str(row.get("evidence_standard_met_reason", "")).strip()
        if reason.lower() not in JUNK:
            continue
        claim_object = str(row["claim_object"]).strip().lower()
        parsed = parse_claim(str(row["user_claim"]), claim_object)
        rule = rules.match(claim_object, parsed["issue"], parsed["part"])
        rule_text = rule.get("minimum_image_evidence", "").strip()
        met = str(row.get("evidence_standard_met", "")).strip().lower() == "true"
        if met:
            new = f"The claimed part is visible and assessable: {rule_text}"[:200]
        else:
            new = (f"The claimed part is not clearly assessable in the images; "
                   f"required: {rule_text}")[:200]
        df.at[i, "evidence_standard_met_reason"] = new
        fixed += 1
        print(f"  fixed row {i+1} ({row['user_id']})")

    df.to_csv(args.output, index=False)
    print(f"\nRepaired {fixed} reason cell(s). Saved -> {args.output}")


if __name__ == "__main__":
    main()
