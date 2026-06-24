#!/usr/bin/env python3
"""
run_all.py  —  PHASE 7 one-command runner.

Does the full submission flow:
  1. Evaluate on sample_claims.csv (per-column metrics + report)   [evaluation/]
  2. Run the pipeline on the test claims -> ../output.csv          [deliverable]
  3. Append a test-set operational analysis to the eval report     [Phase 6]
  4. Print the submission checklist mapping requirements -> files

Usage (from code/):
    python run_all.py                       # full run
    python run_all.py --skip-eval           # only produce output.csv
    python run_all.py --limit 3             # quick smoke of both stages

Backend (Gemini primary / Ollama fallback) is chosen in config.yaml.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "evaluation"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pipeline import Pipeline                       # noqa: E402
from operational_report import build_operational_section  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=str(HERE.parent / "output.csv"))
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    backend = cfg["ollama"].get("backend", "ollama")
    model = cfg.get("gemini", {}).get("vision_model", "gemini-2.5-flash")
    input_csv = args.input or cfg["paths"]["test_claims"]

    print(f"Backend: {backend}")

    # 1. Evaluation on the labeled sample set
    if not args.skip_eval:
        print("\n=== [1/3] Evaluating on sample_claims.csv ===")
        cmd = [sys.executable, str(HERE / "evaluation" / "main.py")]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        subprocess.run(cmd, check=True)

    # 2. Final run -> output.csv
    print(f"\n=== [2/3] Running pipeline on {input_csv} -> {args.output} ===")
    if not Path(input_csv).exists():
        print(f"[ERROR] Input not found: {input_csv}. Pass --input <path>.")
        sys.exit(1)
    pipe = Pipeline(cfg, prompts_dir=str(HERE / "prompts"))
    out, acct = pipe.run(input_csv, args.output, exclude_self=False, limit=args.limit)
    print(f"Wrote {len(out)} rows -> {args.output}")

    # 3. Append test-set operational analysis to the eval report
    print("\n=== [3/3] Operational analysis (test run) ===")
    report = HERE / "evaluation" / "evaluation_report.md"
    section = build_operational_section(acct, backend, "test (claims.csv)", model)
    with open(report, "a") as f:
        f.write("\n\n" + section + "\n")
    print(f"Appended test-set operational analysis to {report}")
    (HERE / "evaluation" / "test_tally.json").write_text(json.dumps(acct, indent=2))

    _print_checklist(args.output, report)


def _print_checklist(output_csv, report):
    print("\n" + "=" * 60)
    print("SUBMISSION CHECKLIST")
    print("=" * 60)
    items = [
        ("code.zip (all source)", "code/  — src/, prompts/, config.yaml, main.py"),
        ("  - both stage prompts (version-controlled)", "code/prompts/{vision_prompt,fusion_prompt}.txt"),
        ("  - configs (no magic constants)", "code/config.yaml"),
        ("  - README (arch, run, decisions, limits)", "code/README.md"),
        ("  - entry points (AGENTS.md §6)", "code/main.py, code/evaluation/main.py"),
        ("output.csv (validated, 14-col, exact order)", str(output_csv)),
        ("evaluation/ folder", "code/evaluation/  (main.py, report, predictions)"),
        ("  - per-column metrics + confusion matrix", "code/evaluation/evaluation_report.md"),
        ("  - operational analysis (cost/latency)", "code/evaluation/evaluation_report.md (Phase 6)"),
        ("chat_transcript", "export this conversation"),
        ("log.txt (AGENTS.md §2)", "~/hackerrank_orchestrate/log.txt (via session_logger.py)"),
    ]
    for req, where in items:
        print(f"[x] {req}")
        print(f"      -> {where}")
    print("\nReminder: zip the code/ folder EXCLUDING .venv, __pycache__, cache/, and dataset/.")
    print("  cd .. && zip -r code.zip code -x '*/.venv/*' '*__pycache__*' '*/cache/*'")


if __name__ == "__main__":
    main()
