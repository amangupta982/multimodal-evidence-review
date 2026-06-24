#!/usr/bin/env python3
"""
evaluation/main.py  —  PHASE 5 evaluation harness.

Runs the full pipeline on dataset/sample_claims.csv (leakage guard ON so a case
never retrieves itself), then scores predictions vs the gold labels:

  - overall row accuracy (all graded columns correct)
  - PER-COLUMN accuracy, plus macro-F1 for categorical columns
  - confusion matrix for claim_status
  - risk_flags set-level precision/recall/F1 (semicolon multi-label)
  - error-analysis table of the worst misses (with image paths to inspect)
  - writes evaluation/evaluation_report.md and evaluation/predictions_sample.csv

Usage (from code/):
    python evaluation/main.py
    python evaluation/main.py --limit 5      # quick subset
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml
try:
    from dotenv import load_dotenv      # optional: load GEMINI_API_KEY from .env
    load_dotenv()
except ImportError:
    pass

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE / "src"))

from pipeline import Pipeline                       # noqa: E402
from retrieval import case_id_of                    # noqa: E402

# Columns we grade (the derived outputs; inputs are echoed verbatim).
GRADED = ["evidence_standard_met", "risk_flags", "issue_type", "object_part",
          "claim_status", "supporting_image_ids", "valid_image", "severity"]
# Single-label categoricals get accuracy + macro-F1.
CATEGORICAL = ["issue_type", "object_part", "claim_status", "severity",
               "evidence_standard_met", "valid_image"]


def _norm(v):
    return str(v).strip().lower()


def _flagset(v):
    return {t.strip().lower() for t in str(v).split(";") if t.strip() and t.strip().lower() != "none"}


def macro_f1(y_true, y_pred):
    labels = set(y_true) | set(y_pred)
    f1s = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def confusion(y_true, y_pred, labels):
    m = {a: {b: 0 for b in labels} for a in labels}
    for t, p in zip(y_true, y_pred):
        if t in m and p in m[t]:
            m[t][p] += 1
    return m


def multilabel_prf(gold_series, pred_series):
    tp = fp = fn = 0
    for g, p in zip(gold_series, pred_series):
        gs, ps = _flagset(g), _flagset(p)
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CODE / "config.yaml"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    sample_csv = cfg["paths"]["sample_claims"]
    gold = pd.read_csv(sample_csv)
    if args.limit:
        gold = gold.head(args.limit)

    print(f"Evaluating on {len(gold)} sample rows (leakage guard ON)...")
    pipe = Pipeline(cfg, prompts_dir=str(CODE / "prompts"))

    preds = []
    for _, r in gold.iterrows():
        preds.append(pipe.process_row(r, exclude_self=True))
    pred = pd.DataFrame(preds)

    pred_out = HERE / "predictions_sample.csv"
    pred.to_csv(pred_out, index=False)

    # ---- per-column metrics ----
    col_acc, col_f1 = {}, {}
    for c in GRADED:
        if c == "risk_flags":
            continue
        yt = [_norm(x) for x in gold[c]]
        yp = [_norm(x) for x in pred[c]]
        col_acc[c] = sum(t == p for t, p in zip(yt, yp)) / len(yt)
        if c in CATEGORICAL:
            col_f1[c] = macro_f1(yt, yp)
    # supporting_image_ids: set-level match
    sii_acc = sum(_flagset(g) == _flagset(p)
                  for g, p in zip(gold["supporting_image_ids"], pred["supporting_image_ids"])) / len(gold)
    col_acc["supporting_image_ids"] = sii_acc
    rf_p, rf_r, rf_f1 = multilabel_prf(gold["risk_flags"], pred["risk_flags"])

    # overall row accuracy (all graded cols correct, multilabel set-equal)
    def row_ok(i):
        for c in GRADED:
            g, p = gold.iloc[i][c], pred.iloc[i][c]
            if c in ("risk_flags", "supporting_image_ids"):
                if _flagset(g) != _flagset(p):
                    return False
            elif _norm(g) != _norm(p):
                return False
        return True
    overall = sum(row_ok(i) for i in range(len(gold))) / len(gold)

    # claim_status confusion
    labels = ["supported", "contradicted", "not_enough_information"]
    cm = confusion([_norm(x) for x in gold["claim_status"]],
                   [_norm(x) for x in pred["claim_status"]], labels)

    # ---- worst misses ----
    misses = []
    for i in range(len(gold)):
        wrong = []
        for c in GRADED:
            g, p = gold.iloc[i][c], pred.iloc[i][c]
            ok = (_flagset(g) == _flagset(p)) if c in ("risk_flags", "supporting_image_ids") else (_norm(g) == _norm(p))
            if not ok:
                wrong.append(c)
        if wrong:
            misses.append((len(wrong), case_id_of(gold.iloc[i]["image_paths"]),
                           gold.iloc[i]["claim_object"], wrong, gold.iloc[i]["image_paths"]))
    misses.sort(reverse=True)

    _write_report(HERE / "evaluation_report.md", len(gold), overall, col_acc,
                  col_f1, (rf_p, rf_r, rf_f1), cm, labels, misses,
                  pipe.acct.as_dict(), gold, pred)

    # ---- console summary ----
    print(f"\nOverall row accuracy: {overall:.1%}")
    print("Per-column accuracy:")
    for c in GRADED:
        extra = f"  macroF1={col_f1[c]:.2f}" if c in col_f1 else ""
        if c == "risk_flags":
            print(f"  {c:24} P={rf_p:.2f} R={rf_r:.2f} F1={rf_f1:.2f}")
        else:
            print(f"  {c:24} acc={col_acc[c]:.1%}{extra}")
    print(f"\nReport -> {HERE/'evaluation_report.md'}")
    print(f"Predictions -> {pred_out}")


def _write_report(path, n, overall, col_acc, col_f1, rf, cm, labels, misses, acct, gold, pred):
    rf_p, rf_r, rf_f1 = rf
    L = []
    L.append("# Evaluation Report — Multi-Modal Evidence Review\n")
    L.append(f"Evaluated on **{n}** labeled sample rows. Leakage guard ON "
             f"(no case retrieves itself).\n")
    L.append(f"\n## Overall\n\n- **Row accuracy (all graded columns correct): {overall:.1%}**\n")
    L.append("\n## Per-column metrics\n\n| Column | Accuracy | Macro-F1 |\n|---|---|---|")
    for c in ["evidence_standard_met", "issue_type", "object_part",
              "claim_status", "severity", "valid_image", "supporting_image_ids"]:
        f1 = f"{col_f1[c]:.2f}" if c in col_f1 else "—"
        L.append(f"| {c} | {col_acc[c]:.1%} | {f1} |")
    L.append(f"\n**risk_flags** (multi-label, set-level): "
             f"Precision {rf_p:.2f} · Recall {rf_r:.2f} · F1 {rf_f1:.2f}\n")
    L.append("\n## claim_status confusion matrix\n\n(rows = gold, cols = predicted)\n")
    L.append("| gold \\ pred | " + " | ".join(labels) + " |")
    L.append("|---|" + "---|" * len(labels))
    for a in labels:
        L.append(f"| {a} | " + " | ".join(str(cm[a][b]) for b in labels) + " |")
    L.append(f"\n## Worst misses ({len(misses)} rows with >=1 error)\n")
    L.append("| case | object | #wrong | wrong columns | images |")
    L.append("|---|---|---|---|---|")
    for nwrong, cid, obj, cols, paths in misses[:15]:
        L.append(f"| {cid} | {obj} | {nwrong} | {', '.join(cols)} | `{paths}` |")
    L.append("\n### Side-by-side for the worst cases\n")
    miss_ids = {m[1] for m in misses[:6]}
    for i in range(len(gold)):
        cid = case_id_of(gold.iloc[i]["image_paths"])
        if cid not in miss_ids:
            continue
        L.append(f"\n**{cid}** — claim: _{str(gold.iloc[i]['user_claim'])[:140]}_\n")
        L.append("| col | gold | pred |")
        L.append("|---|---|---|")
        for c in ["evidence_standard_met", "issue_type", "object_part",
                  "claim_status", "severity", "valid_image", "risk_flags",
                  "supporting_image_ids"]:
            L.append(f"| {c} | {gold.iloc[i][c]} | {pred.iloc[i][c]} |")
    L.append("\n## Operational analysis\n")
    try:
        import yaml as _yaml
        _cfg = _yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config.yaml"))
        _backend = _cfg["ollama"].get("backend", "ollama")
        _model = _cfg.get("gemini", {}).get("vision_model", "gemini-2.5-flash")
    except Exception:
        _backend, _model = "ollama", "gemini-2.5-flash"
    from operational_report import build_operational_section
    L.append(build_operational_section(acct, _backend, "sample", _model))
    path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
