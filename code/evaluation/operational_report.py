#!/usr/bin/env python3
"""
evaluation/operational_report.py  —  PHASE 6 operational analysis.

Consumes the accounting tally produced by a pipeline run (calls, tokens,
images, measured latency) and writes a cost/latency/throughput analysis with
stated assumptions and the math shown. Backend-aware: reports the local
near-zero-cost story for Ollama and the per-token cost story for Gemini.

Usage (called automatically by run_all.py; can also run standalone):
    python evaluation/operational_report.py --tally tally.json --backend gemini
"""
import argparse
import json
from pathlib import Path

# Public Gemini pricing assumptions (USD per 1M tokens). EDIT if your tier
# differs; these are stated assumptions, not guarantees, and may change.
GEMINI_PRICE = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-3.5-flash": {"input": 0.30, "output": 2.50},   # placeholder; verify
}


def build_operational_section(tally: dict, backend: str, split_name: str,
                              model: str = "gemini-2.5-flash") -> str:
    calls = tally.get("vision_calls", 0) + tally.get("fusion_calls", 0) + tally.get("embed_calls", 0)
    pt = tally.get("prompt_tokens", 0)
    ot = tally.get("output_tokens", 0)
    tot = pt + ot
    imgs = tally.get("images_processed_unique", 0)
    hits = tally.get("vision_cache_hits", 0)
    lat = tally.get("total_latency_s", 0.0)
    n_calls = max(calls, 1)
    avg_lat = lat / n_calls

    L = [f"### Operational analysis — {split_name} (backend: {backend})\n"]
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Vision calls | {tally.get('vision_calls', 0)} |")
    L.append(f"| Fusion calls | {tally.get('fusion_calls', 0)} |")
    L.append(f"| Embedding calls | {tally.get('embed_calls', 0)} |")
    L.append(f"| Unique images processed | {imgs} |")
    L.append(f"| Vision cache hits (reused) | {hits} |")
    L.append(f"| Input (prompt) tokens | {pt:,} |")
    L.append(f"| Output tokens | {ot:,} |")
    L.append(f"| Total tokens | {tot:,} |")
    L.append(f"| Measured wall-clock latency | {lat:,.1f} s |")
    L.append(f"| Avg latency / call | {avg_lat:,.1f} s |")

    L.append("\n**Cost story.**")
    if backend == "ollama":
        L.append(
            "\nRunning fully local via Ollama, the **marginal cost per claim is ~$0** "
            "(no per-token billing, no network). The only costs are fixed: the one-time "
            "model download (~3.5 GB) and local electricity/compute. This is fully "
            "reproducible offline — a judge can re-run identical inference with no "
            "credentials. Throughput is bounded by local hardware, not an API quota.")
    else:
        price = GEMINI_PRICE.get(model, GEMINI_PRICE["gemini-2.5-flash"])
        in_cost = pt / 1_000_000 * price["input"]
        out_cost = ot / 1_000_000 * price["output"]
        total_cost = in_cost + out_cost
        per_claim = total_cost / max(tally.get("vision_calls", 0) +
                                     tally.get("fusion_calls", 0), 1)
        L.append(
            f"\nUsing **{model}** at assumed public pricing "
            f"(${price['input']:.2f}/1M input, ${price['output']:.2f}/1M output tokens):\n")
        L.append(f"- Input:  {pt:,} tok / 1e6 × ${price['input']:.2f} = **${in_cost:.4f}**")
        L.append(f"- Output: {ot:,} tok / 1e6 × ${price['output']:.2f} = **${out_cost:.4f}**")
        L.append(f"- **Total for this run: ${total_cost:.4f}**  "
                 f"(~${per_claim:.4f} per claim row)")
        L.append(
            "\nAssumptions: prices as of the cited tier and may change; image tokens "
            "are counted by the API in the input total; figures exclude any free-tier "
            "credits. The local Ollama backend remains available at ~$0 marginal cost "
            "for a fully reproducible, credential-free re-run.")

    L.append("\n**Throughput / rate-limit story.**")
    if backend == "ollama":
        L.append(
            "\nNo TPM/RPM quota applies locally; the limit is the M1's single-stream "
            "inference. We keep within hardware limits by: (1) one image per vision "
            "call (no oversized multi-image prompts), (2) downscaling images to the "
            "configured long-side before encoding, (3) a content-hash **vision cache** "
            "so each unique image is analyzed exactly once across the whole dataset and "
            "across re-runs, and (4) retry/backoff so a transient stall doesn't fail a "
            "long batch.")
    else:
        L.append(
            "\nGemini enforces TPM/RPM limits per tier. We stay under them with: "
            "(1) temperature 0 + JSON-constrained outputs (short, predictable token "
            "counts), (2) the vision cache (each unique image billed/processed once, "
            "re-runs reuse results), (3) exponential **retry/backoff** that absorbs "
            "429/quota responses, and (4) one image per call to keep request sizes "
            "small. To respect RPM on large batches, increase the backoff multiplier "
            "or add a small inter-call sleep in config.")

    # simple projection math to the test set
    L.append("\n**Projection math.**")
    if imgs:
        L.append(
            f"\nThis run processed {imgs} unique images in {lat:,.0f}s "
            f"(~{lat/max(imgs,1):,.1f}s per image including fusion overhead). "
            "The test set (44 rows / 82 unique images) scales roughly linearly: "
            f"≈ {82 * (lat/max(imgs,1)):,.0f}s wall-clock cold, and near-instant on "
            "a warm cache for unchanged images.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tally", required=True, help="JSON file with the accounting dict.")
    ap.add_argument("--backend", default="gemini")
    ap.add_argument("--split", default="sample")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tally = json.loads(Path(args.tally).read_text())
    section = build_operational_section(tally, args.backend, args.split, args.model)
    if args.out:
        Path(args.out).write_text(section)
        print(f"Wrote {args.out}")
    else:
        print(section)


if __name__ == "__main__":
    main()
