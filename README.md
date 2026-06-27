# Multi-Modal Evidence Review

> A backend-agnostic vision-language pipeline that adjudicates insurance-style
> damage claims (car / laptop / package) from **submitted images + a chat
> transcript + user history**, and emits a validated, schema-strict structured
> verdict for each claim.

Built for the **HackerRank Orchestrate (June 2026)** multi-modal evidence-review
challenge — finished **#248 / 1,773** (top 14%).

---

## The problem

For each claim, the system must read a short support-chat transcript, inspect one
or more submitted images, weigh them against a minimum-evidence rulebook and the
user's claim history, and decide:

- whether the image evidence is **sufficient**
- the visible **issue type** and the relevant **object part**
- whether the claim is **supported / contradicted / not_enough_information**
- which **image IDs** justify that decision
- image-quality, mismatch, authenticity, and user-history **risk flags**
- **severity**, plus short, image-grounded **justifications**

…all constrained to a closed 14-column output schema.

---

## Design philosophy

> **The model does perception and judgment. Deterministic Python does policy.**

A language model is unreliable at exact, rule-bound outputs (closed enums, rule
lookups, provable flags). So the pipeline wraps the model in deterministic guards
that *cannot* emit an illegal value — and makes the task's precedence rules
**structural** rather than something hoped for in a prompt.

---

## Architecture

```
                  ┌──────────────────────────────────────────────┐
   each image ──▶ │ STAGE 1 — VISION  (history-blind)            │
   (dedup +       │ per-image visual facts as JSON:              │
    cached)       │ object, part, issue, assessability, quality, │
                  │ manipulation, embedded-text flags            │
                  └───────────────────────┬──────────────────────┘
                                          │ visual_facts[]
   user_claim ───────────┐   ┌────────────▼─────────────────────────┐
   evidence_rules ───────┼──▶│ DETERMINISTIC POLICY (pure Python)    │
   user_history ─────────┘   │ • parse claim → claimed part / issue  │
                             │ • evidence_standard_met = RULE MATCH  │
                             │ • merge history → risk context        │
                             └────────────┬──────────────────────────┘
                                          │ evidence packet + few-shot examples
                  ┌───────────────────────▼──────────────────────┐
                  │ STAGE 2 — FUSION                              │
                  │ ordered gates: NEI / contradicted / supported│
                  └───────────────────────┬──────────────────────┘
                                          │
                  ┌───────────────────────▼──────────────────────┐
                  │ FINALIZERS (pure Python)                      │
                  │ hybrid risk_flags · enum validate / repair ·  │
                  │ cross-field clamps                            │
                  └───────────────────────┬──────────────────────┘
                                          ▼
                          one validated 14-column output row
```

**Why history-blind vision?** The vision stage never receives user history, so the
rule "images are the source of truth; history only adds risk and never flips a
visually clear decision" is enforced by the data flow itself — not by prompt wording.

---

## Key features

| Feature | What it does |
|---|---|
| **Deterministic evidence check** | `evidence_standard_met` is a lookup against the evidence rulebook, not an LLM guess. |
| **Hybrid risk flags** | The model proposes flags; a deterministic step force-unions provable ones (history, image-quality, manipulation, embedded-text) and normalizes to a closed vocabulary. |
| **Enum validator / repair** | Every categorical field is coerced into its closed enum; out-of-vocabulary values are repaired, never shipped. |
| **Few-shot retrieval** | Labeled examples are used as retrievable in-context exemplars by semantic similarity (never case-ID lookup), with a self-exclusion leakage guard during evaluation. No training. |
| **Vision cache** | Content-hash dedup so each unique image is analyzed exactly once, across the dataset and across re-runs. |
| **Swappable backend** | Groq, Gemini, or fully local Ollama — one line in `config.yaml`. Deterministic core is identical across all three. |
| **Resilient runs** | Per-row error isolation + incremental save: one failed row writes a safe fallback and the run continues, so the output is always complete. |

---

## Output schema (14 columns)

`user_id`, `image_paths`, `user_claim`, `claim_object`, `evidence_standard_met`,
`evidence_standard_met_reason`, `risk_flags`, `issue_type`, `object_part`,
`claim_status`, `claim_status_justification`, `supporting_image_ids`,
`valid_image`, `severity`

---

## Model backends

Selected via `ollama.backend` in `config.yaml`:

| Backend  | Models | Notes |
|----------|--------|-------|
| `groq`   | `llama-4-scout-17b` (vision + fusion) + local embeddings | Free tier, fast |
| `gemini` | `gemini-2.5-flash` family via `google-genai` | Cloud |
| `ollama` | `qwen2.5vl:3b` + `nomic-embed-text` | Fully local, offline, reproducible |

API keys are read from a `.env` file (`GROQ_API_KEY` / `GEMINI_API_KEY`) — never
hardcoded. The local Ollama backend needs no key and runs entirely offline, which
makes it the reproducible fallback any reviewer can run without credentials.

---

## Quick start

```bash
# from code/
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# add a key for a cloud backend (optional — Ollama needs none)
cp .env.example .env          # then edit GROQ_API_KEY or GEMINI_API_KEY

# few-shot embeddings run locally even on cloud backends:
ollama pull nomic-embed-text
ollama serve &

# run everything: evaluate on sample + produce output.csv + reports
python run_all.py
```

Run stages individually:

```bash
python smoke_test.py          # prove the backend works end to end
python evaluation/main.py     # per-column metrics + confusion matrix → report
python main.py                # full run on the test claims → ../output.csv
```

---

## Repository layout

```
code/
├── main.py                  # entry point: produce output.csv
├── run_all.py               # one command: eval + output + analysis + checklist
├── config.yaml              # models, paths, limits — single source of truth
├── prompts/                 # version-controlled vision + fusion prompts
│   ├── vision_prompt.txt
│   ├── fusion_prompt.txt
│   └── risk_flags_contract.md
├── src/
│   ├── backends.py          # swappable Groq / Gemini / Ollama backends
│   ├── pipeline.py          # two-stage orchestration
│   ├── evidence.py          # claim parser + deterministic evidence-rule matcher
│   ├── risk_flags.py        # hybrid risk-flag finalizer
│   ├── validate.py          # closed-enum validator / repair
│   ├── retrieval.py         # local few-shot retrieval (with leakage guard)
│   ├── vision_cache.py      # content-hash dedup (each image analyzed once)
│   ├── enums.py             # closed vocabulary (single source of truth)
│   └── ollama_client.py     # local client: retry, backoff, token accounting
└── evaluation/
    ├── main.py              # evaluation harness
    └── operational_report.py
```

---

## Evaluation

The harness scores predictions against the labeled sample set with the leakage
guard on, and reports:

- overall row accuracy and **per-column** accuracy + macro-F1
- a `claim_status` **confusion matrix**
- set-level precision / recall / F1 for the multi-label `risk_flags`
- a worst-miss table for error analysis
- an **operational report**: model calls, token totals, measured latency, and a
  cost / throughput analysis with stated assumptions

It prioritizes the core decision columns (`claim_status`, `object_part`,
`valid_image`, `supporting_image_ids`), which are the most reliable; `severity`
and fine-grained `issue_type` are inherently subjective and noisier.

---

## Design decisions & limitations

- **No fine-tuning.** Labeled data is used only as retrievable few-shot exemplars,
  selected by semantic similarity — never by case ID. No test-label hardcoding.
- **Deterministic where it counts.** Evidence rule, enum validity, and provable
  risk flags are computed in Python so they can't drift with the model.
- The local 3B backend has weaker perception than the cloud backends; the
  deterministic guards keep outputs schema-valid regardless of backend.
- On a free cloud tier a long run can hit a daily token cap; the pipeline saves
  incrementally and writes safe fallback rows so `output.csv` is always complete.

---

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
