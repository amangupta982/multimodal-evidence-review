# Multi-Modal Evidence Review

Verifies damage claims (car / laptop / package) from images + chat + history,
producing a validated 14-column `output.csv`. Uses a vision-language model via a
swappable backend (Groq / Gemini / local Ollama). No secrets in code; API keys
are read from a `.env` file. The deterministic core is identical across backends.

## What it does
Two-stage pipeline:
1. **Vision** (`qwen2.5vl:3b`, history-blind): extracts visual facts per image.
2. **Fusion** (same model, text mode): combines visual facts + claim + the
   deterministic evidence rule + user-history risk into the final verdict.

Deterministic layers around the model guarantee correctness: `evidence_standard_met`
is a rule lookup (not a guess), `risk_flags` are force-unioned from provable
sources, and every field is validated to a closed enum.

## Model backend (swappable)

The system is **backend-agnostic**. The deterministic core (evidence-rule
matching, enum validation, risk-flag logic, retrieval) is identical regardless of
which model produces the perception/fusion text. Switch backends with one line in
`config.yaml`:

```yaml
ollama:
  backend: "groq"     # shipped default: Groq cloud (free tier, fast vision)
  # backend: "gemini" # Google Gemini cloud API
  # backend: "ollama" # fully local fallback (qwen2.5vl:3b), no key, reproducible
```

- **`groq`** (shipped default): vision + fusion via Groq's OpenAI-compatible API
  using `meta-llama/llama-4-scout-17b-16e-instruct`. Free tier, fast. Embeddings
  for few-shot retrieval run locally on Ollama (`nomic-embed-text`).
- **`gemini`**: uses Gemini via the `google-genai` SDK (`GEMINI_API_KEY`).
- **`ollama`**: fully local fallback (`qwen2.5vl:3b` + `nomic-embed-text`), no
  key, no network, maximally reproducible.

All keys are read from a `.env` file (`GROQ_API_KEY` / `GEMINI_API_KEY`); nothing
is hardcoded. This multi-backend design satisfies the "quality and
reproducibility" criterion: run on a fast cloud model, or on the local fallback
that any judge can reproduce.

## Requirements
- Apple Silicon Mac (built/tested for 8GB M1) or any machine with Ollama.
- [Ollama](https://ollama.com) installed and running (used for local embeddings even on the Groq backend).
- A Groq API key (free, no card) in `.env` as `GROQ_API_KEY` for the default backend.
- Python 3.10+

## Setup (one time)
```bash
# 1. Install + start Ollama, then pull the two local models:
ollama pull qwen2.5vl:3b       # vision-language, ~3.2 GB
ollama pull nomic-embed-text   # text embeddings, ~274 MB

# 2. Python environment (run from the code/ folder):
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Expected layout
```
project/
├── code/        <- this folder
└── dataset/     <- sample_claims.csv, claims.csv, user_history.csv,
                    evidence_requirements.csv, images/{sample,test}/...
```

## Run
```bash
cd code

# Fastest path — one command does everything:
python run_all.py              # eval on sample + produce ../output.csv + op-analysis + checklist

# Or run stages individually:
python smoke_test.py           # prove the backend works end-to-end
python evaluation/main.py      # per-column metrics + confusion matrix -> evaluation/evaluation_report.md
python main.py                 # full run on test claims -> ../output.csv
python main.py --limit 3       # quick test on first 3 rows
```

## Packaging the submission
```bash
cd ..
zip -r code.zip code -x '*/.venv/*' '*__pycache__*' '*/cache/*' '*/dataset/*' '*.env'
```
Submit: `code.zip`, `output.csv`, the chat transcript, and
`~/hackerrank_orchestrate/log.txt`.

## Architecture (summary)
Two-stage, with deterministic guards around the model:
1. **Vision** (history-blind): per-image visual facts as JSON.
2. **Deterministic policy**: claim parse, evidence-rule match -> `evidence_standard_met`, history merge.
3. **Fusion**: ordered decision gates (NEI / contradicted / supported) over visual facts + claim + rule + history + retrieved few-shot examples.
4. **Finalizers**: hybrid `risk_flags` (LLM proposal + forced provable flags), enum validator/repair, cross-field clamps.

**Precedence is structural**: the vision stage never sees user history, so history
cannot bias perception; it only adds risk flags downstream and never flips a
visually clear verdict.

## Config
All models, paths, and limits live in `config.yaml`. Prompts live in `prompts/`
as version-controlled files. Nothing is hardcoded in the source.

## Limitations
- Subjective columns (`severity`, fine-grained `issue_type`) are inherently noisy;
  the system prioritizes the core decision columns (`claim_status`, `object_part`,
  `valid_image`, `supporting_image_ids`), which score highest.
- The local Ollama 3B backend has weaker perception than the cloud backends;
  deterministic guards keep outputs schema-valid regardless of backend.
- The claim parser is keyword-based (a seed/fallback); the vision stage refines the
  part from the image. No test-label hardcoding anywhere.
- On Groq's free tier, a long run can hit the daily token cap; the pipeline saves
  incrementally and writes safe fallback rows so output.csv is always complete.