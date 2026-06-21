#!/usr/bin/env python3
"""
retrieval.py  —  PHASE 3: local few-shot retrieval.

Embeds the labeled sample_claims.csv cases with a LOCAL embedding model
(nomic-embed-text via Ollama) and, for each new claim, returns the k most
similar labeled cases to use as in-context examples in the fusion prompt.

- Fully offline (hits http://localhost:11434).
- Embeddings cached to disk; computed once.
- Leakage guard: exclude_case_id prevents a sample case retrieving itself
  during evaluation on the sample set.

No model training. Labeled data is used only as retrievable exemplars.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _clean_claim_text(user_claim: str) -> str:
    """Strip the 'Customer:/Support:' scaffolding to a compact gist for embedding."""
    parts = []
    for seg in str(user_claim).split("|"):
        seg = seg.strip()
        for tag in ("Customer:", "Support:"):
            if seg.startswith(tag):
                seg = seg[len(tag):].strip()
        if seg:
            parts.append(seg)
    return " ".join(parts)


def case_id_of(image_paths: str) -> str:
    """images/sample/case_001/img_1.jpg;... -> 'case_001'."""
    first = str(image_paths).split(";")[0]
    for token in first.split("/"):
        if token.startswith("case_"):
            return token
    return first


def _signature(row) -> str:
    """Text we embed for a case: object + claim gist (+ labels if available)."""
    sig = f"object: {row['claim_object']}. claim: {_clean_claim_text(row['user_claim'])}"
    # sample_claims has labels; include them so retrieval clusters by outcome too.
    for col in ("issue_type", "object_part", "claim_status"):
        if col in row and pd.notna(row.get(col)):
            sig += f". {col}: {row[col]}"
    return sig


class FewShotRetriever:
    def __init__(self, cfg, backend=None):
        self.cfg = cfg
        self.model = cfg["ollama"]["embed_model"]
        self.cache_path = Path(cfg["paths"].get("embed_cache", "cache/embed_cache.json"))
        self.examples = []      # list of dicts: {case_id, signature, row(dict)}
        self.matrix = None      # np.ndarray [N, D], L2-normalized
        # Embeddings can use a different backend than vision/fusion. Default:
        # force local Ollama for embeddings (free, unlimited) so Gemini's daily
        # request quota is spent only on vision+fusion. Set retrieval.embed_backend
        # to "same" to use the main backend instead.
        embed_choice = cfg.get("retrieval", {}).get("embed_backend", "ollama")
        if embed_choice == "same" or backend is None:
            self.backend = backend
        else:
            from backends import OllamaBackend
            from ollama_client import Accounting
            self.backend = OllamaBackend(cfg, getattr(backend, "acct", Accounting()))

    # ---- embedding (delegates to the chosen embed backend) ----
    def _embed(self, text: str) -> np.ndarray:
        vec = self.backend.embed(text)
        v = np.asarray(vec, dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _load_cache(self):
        if self.cache_path.exists():
            return json.loads(self.cache_path.read_text())
        return {}

    def _save_cache(self, cache):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(cache))

    # ---- build index from labeled sample set ----
    def fit(self, sample_csv: str):
        df = pd.read_csv(sample_csv)
        cache = self._load_cache()
        vecs = []
        dirty = False
        for _, row in df.iterrows():
            cid = case_id_of(row["image_paths"])
            sig = _signature(row)
            key = f"{getattr(self.backend, 'name', 'na')}::{self.model}::{sig}"
            if key in cache:
                vec = np.asarray(cache[key], dtype=np.float32)
            else:
                vec = self._embed(sig)
                cache[key] = vec.tolist()
                dirty = True
            self.examples.append({"case_id": cid, "signature": sig,
                                  "row": row.to_dict()})
            vecs.append(vec)
        if dirty:
            self._save_cache(cache)
        self.matrix = np.vstack(vecs)
        return self

    # ---- query ----
    def retrieve(self, query_object: str, query_claim: str,
                 k: int = 3, exclude_case_id: str | None = None):
        qsig = f"object: {query_object}. claim: {_clean_claim_text(query_claim)}"
        qvec = self._embed(qsig)
        sims = self.matrix @ qvec                      # cosine (all normalized)
        order = np.argsort(-sims)
        out = []
        for idx in order:
            ex = self.examples[idx]
            if exclude_case_id and ex["case_id"] == exclude_case_id:
                continue                                # leakage guard
            out.append((float(sims[idx]), ex))
            if len(out) >= k:
                break
        return out

    @staticmethod
    def format_for_prompt(retrieved) -> str:
        """Render retrieved cases compactly for the fusion prompt slot."""
        if not retrieved:
            return "(none)"
        lines = []
        for sim, ex in retrieved:
            r = ex["row"]
            lines.append(
                f"- [{ex['case_id']}, sim={sim:.2f}] object={r.get('claim_object')}; "
                f"claim='{_clean_claim_text(r.get('user_claim',''))[:120]}'; "
                f"-> status={r.get('claim_status')}, issue={r.get('issue_type')}, "
                f"part={r.get('object_part')}, severity={r.get('severity')}, "
                f"risk_flags={r.get('risk_flags')}"
            )
        return "\n".join(lines)