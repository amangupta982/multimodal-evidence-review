#!/usr/bin/env python3
"""
pipeline.py  —  PHASE 4 orchestration. Vision -> deterministic policy ->
fusion -> hybrid risk_flags -> enum validation, producing one output row per
input row.

Precedence is enforced structurally: the VISION stage never receives user
history, so history cannot bias what is "seen". History enters only in the
deterministic finalizers.
"""
import json
from pathlib import Path

import pandas as pd

from enums import OUTPUT_COLUMNS
from evidence import EvidenceRules, parse_claim
from ollama_client import Accounting, safe_format, extract_json
from backends import make_backend
from retrieval import FewShotRetriever, case_id_of
from risk_flags import finalize as finalize_risk
from validate import validate_row
from vision_cache import VisionCache


def _image_ids(image_paths: str):
    ids = []
    for p in str(image_paths).split(";"):
        p = p.strip()
        if p:
            ids.append(Path(p).stem)        # img_1.jpg -> img_1
    return ids


def _resolve(image_path: str, dataset_root: str) -> str:
    # CSV paths look like 'images/sample/case_x/img_1.jpg' relative to dataset_root
    return str(Path(dataset_root) / image_path.strip())


class Pipeline:
    def __init__(self, cfg, prompts_dir="prompts"):
        self.cfg = cfg
        self.acct = Accounting()
        self.client = make_backend(cfg, self.acct)
        self.vision_prompt = Path(prompts_dir, "vision_prompt.txt").read_text()
        self.fusion_prompt = Path(prompts_dir, "fusion_prompt.txt").read_text()
        self.rules = EvidenceRules(cfg["paths"]["evidence_requirements"])
        self.history = pd.read_csv(cfg["paths"]["user_history"]).set_index("user_id")
        self.vcache = VisionCache(cfg["paths"]["vision_cache"])
        self.k = cfg["retrieval"]["k"]
        self.dataset_root = cfg["paths"]["dataset_root"]
        # Optional inter-call delay to stay under API RPM limits (e.g. Gemini
        # free tier). 0 = no delay (local Ollama). Set in config: ollama.call_delay_s
        self.call_delay_s = cfg["ollama"].get("call_delay_s", 0)
        self.retriever = FewShotRetriever(cfg, backend=self.client).fit(cfg["paths"]["sample_claims"])

    # ---------- Stage A ----------
    def run_vision(self, image_path, image_id, claim_object, claim_hint):
        key = VisionCache.key_for(image_path, claim_object)
        cached = self.vcache.get(key)
        if cached is not None:
            self.acct.cache_hits += 1
            fact = dict(cached)
        else:
            prompt = safe_format(self.vision_prompt,
                                 claim_object=claim_object, claim_hint=claim_hint)
            raw = self.client.vision(prompt, image_path)
            if self.call_delay_s:
                import time as _t; _t.sleep(self.call_delay_s)
            fact = extract_json(raw)
            self.vcache.put(key, fact)
        fact["image_id"] = image_id
        return fact

    # ---------- history ----------
    def _history_for(self, user_id):
        if user_id in self.history.index:
            row = self.history.loc[user_id]
            return (str(row.get("history_summary", "")),
                    str(row.get("history_flags", "none")))
        return ("No history on file.", "none")

    # ---------- one row ----------
    def process_row(self, row, exclude_self=False):
        claim_object = str(row["claim_object"]).strip().lower()
        user_claim = str(row["user_claim"])
        image_paths = str(row["image_paths"])
        ids = _image_ids(image_paths)

        parsed = parse_claim(user_claim, claim_object)
        claim_hint = f"{parsed['issue']} on {parsed['part']}"

        # Stage A: vision per image (history-blind)
        vision_facts = []
        for raw_path, img_id in zip(image_paths.split(";"), ids):
            ipath = _resolve(raw_path, self.dataset_root)
            vision_facts.append(
                self.run_vision(ipath, img_id, claim_object, claim_hint))

        # Deterministic: evidence rule + evidence_standard_met
        rule = self.rules.match(claim_object, parsed["issue"], parsed["part"])
        part_assessable = any(vf.get("part_assessable") is True for vf in vision_facts)
        evidence_met = bool(part_assessable)

        # history
        hist_summary, hist_flags = self._history_for(row["user_id"])

        # Few-shot retrieval (exclude self when evaluating on sample set)
        excl = case_id_of(image_paths) if exclude_self else None
        retrieved = self.retriever.retrieve(claim_object, user_claim,
                                            k=self.k, exclude_case_id=excl)
        examples_str = FewShotRetriever.format_for_prompt(retrieved)

        # Stage B: fusion
        fusion_in = safe_format(
            self.fusion_prompt,
            claim_object=claim_object,
            user_claim=user_claim,
            minimum_image_evidence=rule.get("minimum_image_evidence", ""),
            evidence_standard_met=str(evidence_met).lower(),
            vision_facts_list=json.dumps(vision_facts, ensure_ascii=False),
            history_summary=hist_summary,
            history_flags=hist_flags,
            retrieved_examples=examples_str,
        )
        fused = extract_json(self.client.text(fusion_in))
        if self.call_delay_s:
            import time as _t; _t.sleep(self.call_delay_s)

        # Hybrid risk_flags (LLM proposal + forced provable flags)
        fused["risk_flags"] = finalize_risk(
            fused.get("risk_flags", "none"), hist_flags, vision_facts)

        # evidence_standard_met is deterministic; override whatever LLM said
        fused["evidence_standard_met"] = evidence_met
        # Repair the reason if the model returned junk (empty, or literally
        # "true"/"false") — fall back to the matched evidence-rule text so the
        # reason is always a meaningful sentence consistent with the standard.
        _reason = str(fused.get("evidence_standard_met_reason", "")).strip()
        if _reason.lower() in ("", "true", "false", "none", "null"):
            rule_text = rule.get("minimum_image_evidence", "").strip()
            if evidence_met:
                fused["evidence_standard_met_reason"] = (
                    f"The claimed part is visible and assessable: {rule_text}"[:200])
            else:
                fused["evidence_standard_met_reason"] = (
                    f"The claimed part is not clearly assessable in the images; "
                    f"required: {rule_text}"[:200])

        # Validate / repair to closed enums; default part to claimed part
        validated = validate_row(fused, claim_object, parsed["part"])

        # Assemble full 14-col row (echo inputs verbatim)
        out = {
            "user_id": row["user_id"],
            "image_paths": image_paths,
            "user_claim": user_claim,
            "claim_object": claim_object,
            **validated,
        }
        return {c: out.get(c, "") for c in OUTPUT_COLUMNS}

    # ---------- batch ----------
    def run(self, input_csv, output_csv, exclude_self=False, limit=None):
        df = pd.read_csv(input_csv)
        if limit:
            df = df.head(limit)
        rows = []
        for i, (_, r) in enumerate(df.iterrows(), 1):
            try:
                rows.append(self.process_row(r, exclude_self=exclude_self))
            except Exception as e:                       # noqa: BLE001
                # Don't let one bad row kill a long run. Emit a safe, schema-valid
                # fallback (NEI / manual_review_required) and keep going.
                print(f"  [warn] row {i} ({r.get('user_id')}) failed: {e}")
                fallback = {
                    "user_id": r.get("user_id", ""),
                    "image_paths": r.get("image_paths", ""),
                    "user_claim": r.get("user_claim", ""),
                    "claim_object": str(r.get("claim_object", "")).strip().lower(),
                    "evidence_standard_met": "false",
                    "evidence_standard_met_reason": "processing error; needs manual review",
                    "risk_flags": "manual_review_required",
                    "issue_type": "unknown", "object_part": "unknown",
                    "claim_status": "not_enough_information",
                    "claim_status_justification": "row could not be processed automatically",
                    "supporting_image_ids": "none", "valid_image": "false",
                    "severity": "unknown",
                }
                rows.append({c: fallback.get(c, "") for c in OUTPUT_COLUMNS})
            if i % 5 == 0:
                self.vcache.save()
                # incremental save so a crash never loses completed rows
                pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
                print(f"  processed {i}/{len(df)}")
        self.vcache.save()
        out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
        return out, self.acct.as_dict()