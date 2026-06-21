#!/usr/bin/env python3
"""
risk_flags.py  —  PHASE 4 hybrid risk_flags finalizer.

Implements prompts/risk_flags_contract.md:
  final = normalize( union( llm_flags , forced_flags ) )

forced_flags are provable without the LLM (history CSV + vision facts), so we
never drop them. llm_flags keep the model's judgment calls (claim_mismatch,
manual_review_required, damage_not_visible). normalize() enforces the closed
vocab and stable ordering.
"""
from enums import RISK_FLAGS, RISK_FLAGS_ORDER


def _split(s):
    return [t.strip().lower() for t in str(s).split(";") if t.strip()]


def forced_from_history(history_flags: str) -> set:
    return {t for t in _split(history_flags) if t and t != "none"}


def forced_from_vision(vision_facts: list[dict]) -> set:
    """Derive mechanical flags from per-image vision JSON (Stage A)."""
    flags = set()
    for vf in vision_facts:
        for q in (vf.get("image_quality_flags") or []):
            flags.add(str(q).strip().lower())
        if vf.get("object_mismatch") is True:
            flags.add("wrong_object")
        if vf.get("wrong_object_part") is True:
            flags.add("wrong_object_part")
        if vf.get("possible_manipulation") is True:
            flags.add("possible_manipulation")
        if vf.get("non_original_image") is True:
            flags.add("non_original_image")
        eti = vf.get("embedded_text_instructions")
        if eti and str(eti).strip().lower() not in ("none", "", "null"):
            flags.add("text_instruction_present")
    return flags


def normalize(flags: set) -> str:
    """Keep only allowed tokens, drop 'none' if others present, stable order."""
    kept = {f for f in flags if f in RISK_FLAGS and f != "none"}
    if not kept:
        return "none"
    ordered = [f for f in RISK_FLAGS_ORDER if f in kept]
    # any allowed token not in the canonical order list (shouldn't happen) appended
    ordered += sorted(kept - set(ordered))
    return ";".join(ordered)


def finalize(llm_flags: str, history_flags: str, vision_facts: list[dict]) -> str:
    union = set(_split(llm_flags))
    union |= forced_from_history(history_flags)
    union |= forced_from_vision(vision_facts)
    return normalize(union)
