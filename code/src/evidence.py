#!/usr/bin/env python3
"""
evidence.py  —  PHASE 4 deterministic layer.

Two deterministic jobs (no LLM guessing):
  1. parse_claim(): pull the claimed issue family + part keywords from the
     chat transcript, used to (a) pick the evidence rule, (b) hint the vision
     stage, (c) default object_part when the part isn't visible.
  2. match_evidence_rule(): given claim_object + claimed issue family, return
     the matching row from evidence_requirements.csv. evidence_standard_met is
     then decided by whether the vision stage found the relevant part
     ASSESSABLE (computed in pipeline.py), grounded on this rule's text.
"""
import re

import pandas as pd

# Map free-text claim keywords -> issue family used in evidence_requirements.applies_to
# Families are matched per object; 'applies_to' strings come from the CSV.
_ISSUE_KEYWORDS = {
    "dent": ["dent", "dented", "ding"],
    "scratch": ["scratch", "scrape", "scuff", "mark", "scratched"],
    "crack": ["crack", "cracked", "cracking"],
    "glass_shatter": ["shatter", "shattered", "smash"],
    "broken_part": ["broke", "broken", "snapped", "detached", "hanging"],
    "missing_part": ["missing", "fell off", "gone", "lost", "not there"],
    "torn_packaging": ["torn", "ripped", "tear", "opened", "seal"],
    "crushed_packaging": ["crushed", "crush", "smashed box", "dented box"],
    "water_damage": ["water", "wet", "soaked", "moisture", "damp"],
    "stain": ["stain", "stained", "discolor", "spill"],
}

# Part keywords per object -> canonical object_part value.
_PART_KEYWORDS = {
    "car": {
        "rear_bumper": ["rear bumper", "back bumper", "rear", "back of the car", "behind"],
        "front_bumper": ["front bumper", "front", "front-end", "front end"],
        "door": ["door"],
        "hood": ["hood", "bonnet", "top panel"],
        "windshield": ["windshield", "windscreen", "front glass"],
        "side_mirror": ["side mirror", "wing mirror", "mirror"],
        "headlight": ["headlight", "head light", "head lamp"],
        "taillight": ["taillight", "tail light", "rear light", "brake light"],
        "fender": ["fender"],
        "quarter_panel": ["quarter panel"],
        "body": ["body", "panel"],
    },
    "laptop": {
        "screen": ["screen", "display", "lcd", "panel"],
        "keyboard": ["keyboard", "keys", "key"],
        "trackpad": ["trackpad", "touchpad"],
        "hinge": ["hinge"],
        "lid": ["lid", "top cover"],
        "corner": ["corner"],
        "port": ["port", "usb", "hdmi", "jack", "charging port"],
        "base": ["base", "bottom"],
        "body": ["body", "chassis", "casing"],
    },
    "package": {
        "package_corner": ["corner"],
        "seal": ["seal", "tape", "torn open", "arrived opened", "opened"],
        "label": ["label", "address", "shipping label"],
        "contents": ["contents", "inside", "item inside", "product inside", "not inside"],
        "package_side": ["side", "flap"],
        "item": ["item", "product"],
        "box": ["box", "package", "parcel", "carton"],
    },
}


def _clean(text: str) -> str:
    parts = []
    for seg in str(text).split("|"):
        seg = seg.strip()
        for tag in ("Customer:", "Support:"):
            if seg.startswith(tag):
                seg = seg[len(tag):].strip()
        parts.append(seg)
    return " ".join(parts).lower()


def parse_claim(user_claim: str, claim_object: str) -> dict:
    """Return {'issue': <issue_type or 'unknown'>, 'part': <object_part or 'unknown'>}."""
    t = _clean(user_claim)
    issue = "unknown"
    for fam, kws in _ISSUE_KEYWORDS.items():
        if any(k in t for k in kws):
            issue = fam
            break
    part = "unknown"
    for canon, kws in _PART_KEYWORDS.get(claim_object, {}).items():
        if any(k in t for k in kws):
            part = canon
            break
    return {"issue": issue, "part": part}


# Map a parsed issue family -> the applies_to bucket(s) in evidence_requirements.
def _family_to_rule(claim_object: str, issue: str, part: str):
    co = claim_object
    if co == "car":
        if issue in ("dent", "scratch"):
            return "dent or scratch"
        if issue in ("crack", "glass_shatter", "broken_part", "missing_part"):
            return "crack, broken, or missing part"
        return "vehicle identity or orientation"
    if co == "laptop":
        if part in ("screen", "keyboard", "trackpad") or issue in ("crack", "stain"):
            return "screen, keyboard, or trackpad"
        return "hinge, lid, corner, body, or port"
    if co == "package":
        if issue in ("crushed_packaging", "torn_packaging") or part in ("seal", "package_corner", "package_side", "box"):
            return "crushed, torn, or seal damage"
        if issue in ("water_damage", "stain") or part in ("label",):
            return "water, stain, or label damage"
        if part in ("contents", "item") or issue == "missing_part":
            return "contents or inner item"
        return "crushed, torn, or seal damage"
    return None


class EvidenceRules:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

    def match(self, claim_object: str, issue: str, part: str) -> dict:
        """Return the best-matching rule row as a dict (always returns something)."""
        applies = _family_to_rule(claim_object, issue, part)
        df = self.df
        # try object-specific + applies_to
        if applies:
            hit = df[(df.claim_object == claim_object) & (df.applies_to == applies)]
            if len(hit):
                return hit.iloc[0].to_dict()
        # fall back to object-level general, then the 'all' general rule
        hit = df[(df.claim_object == claim_object)]
        if len(hit):
            return hit.iloc[0].to_dict()
        gen = df[(df.claim_object == "all") & (df.applies_to == "general claim review")]
        return gen.iloc[0].to_dict() if len(gen) else df.iloc[0].to_dict()
