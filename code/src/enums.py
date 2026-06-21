#!/usr/bin/env python3
"""
enums.py  —  the closed allowed-value vocabulary from problem_statement.md.

Single source of truth. The validator (validate.py) imports ONLY from here so
no illegal value can ever reach output.csv.
"""

OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
    "issue_type", "object_part", "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
]

CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}

ISSUE_TYPE = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain",
    "none", "unknown",
}

OBJECT_PART = {
    "car": {
        "front_bumper", "rear_bumper", "door", "hood", "windshield",
        "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
        "body", "unknown",
    },
    "laptop": {
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port",
        "base", "body", "unknown",
    },
    "package": {
        "box", "package_corner", "package_side", "seal", "label", "contents",
        "item", "unknown",
    },
}

SEVERITY = {"none", "low", "medium", "high", "unknown"}

# Canonical order used to render risk_flags deterministically (stable output).
RISK_FLAGS_ORDER = [
    "blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle",
    "wrong_object", "wrong_object_part", "damage_not_visible", "claim_mismatch",
    "possible_manipulation", "non_original_image", "text_instruction_present",
    "user_history_risk", "manual_review_required",
]
RISK_FLAGS = set(RISK_FLAGS_ORDER) | {"none"}
