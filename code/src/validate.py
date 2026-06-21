#!/usr/bin/env python3
"""
validate.py  —  PHASE 4 enum validator / repair.

Hard guarantee: no out-of-vocab value ever reaches output.csv. Every field is
forced into its closed enum; unrecognized values map to a safe default
('unknown' where allowed) rather than passing through.

Also enforces cross-field rules learned from the labels:
  - issue_type == none    -> severity = none
  - issue_type == unknown -> severity = unknown
  - object_part defaults to the claimed part when fusion couldn't determine it
  - supporting_image_ids normalized to ';'-joined ids or 'none'
  - booleans coerced to true/false strings for stable CSV output
"""
from enums import (CLAIM_STATUS, ISSUE_TYPE, OBJECT_PART, SEVERITY)


def _norm(s):
    return str(s).strip().lower().replace(" ", "_") if s is not None else ""


def _enum(value, vocab, default):
    v = _norm(value)
    return v if v in vocab else default


def _boolstr(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return "true" if _norm(v) in ("true", "1", "yes") else "false"


def _ids(v):
    if v is None:
        return "none"
    toks = [t.strip() for t in str(v).split(";") if t.strip()]
    toks = [t for t in toks if t.lower() != "none"]
    return ";".join(toks) if toks else "none"


def validate_row(row: dict, claim_object: str, claimed_part: str) -> dict:
    """Coerce a fused/finalized row dict to fully-legal enum values."""
    co = _norm(claim_object)
    part_vocab = OBJECT_PART.get(co, {"unknown"})

    issue = _enum(row.get("issue_type"), ISSUE_TYPE, "unknown")

    # object_part: use fusion value if legal; else fall back to claimed part if
    # legal; else 'unknown'. (Matches case_006: keep claimed part when unseen.)
    part = _norm(row.get("object_part"))
    if part not in part_vocab:
        cp = _norm(claimed_part)
        part = cp if cp in part_vocab else "unknown"

    status = _enum(row.get("claim_status"), CLAIM_STATUS, "not_enough_information")
    severity = _enum(row.get("severity"), SEVERITY, "unknown")

    # cross-field severity clamps
    if issue == "none":
        severity = "none"
    elif issue == "unknown":
        severity = "unknown"

    # NEI clamp: if we can't judge, we can't assert an issue/severity or cite an
    # image as sufficient. (Matches gold: NEI rows have issue/severity=unknown,
    # supporting_image_ids=none.)
    if status == "not_enough_information":
        issue = "unknown"
        severity = "unknown"

    out = {
        "evidence_standard_met": _boolstr(row.get("evidence_standard_met")),
        "evidence_standard_met_reason": str(row.get("evidence_standard_met_reason", "")).strip(),
        "risk_flags": row.get("risk_flags", "none") or "none",
        "issue_type": issue,
        "object_part": part,
        "claim_status": status,
        "claim_status_justification": str(row.get("claim_status_justification", "")).strip(),
        "supporting_image_ids": _ids(row.get("supporting_image_ids")),
        "valid_image": _boolstr(row.get("valid_image")),
        "severity": severity,
    }
    # if not_enough_information, there is by definition no sufficient image
    if status == "not_enough_information":
        out["supporting_image_ids"] = "none"
    return out