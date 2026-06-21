# risk_flags finalization contract (HYBRID)

The fusion LLM proposes `risk_flags`. A deterministic post-step then computes the
FINAL `risk_flags` so we never drop a provable flag and never emit out-of-vocab.

final = normalize( union( llm_flags , forced_flags ) )

## forced_flags (computed without the LLM — these are provable)

From user_history.csv for this user_id:
  - every token in `history_flags` (drops "none") is forced in.
    (verified: in all 20 sample rows, history_flags ⊆ row risk_flags)

From the per-image vision facts (Stage A):
  - each entry in `image_quality_flags`  -> same-named risk flag
        (blurry_image, cropped_or_obstructed, low_light_or_glare, wrong_angle)
  - any image with object_mismatch == true        -> wrong_object
  - any image with wrong_object_part == true       -> wrong_object_part
  - any image with possible_manipulation == true   -> possible_manipulation
  - any image with non_original_image == true      -> non_original_image
  - any image with embedded_text_instructions != "none" -> text_instruction_present

## llm_flags (kept from the fusion output — judgment calls the LLM is good at)
  - claim_mismatch
  - manual_review_required
  - damage_not_visible
  (and any of the above mechanical flags it also produced; union is idempotent)

## normalize()
  1. split on ';', strip, lowercase, drop empties
  2. keep ONLY tokens in the allowed risk vocabulary:
       none, blurry_image, cropped_or_obstructed, low_light_or_glare,
       wrong_angle, wrong_object, wrong_object_part, damage_not_visible,
       claim_mismatch, possible_manipulation, non_original_image,
       text_instruction_present, user_history_risk, manual_review_required
     (illegal tokens are dropped; no repair-to-nearest for risk flags —
      dropping is safer than inventing a different risk)
  3. if the set is empty -> ["none"]; otherwise remove "none" if other flags present
  4. de-duplicate, then sort by the canonical vocab order above for stable output
  5. join with ';'

## interaction with manual_review_required
  - if history forces user_history_risk, we do NOT auto-add manual_review_required
    (sample shows user_history_risk can appear alone, e.g. user_031/case_017-style);
    manual_review_required comes from history_flags or the LLM's judgment only.
