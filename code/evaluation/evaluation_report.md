# Evaluation Report — Multi-Modal Evidence Review

Evaluated on **20** labeled sample rows. Leakage guard ON (no case retrieves itself).


## Overall

- **Row accuracy (all graded columns correct): 10.0%**


## Per-column metrics

| Column | Accuracy | Macro-F1 |
|---|---|---|
| evidence_standard_met | 75.0% | 0.57 |
| issue_type | 35.0% | 0.27 |
| object_part | 75.0% | 0.61 |
| claim_status | 80.0% | 0.66 |
| severity | 40.0% | 0.29 |
| valid_image | 90.0% | 0.72 |
| supporting_image_ids | 80.0% | — |

**risk_flags** (multi-label, set-level): Precision 0.34 · Recall 0.38 · F1 0.36


## claim_status confusion matrix

(rows = gold, cols = predicted)

| gold \ pred | supported | contradicted | not_enough_information |
|---|---|---|---|
| supported | 12 | 0 | 0 |
| contradicted | 1 | 3 | 1 |
| not_enough_information | 1 | 1 | 1 |

## Worst misses (18 rows with >=1 error)

| case | object | #wrong | wrong columns | images |
|---|---|---|---|---|
| case_018 | package | 8 | evidence_standard_met, risk_flags, issue_type, object_part, claim_status, supporting_image_ids, valid_image, severity | `images/sample/case_018/img_1.jpg;images/sample/case_018/img_2.jpg` |
| case_008 | car | 7 | evidence_standard_met, risk_flags, issue_type, object_part, claim_status, supporting_image_ids, severity | `images/sample/case_008/img_1.jpg` |
| case_002 | car | 6 | evidence_standard_met, risk_flags, issue_type, claim_status, supporting_image_ids, severity | `images/sample/case_002/img_1.jpg;images/sample/case_002/img_2.jpg` |
| case_019 | package | 5 | evidence_standard_met, risk_flags, issue_type, object_part, severity | `images/sample/case_019/img_1.jpg` |
| case_014 | laptop | 5 | evidence_standard_met, risk_flags, issue_type, object_part, severity | `images/sample/case_014/img_1.jpg` |
| case_020 | package | 4 | risk_flags, issue_type, claim_status, severity | `images/sample/case_020/img_1.jpg;images/sample/case_020/img_2.jpg` |
| case_005 | car | 4 | risk_flags, issue_type, supporting_image_ids, severity | `images/sample/case_005/img_1.jpg;images/sample/case_005/img_2.jpg` |
| case_017 | package | 3 | risk_flags, issue_type, object_part | `images/sample/case_017/img_1.jpg` |
| case_003 | car | 3 | risk_flags, issue_type, severity | `images/sample/case_003/img_1.jpg;images/sample/case_003/img_2.jpg` |
| case_013 | laptop | 2 | issue_type, severity | `images/sample/case_013/img_1.jpg` |
| case_011 | laptop | 2 | risk_flags, issue_type | `images/sample/case_011/img_1.jpg` |
| case_010 | laptop | 2 | risk_flags, severity | `images/sample/case_010/img_1.jpg;images/sample/case_010/img_2.jpg` |
| case_009 | laptop | 2 | issue_type, severity | `images/sample/case_009/img_1.jpg` |
| case_006 | car | 2 | risk_flags, valid_image | `images/sample/case_006/img_1.jpg` |
| case_012 | laptop | 1 | risk_flags | `images/sample/case_012/img_1.jpg;images/sample/case_012/img_2.jpg` |

### Side-by-side for the worst cases


**case_002** — claim: _Customer: Parking lot mein meri car ko scrape lag gaya. | Support: Aap kis type ka damage report karna chahte hain? | Customer: Front side p_

| col | gold | pred |
|---|---|---|
| evidence_standard_met | False | true |
| issue_type | broken_part | scratch |
| object_part | front_bumper | front_bumper |
| claim_status | not_enough_information | supported |
| severity | unknown | medium |
| valid_image | True | true |
| risk_flags | wrong_object;claim_mismatch;manual_review_required | none |
| supporting_image_ids | img_1;img_2 | img_1 |

**case_008** — claim: _Customer: I picked up my car after service and noticed a mark on the hood. | Support: What kind of mark is it? | Customer: It looks like a s_

| col | gold | pred |
|---|---|---|
| evidence_standard_met | True | false |
| issue_type | broken_part | unknown |
| object_part | front_bumper | hood |
| claim_status | contradicted | not_enough_information |
| severity | high | unknown |
| valid_image | False | false |
| risk_flags | claim_mismatch;non_original_image;user_history_risk;manual_review_required | blurry_image;low_light_or_glare;wrong_object_part;possible_manipulation;non_original_image;user_history_risk |
| supporting_image_ids | img_1 | none |

**case_014** — claim: _Customer: The laptop trackpad has stopped working properly. | Support: Did anything happen before it stopped working? | Customer: The front _

| col | gold | pred |
|---|---|---|
| evidence_standard_met | True | false |
| issue_type | none | scratch |
| object_part | trackpad | base |
| claim_status | contradicted | contradicted |
| severity | none | low |
| valid_image | True | true |
| risk_flags | damage_not_visible;user_history_risk;manual_review_required | claim_mismatch;possible_manipulation;non_original_image;user_history_risk |
| supporting_image_ids | img_1 | img_1 |

**case_018** — claim: _Customer: The item I ordered was not inside the box. | Support: Did the package look opened when you received it? | Customer: I checked it a_

| col | gold | pred |
|---|---|---|
| evidence_standard_met | False | true |
| issue_type | unknown | none |
| object_part | contents | seal |
| claim_status | not_enough_information | contradicted |
| severity | unknown | none |
| valid_image | False | true |
| risk_flags | cropped_or_obstructed;damage_not_visible;manual_review_required | user_history_risk;manual_review_required |
| supporting_image_ids | none | img_1;img_2 |

**case_019** — claim: _Customer: The shipping box arrived in bad condition. | Support: What kind of condition issue are you reporting? | Customer: It looked badly _

| col | gold | pred |
|---|---|---|
| evidence_standard_met | True | false |
| issue_type | unknown | crushed_packaging |
| object_part | unknown | package_corner |
| claim_status | contradicted | contradicted |
| severity | low | none |
| valid_image | True | true |
| risk_flags | wrong_object;claim_mismatch;user_history_risk;manual_review_required | claim_mismatch;user_history_risk |
| supporting_image_ids | img_1 | img_1 |

**case_020** — claim: _Customer: My delivery box arrived opened. | Support: Was the package crushed or was the seal affected? | Customer: The seal area looked torn_

| col | gold | pred |
|---|---|---|
| evidence_standard_met | True | true |
| issue_type | none | torn_packaging |
| object_part | seal | seal |
| claim_status | contradicted | supported |
| severity | none | medium |
| valid_image | True | true |
| risk_flags | damage_not_visible;text_instruction_present;user_history_risk;manual_review_required | possible_manipulation;non_original_image;text_instruction_present;user_history_risk |
| supporting_image_ids | img_1;img_2 | img_1;img_2 |

## Operational analysis

### Operational analysis — sample (backend: groq)

| Metric | Value |
|---|---|
| Vision calls | 29 |
| Fusion calls | 20 |
| Embedding calls | 40 |
| Unique images processed | 29 |
| Vision cache hits (reused) | 0 |
| Input (prompt) tokens | 81,641 |
| Output tokens | 7,274 |
| Total tokens | 88,915 |
| Measured wall-clock latency | 37.4 s |
| Avg latency / call | 0.4 s |

**Cost story.**

Using **gemini-2.5-flash-lite** at assumed public pricing ($0.30/1M input, $2.50/1M output tokens):

- Input:  81,641 tok / 1e6 × $0.30 = **$0.0245**
- Output: 7,274 tok / 1e6 × $2.50 = **$0.0182**
- **Total for this run: $0.0427**  (~$0.0009 per claim row)

Assumptions: prices as of the cited tier and may change; image tokens are counted by the API in the input total; figures exclude any free-tier credits. The local Ollama backend remains available at ~$0 marginal cost for a fully reproducible, credential-free re-run.

**Throughput / rate-limit story.**

Gemini enforces TPM/RPM limits per tier. We stay under them with: (1) temperature 0 + JSON-constrained outputs (short, predictable token counts), (2) the vision cache (each unique image billed/processed once, re-runs reuse results), (3) exponential **retry/backoff** that absorbs 429/quota responses, and (4) one image per call to keep request sizes small. To respect RPM on large batches, increase the backoff multiplier or add a small inter-call sleep in config.

**Projection math.**

This run processed 29 unique images in 37s (~1.3s per image including fusion overhead). The test set (44 rows / 82 unique images) scales roughly linearly: ≈ 106s wall-clock cold, and near-instant on a warm cache for unchanged images.

### Operational analysis — test (claims.csv) (backend: groq)

| Metric | Value |
|---|---|
| Vision calls | 67 |
| Fusion calls | 36 |
| Embedding calls | 36 |
| Unique images processed | 67 |
| Vision cache hits (reused) | 0 |
| Input (prompt) tokens | 168,855 |
| Output tokens | 15,445 |
| Total tokens | 184,300 |
| Measured wall-clock latency | 127.1 s |
| Avg latency / call | 0.9 s |

**Cost story.**

Using **gemini-2.5-flash-lite** at assumed public pricing ($0.30/1M input, $2.50/1M output tokens):

- Input:  168,855 tok / 1e6 × $0.30 = **$0.0507**
- Output: 15,445 tok / 1e6 × $2.50 = **$0.0386**
- **Total for this run: $0.0893**  (~$0.0009 per claim row)

Assumptions: prices as of the cited tier and may change; image tokens are counted by the API in the input total; figures exclude any free-tier credits. The local Ollama backend remains available at ~$0 marginal cost for a fully reproducible, credential-free re-run.

**Throughput / rate-limit story.**

Gemini enforces TPM/RPM limits per tier. We stay under them with: (1) temperature 0 + JSON-constrained outputs (short, predictable token counts), (2) the vision cache (each unique image billed/processed once, re-runs reuse results), (3) exponential **retry/backoff** that absorbs 429/quota responses, and (4) one image per call to keep request sizes small. To respect RPM on large batches, increase the backoff multiplier or add a small inter-call sleep in config.

**Projection math.**

This run processed 67 unique images in 127s (~1.9s per image including fusion overhead). The test set (44 rows / 82 unique images) scales roughly linearly: ≈ 156s wall-clock cold, and near-instant on a warm cache for unchanged images.
