# Playing Models Phase 1.6A Completion Report

## Status

Phase 1.6A semantic-quality audit and external-corpus intake preparation is complete. Parser V1 and its 49 canonical outputs were not changed. No detector was retrained, no external image was downloaded or ingested, no generative model was trained, and Point Cloud/Rhino code was untouched.

## A. Audit coverage

- Source scenes audited: **49**
- Semantic tags: **18**, plus `window_group` hierarchy
- Instances audited: **3,888**
- Contact sheets: **15**
- Human review rows: **249**
- Invalid/NaN/Inf instance depth records: **0**
- Instance centroids outside bboxes: **0**
- Masks with `mask_area / bbox_area > 1.2`: **12/3,888**

Outputs include scene/class quantiles, per-scene coverage and outlier flags, instance area/bbox/confidence/depth distributions, mask/bbox consistency, pairwise overlap proxies, contact sheets, and an editable review manifest.

## B. Main semantic finding

Infrastructure integrity remains valid, but several semantic channels are not suitable as unqualified training targets. Door, balcony, awning, and storefront have median scene coverage of 37.8%, 45.1%, 35.0%, and 40.0%, with p95 values around 96%. The same catastrophic scenes recur across classes: 000001, 000013, 000037, 000039, and 000015.

Contact sheets show that the main failure is an oversized/wrong Grounding DINO proposal covering a facade, building, grille, sign, or full frame. SAM generally stays within that accepted bbox; broad mask spill outside the bbox is rare. Therefore the anomaly is primarily proposal-level semantic/scale confusion, with SAM coarseness and legitimate multi-label overlap secondary. This is a heuristic diagnosis, not a ground-truth precision/recall result.

AC, person, signboard, utility pole, rolling shutter, and street object also contain isolated catastrophic outliers. Vehicle is the most consistently bounded instance channel. Window remains architecturally valuable but visibly confuses signs, panels, doors, and openings in some samples. Facade is useful broad context but reaches full-image coverage in several scenes.

## C. Overlap proxies

Aggregate overlap over the smaller mask is 95.6% for storefront–door, 95.3% for awning–storefront, 95.0% for balcony–awning, 90.3% for balcony–railing, 79.3% for grille–railing, and 76.8% for Window–balcony. These values identify review priorities; they are not a confusion matrix because overlapping labels can be legitimate.

## D. Training channel policy

`TRAINING_CHANNEL_POLICY_V1.json` separates parser reliability from training role, loss weight, and inclusion. The conservative candidate is:

```text
depth_norm
facade
window
signboard
vegetation
person
vehicle
```

This is a **7-channel candidate**, not a frozen final schema. Vehicle and reduced-weight facade are CORE; Window, signboard, vegetation, and person are AUXILIARY. Railing, rolling shutter, and utility pole are retained as LOW_WEIGHT candidates but excluded until review. Door, balcony, grille, awning, storefront, AC, wire, street object, and arcade are EXCLUDE_V1. Exclusion affects exports only; all canonical layers remain intact.

The exporter now accepts a versioned `--policy` JSON and writes class-level weights into its channel schema. Parser V1 has no reliable pixel-level confidence map, so none was fabricated.

## E. Policy export verification

- Source scenes: **49**
- Preview samples: **98**
- Resolution: **256×256**
- Channels: **7**
- Train/Val/Test/Regression: **66/8/8/16**
- Tensor and group-leakage validation: **PASS**
- Export size: approximately **20.4 MB**

## F. Patch-resolution review

Current Window bbox size is approximately 24×29 pixels at p10, 66×78 at median, and 207×219 at p90 in source pixels. AC/person p10 dimensions are around 20 and 15×30 pixels. A 256 crop is efficient and valid for pipeline testing but often lacks facade/street context. A 384 crop costs about 2.25× the spatial memory/compute of 256 while providing 50% more context per dimension; it is the recommended next comparison. A 512 crop costs 4× 256 and better represents large facade structure but reduces batch size substantially on 6 GB VRAM. No final resolution is frozen; no large 384/512 export was generated.

## G. External intake readiness

`external_corpus_intake.py` accepts approved images plus approved metadata CSV, derives stable IDs without overwriting scenes 1–49, preserves full provenance, generates deterministic fallback source groups, and compares both source SHA-256 and pHash against the existing corpus. Source-file hashes for the legacy 49 scenes were backfilled from `training_images` where available.

A temporary dry run using existing `street01.jpg` now correctly skips it as an exact duplicate of scene 000001. No scene or manifest row was created, and the fixture was removed.

The handoff is unified:

```text
external intake → DEPTH_PENDING
corpus_depth.py → PARSING_PENDING
corpus_pipeline.py → QA/READY
```

`corpus_depth.py` is resumable and shardable, preserves provenance metadata, writes relative raw/normalized depth, records future timing, and shares the persistent failure log. Its current dry run selected 49 scenes, queued 0, and skipped all 49 valid depth outputs.

## H. Bulk estimate and placement

Measured Parser V1 performance remains 60.4 s/scene. Deep QA is approximately 3.0 s/scene. Existing depth file timestamps suggest roughly 0.45 s/scene after model initialization, but this is explicitly treated as a proxy until the new depth worker records controlled timings.

- 500 new scenes: about 8.87 wall-hours and 8.04 GB.
- 750 new scenes: about 13.30 wall-hours and 12.07 GB.
- 1000 new scenes: about 17.74 wall-hours and 16.09 GB.

Ingestion/provenance, reconciliation, QA, review, statistics, and export should remain local. Depth is fast enough locally unless data is already cloud-resident. Semantic parsing dominates runtime and is the best Colab/sharded candidate. Recommended shard size is approximately 100 scenes, with 5 shards for 500 and 10 for 1000.

## I. Readiness decisions

- **Infrastructure Ready? YES.** Manifest, state machine, QA, resume, shard, policy export, and review artifacts are operational.
- **Semantic Core Channels Ready? YES, conditionally.** The conservative 7-channel candidate is suitable for controlled experiments after review; weights do not imply detector accuracy.
- **All 18 Semantic Channels Ready? NO.** Catastrophic oversized proposal confusion blocks all-channel training.
- **Ready for 500–1000 Corpus Intake? YES.** Approved CSV/image intake, provenance, exact/pHash dedupe, stable IDs, depth handoff, parsing, QA, and shards are connected.
- **Ready for Formal Generative Training? NO.** The external corpus is not complete, human review is unchecked, legacy source grouping is provisional, and several semantic classes remain excluded.

## Stop condition

Phase 1.6A stops here. Wait for completion of Taiwan Streetscape Source Corpus download/screening and the next instruction.
