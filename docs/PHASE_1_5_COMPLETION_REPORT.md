# Playing Models Phase 1.5 Completion Report

## Status

Phase 1.5 scalable corpus and training-data infrastructure is complete for the current local corpus. No neural network was trained, no model was downloaded, and frozen `street_parser_v1.py`, Point Cloud, Rhino conventions, and core RGB/depth files were not modified.

## A. Current Corpus

- Corpus version: `playing-models-corpus-v1`
- Independent local source scenes: **49**
- Readable RGB: **49/49**
- Finite, RGB-aligned `depth_raw` and `depth_norm`: **49/49**
- Complete `playing-models-street-parser-v1.0` outputs: **49/49**
- Deep automated QA pass: **49/49**
- Pipeline failures: **0**
- Pipeline status: **49 READY**
- Fixed regression/QA set: 8 scenes, retained outside ordinary train/val/test

The batch added Parser V1 output for the 41 previously missing scenes and skipped the 8 valid existing outputs. Automated QA checks integrity, not semantic correctness.

## B. Current Storage

Measured canonical mean is **16.48 MB/scene**: RGB 0.49, depth raw 5.42, depth normalized 5.42, depth preview 0.15, semantic masks 0.12, instance masks 0.22, parser JSON 1.72, parser previews 2.79, and debug 0.14 MB. Projection is **0.79 GB for 49**, **8.04 GB for 500**, and **16.09 GB for 1000** scenes.

Depth arrays dominate storage. Compressed instance masks project to only **0.22 GB at 1000 scenes** in this corpus, so an immediate representation migration is not justified. Parser JSON is currently larger than instance masks because dense pairwise relationships grow approximately quadratically with instance count; this should be monitored before 1000-scene expansion.

## C. Duplicate Analysis

SHA-256 exact hashes and 64-bit perceptual hashes were computed. With pHash Hamming distance ≤ 8 used only as a review trigger:

- Exact duplicates: **0**
- Near-duplicate candidates: **0**

No scene is auto-deleted on perceptual similarity. Future adjacent-frame and visual-diversity analysis can add CLIP embeddings without changing manifest identity.

## D. Diversity

The corpus is facade-heavy: facade covers **62.96%** of parsed pixels, and Window occurs in 47/49 scenes. Least represented stable evidence includes vegetation (27/49 scenes, 0.83% pixels), vehicle (36/49, 2.67%), person (42/49, 1.10%), and utility pole (45/49, 1.78%). `arcade_candidate` remains intentionally empty and experimental.

Several MEDIUM channels have implausibly high pixel coverage across the corpus—door 31.98%, balcony 38.51%, awning 35.61%, and storefront 33.93%. This is a semantic-quality warning consistent with coarse/overlapping SAM masks, not an integrity failure. Before formal training, visually audit a stratified sample and consider reliability-aware channel selection. Future lawful acquisition should prioritize independent source groups containing narrow alleys, arcade/ground-floor commercial structure, old apartments, townhouses, sign-heavy shophouses, metal-sheet additions, and vegetation/vehicle-heavy streets, while maintaining dense-facade versus perspective balance.

## E. Parser Scalability

`corpus_pipeline.py` reads corpus state, selects missing/failed/version-mismatched scenes, skips valid outputs, supports `--start/--end`, explicit `--scenes`, and zero-based `--shard/--num-shards`, and records per-scene failures in persistent `dataset/pipeline_failures.jsonl`. Reruns are idempotent unless `--force` is given. Hugging Face offline mode is set for parser workers.

The current run processed 41 scenes in approximately **41.3 minutes** on the local RTX 4050, about **60.4 seconds per scene** end-to-end on average. The 4096×3072 scene was substantially slower than typical scenes, demonstrating the effect of tiled search and full-resolution mask compression.

## F. Local vs Colab

Local is appropriate for ingestion, manifest refresh, duplicate review, QA, statistics, dry-run export, debugging, and small incremental parser batches. At the measured rate, 500 new scenes would require roughly 8.4 GPU-hours and 1000 roughly 16.8 GPU-hours before overhead and large-image variance. Colab or another persistent GPU worker is therefore appropriate for bulk depth/parsing shards. Outputs must be synchronized to persistent storage after every scene; `/content` must never be the canonical corpus.

## G. Candidate Training Representation V1

Candidate V1 is `[depth_norm, facade, window, door, balcony, railing, awning, storefront, rolling_shutter, signboard, air_conditioner, utility_pole, vegetation, person, vehicle, street_object]`, yielding **16 channels** in `[C,H,W]`. `depth_norm` is float32 scene-relative depth; semantic planes are exported as float32 binary values. Raw depth remains canonical but is not treated as cross-scene metric distance. Grille, wire, and arcade are retained canonically and excluded by default; `--include-experimental` restores them. RGB crops are optional.

This is a candidate schema, not a permanently frozen taxonomy. Given the pixel-prevalence warning above, a conservative first experiment may use a smaller high-confidence subset or reliability weighting after visual review.

## H. Split Strategy

The hard ordering is implemented as original scene → source group → split → patches. Every `split_group` is assigned once using deterministic hash ordering and then allocated approximately 80/10/10 among non-regression groups. The eight regression scenes remain a separate split. Validation confirmed no source-group appears in more than one split.

Legacy provenance is `existing_local`; unavailable license/attribution fields remain empty rather than invented. Current legacy scenes use conservative singleton source groups because their original sequence/building grouping is unknown. Before treating these splits as research-grade generalization evidence, manually populate known shared groups for adjacent frames, panoramas, or repeated viewpoints.

## I. Storage Optimization

Keep existing lossless compressed NPZ instance masks for now. They are not the measured storage bottleneck. If future higher-resolution scenes change that conclusion, introduce a versioned bit-packed or COCO-RLE archive with exact full-resolution reconstruction and retain compatibility with Parser V1. Training exports correctly omit instance masks, debug images, raw proposals, and canonical previews.

## J. Dry-run Training Dataset

- Export: `dataset/training_v1_dryrun`
- Version: `playing-models-training-v1`
- Source scenes: **49**
- Patch resolution: **256×256**
- Samples: **384**
- Channels: **16**
- Train: **256** samples
- Validation: **32** samples
- Test: **32** samples
- Regression: **64** samples
- Export size: approximately **77.3 MB**

Validation passed for every sample: finite tensor, correct channel/shape, binary semantics, scene/crop traceability, and zero group leakage. The 256 target is suitable for this infrastructure dry run and lower-memory Colab experiments, but 384 should be compared later for small architectural objects before a final training resolution is selected.

## K. Scaling to 500

Acquire lawful, attributable independent source scenes; assign real source groups during ingestion; run duplicate review; generate/resume missing depth and parsing in shards; perform deep QA; visually audit stratified reliability; refresh diversity and storage reports; then create a versioned export. Expected canonical storage is about 8.04 GB and parser compute about 8.4 GPU-hours at the current mean.

## L. Scaling to 1000

Use persistent multi-session GPU shards, reconcile one canonical manifest, retain never-tuned source groups, monitor relationship JSON growth and resolution outliers, and rebalance acquisition based on feature coverage rather than patch count. Expected canonical storage is about 16.09 GB and parser compute about 16.8 GPU-hours at the current mean.

## M. Ready for Corpus Expansion?

**YES.** Stable IDs, provenance fields, hashes, duplicate review, explicit states, resume/skip/retry, shards, integrity QA, feature statistics, group-safe splits, and traceable exports are implemented for the target 500–1000-scene scale.

## N. Ready for Colab Training?

**Pipeline/interface: YES. Data volume and semantic-label confidence: NO.** The dry run proves Colab-readable tensors, schemas, splits, and resume interfaces. Forty-nine scenes are not sufficient for the intended generative study, legacy source grouping needs enrichment, and medium-reliability mask coverage requires visual/quantitative review before formal model training.

## Stop Condition

Phase 1.5 stops here. No VAE, autoencoder, GAN, diffusion, CNN, latent-space, point-cloud, or geometry training has started.
