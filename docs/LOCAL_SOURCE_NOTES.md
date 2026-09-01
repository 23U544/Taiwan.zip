# Local Source Notes

This package is a curated source-only recovery from the Windows Taiwan.zip project. It intentionally excludes datasets, model weights, generated arrays, point clouds, media, caches, and development-history detector variants.

## Repository placement

- `scripts/depth/`: Depth Anything preprocessing and pseudo-3D export; includes the required project-local `depth_anything_v2` Python package.
- `scripts/parsing/`: frozen multi-label `street_parser_v1.py`.
- `scripts/corpus/`: canonical 604-scene manifest, ingestion, processing, QA, and final curation tools.
- `scripts/dataset/`: Spatial V2 seven-channel dataset builder and validator.
- `scripts/training/`: recovered Spatial VAE Colab trainer candidate.
- `scripts/export/`: generated-state PLY exporter and latest presentation renderer.
- `scripts/utilities/`: small validation/cleanup utilities.
- `configs/`: channel policy/schema and the small Spatial V2 dataset summary.
- `docs/`: research context and phase records.

Paths in the recovered scripts still reflect the original project layout and should be converted to project-root-relative configuration during a future repository integration pass. This package preserves source fidelity and does not rewrite the recovered programs.
