# Taiwan.zip

Taiwan.zip is an architectural research workflow that transforms Taiwanese streetscape photographs into relative depth, multi-label semantic fields, pseudo-3D point clouds, and learned spatial latent states.

The project does not ask AI to directly generate architecture as a finished image. Machine perception, abstraction, misreading, spatial reconstruction, and latent recombination become design operations.

```text
Taiwanese street image
→ relative depth
→ semantic decomposition
→ spatial dataset
→ Spatial VAE V2
→ latent transformations
→ point cloud / Rhino–Grasshopper workflow
```

## Repository structure

- `scripts/depth/` — Depth Anything V2 preprocessing and image-to-point-cloud tools
- `scripts/parsing/` — frozen multi-label Taiwan streetscape parser
- `scripts/corpus/` — corpus discovery, review, manifest, bulk processing, and QA
- `scripts/dataset/` — seven-channel Spatial V2 dataset builder and validator
- `scripts/training/` — final spatial-latent VAE V2 trainer
- `scripts/render/` — checkpoint-driven latent morph and presentation renderers
- `scripts/export/` — generated semantic/depth fields to PLY/XYZ
- `scripts/legacy/` — preserved earlier global-latent and bundle-fallback programs
- `notebooks/` — Colab presentation/training notebook
- `configs/` — channel policy and dataset schema/summary
- `media/` — lightweight representative project media
- `metadata/` — compact recorded run metadata
- `docs/` — technical context and phase reports
- `inventories/` — source selection and exclusion provenance

## Final model entry points

Training:

```bash
python scripts/training/taiwan_zip_spatial_vae_v2.py --help
```

Checkpoint-driven final journey renderer:

```bash
python scripts/render/render_taiwan_zip_v6_3_model_fix.py --help
```

Compact latent morph renderer:

```bash
python scripts/render/render_taiwan_zip_morph.py --help
```

Generated-state PLY export:

```bash
python scripts/export/taiwan_zip_generated_to_ply.py --help
```

## Data and checkpoints

This GitHub folder is intentionally source-first. It does not include the 604-scene image dataset, generated NPZ arrays, point-cloud exports, downloaded foundation-model weights, or the trained Spatial VAE checkpoint.

Expected external artifacts include:

- `taiwan_zip_spatial_v2_bundle.zip`
- `taiwan_zip_spatial_v2_best.pt`
- Depth Anything V2 checkpoint
- local `dataset/scenes/` when running the full pipeline

Keep large artifacts in release storage, cloud drive, or another dataset/model registry rather than Git history.

## Technical conventions

- Training channels: `depth_norm`, `facade`, `window`, `signboard`, `vegetation`, `person`, `vehicle`
- Relative depth: `0 = farther`, `1 = nearer`; it is not metric depth
- Semantics are independent multi-label fields
- Rhino convention: `+X = image right`, `+Y = forward/depth`, `+Z = up`

See `docs/PLAYING_MODELS_CONTEXT.md` and `GITHUB_INTEGRATION_REPORT.md` before reproducing the complete workflow.
