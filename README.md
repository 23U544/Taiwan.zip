# Taiwan.zip

<p align="center">
  <img src="media/hero_main_visual.png" alt="Taiwan.zip — final hybrid spatial field" width="100%">
</p>

<p align="center">
  <em>Taiwanese streetscape → machine perception → latent spatial transformation → reconstructed point-cloud field</em>
</p>

Taiwan.zip is an architectural research workflow that transforms Taiwanese streetscape photographs into relative depth, multi-label semantic fields, pseudo-3D point clouds, and learned spatial latent states.

The project does not ask AI to directly generate architecture as a finished image. Machine perception, abstraction, misreading, spatial reconstruction, and latent recombination become design operations.

## Visual workflow

<p align="center">
  <img src="media/diagrams/diagram_01_2d_to_3d.png" alt="From streetscape image to relative depth and semantic point cloud" width="100%">
</p>

```text
Observed Taiwanese street
→ estimated relative depth
→ multi-label semantic decomposition
→ depth-lifted spatial field
→ Spatial VAE V2
→ latent transformation
→ point cloud / Rhino–Grasshopper workflow
```

The input remains pixel-aligned through depth and semantic processing. Point clouds are derived spatial representations rather than the primary training format.

## Playing the latent model

<p align="center">
  <img src="media/diagrams/diagram_02_latent_operations.png" alt="Interpolation, extrapolation and local perturbation in Taiwan.zip" width="100%">
</p>

The learned spatial field can be navigated through three related operations:

| Operation | Spatial reading |
|---|---|
| **Interpolation** | Moves between two learned streetscape states |
| **Extrapolation** | Extends a learned direction beyond its source pair |
| **Local perturbation** | Mutates one latent region while retaining surrounding context |

These are latent-space operations. They are not RGB cross-fades between photographs.

## Semantic spatial representation

<table>
  <tr>
    <td width="50%"><img src="media/diagrams/diagram_03_elements_to_behaviors.png" alt="Street elements and spatial behaviors"></td>
    <td width="50%"><img src="media/diagrams/diagram_04_semantic_color.png" alt="Semantic color as spatial data"></td>
  </tr>
  <tr>
    <td><strong>Street elements → spatial tendencies</strong><br>Different local scenes clarify how facade, window, signboard, vegetation, person, and vehicle fields contribute distinct spatial behaviors.</td>
    <td><strong>Color = semantic data</strong><br>A fixed palette makes probability fields and model states comparable without implying generated texture or material.</td>
  </tr>
</table>

The six semantic layers remain independent because Taiwanese streetscape elements frequently overlap—for example, windows with grilles, facades with signs, or balconies with railings.

| Channel | Meaning in the spatial representation |
|---|---|
| Facade | envelope, mass, boundary, continuous frontage |
| Window | opening, repetition, perforation, facade rhythm |
| Signboard | attached frontal marker and commercial layer |
| Vegetation | organic interruption, clusters, soft boundaries |
| Person | temporary occupation and human scale |
| Vehicle | mobile ground occupation and street-level density |

## Sixteen-state latent journey

<p align="center">
  <img src="media/final_16state_grid.png" alt="Taiwan.zip sixteen-state latent transformation grid" width="100%">
</p>

The state sequence records a continuous spatial journey rather than sixteen disconnected image effects. Model-derived depth and semantic fields are decoded into point-cloud configurations using a consistent camera and semantic color system.

## Repository structure

| Path | Role |
|---|---|
| `scripts/depth/` | Depth Anything V2 preprocessing and image-to-point-cloud tools |
| `scripts/parsing/` | frozen multi-label Taiwan streetscape parser |
| `scripts/corpus/` | corpus discovery, review, manifest, bulk processing, and QA |
| `scripts/dataset/` | seven-channel Spatial V2 dataset builder and validator |
| `scripts/training/` | final spatial-latent VAE V2 trainer |
| `scripts/render/` | checkpoint-driven latent morph and presentation renderers |
| `scripts/export/` | generated semantic/depth fields to PLY/XYZ |
| `scripts/legacy/` | preserved earlier global-latent and bundle-fallback programs |
| `notebooks/` | Colab presentation/training notebook |
| `configs/` | channel policy and dataset schema/summary |
| `media/` | project-generated representative media and diagrams |
| `metadata/` | compact recorded run metadata |
| `docs/` | technical context and phase reports |
| `inventories/` | source selection and exclusion provenance |

## Reproduction tutorial

The commands below assume the current directory is the repository root. The complete pipeline is modular: you can begin from raw images, from an already parsed `dataset/`, or directly from the Spatial V2 bundle and checkpoint.

### 1. Install the environment

Python 3.10–3.12 is recommended. A CUDA-capable PyTorch installation is strongly recommended for depth inference, semantic parsing, training, and final rendering.

```bash
git clone <your-repository-url> Taiwan.zip
cd Taiwan.zip
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux, macOS, or Colab:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install [FFmpeg](https://ffmpeg.org/) separately and confirm that this succeeds:

```bash
ffmpeg -version
```

> PyTorch and CUDA versions must match the host GPU driver. If necessary, install the appropriate PyTorch build first and then install the remaining requirements.

### 2. Prepare external artifacts

Large data and model files are deliberately excluded from Git. A practical local layout is:

```text
Taiwan.zip/
├─ artifacts/
│  ├─ taiwan_zip_spatial_v2_bundle.zip
│  └─ taiwan_zip_spatial_v2_best.pt
├─ scripts/depth/checkpoints/
│  └─ depth_anything_v2_vitb.pth
└─ dataset/
   ├─ corpus_manifest.csv
   └─ scenes/
      └─ scene_000001/
         ├─ rgb.jpg
         ├─ depth_raw.npy
         ├─ depth_norm.npy
         └─ parsing_v1/
            └─ semantic_masks.npz
```

The final training representation has seven aligned channels at 256 × 256:

```text
depth_norm + facade + window + signboard + vegetation + person + vehicle
```

If you already have the Spatial V2 bundle and checkpoint, skip directly to step 6 or 7.

### 3. Build a canonical scene corpus from images

Import a folder of source images without duplicating exact matches:

```bash
python scripts/corpus/corpus_ingest.py path/to/source_images \
  --dataset-root dataset \
  --source-type folder \
  --source-dataset your_source_name
```

Before a long run, inspect the inventory and depth queue:

```bash
python scripts/depth/corpus_depth.py \
  --dataset-root dataset \
  --encoder vitb \
  --checkpoint scripts/depth/checkpoints/depth_anything_v2_vitb.pth \
  --dry-run
```

Run resumable relative-depth inference:

```bash
python scripts/depth/corpus_depth.py \
  --dataset-root dataset \
  --encoder vitb \
  --checkpoint scripts/depth/checkpoints/depth_anything_v2_vitb.pth
```

The worker skips complete scenes unless `--force` is supplied. Depth values are relative: `0 = farther`, `1 = nearer`.

### 4. Run the frozen semantic parser

First inspect the parsing queue:

```bash
python scripts/corpus/corpus_pipeline.py \
  --dataset-root dataset \
  --parser scripts/parsing/street_parser_v1.py \
  --inventory-only
```

Then run the resumable parser pipeline:

```bash
python scripts/corpus/corpus_pipeline.py \
  --dataset-root dataset \
  --parser scripts/parsing/street_parser_v1.py
```

The parser uses local Hugging Face model caches and writes independent multi-label masks under each scene's `parsing_v1/` directory. GPU memory is managed model-by-model; avoid running another large GPU process simultaneously.

### 5. Build the Spatial V2 training bundle

The frozen project dataset uses scenes 1–357, three square views per scene, and a fixed seed:

```bash
python scripts/dataset/build_taiwan_zip_spatial_v2.py \
  --dataset-root dataset \
  --manifest dataset/corpus_manifest.csv \
  --output phase_2b_spatial_v2/training_taiwan_zip_spatial_v2 \
  --scene-start 1 \
  --scene-end 357 \
  --size 256 \
  --seed 42
```

Expected bundle:

```text
phase_2b_spatial_v2/taiwan_zip_spatial_v2_bundle.zip
```

It contains `train.npz`, `val.npz`, `test.npz`, `regression.npz`, the manifest, and the dataset summary.

### 6. Train Spatial VAE V2

Local or Colab example:

```bash
python scripts/training/taiwan_zip_spatial_vae_v2.py \
  --data-zip phase_2b_spatial_v2/taiwan_zip_spatial_v2_bundle.zip \
  --output results_spatial_v2 \
  --work-dir work/spatial_v2 \
  --checkpoint-dir work/spatial_v2_checkpoints \
  --epochs 100 \
  --batch-size 16 \
  --latent-channels 16 \
  --base-channels 32 \
  --seed 42
```

The trainer uses early stopping and copies the best checkpoint to:

```text
results_spatial_v2/taiwan_zip_spatial_v2_best.pt
```

It also exports reconstruction, interpolation, extrapolation, and local-perturbation diagnostics. Reduce `--batch-size` if GPU memory is insufficient.

### 7. Render a compact latent morph

Interpolation through and beyond two regression anchors:

```bash
python scripts/render/render_taiwan_zip_morph.py \
  --checkpoint results_spatial_v2/taiwan_zip_spatial_v2_best.pt \
  --data-zip phase_2b_spatial_v2/taiwan_zip_spatial_v2_bundle.zip \
  --output outputs/morph_interpolation.mp4 \
  --mode interpolation \
  --index-a 0 \
  --index-b 1 \
  --t-start 0.0 \
  --t-end 1.5 \
  --frames 91 \
  --fps 30
```

For a local mutation, use `--mode local_perturbation` with `--sigma-end` and `--mutation-radius`. The renderer writes an MP4 plus a same-name JSON metadata file.

### 8. Prepare the real-street opening for the final journey

The V6.3 final renderer expects one presentation-ready local source bundle containing:

```text
local_intro/
├─ rgb.jpg
├─ street46.ply
└─ street46_depth_official.png
```

Choose one real source image, copy it twice as `local_intro/rgb.jpg` and `local_intro/street46.jpg`, then run:

```bash
python scripts/depth/img2pointclouds.py \
  --img-path local_intro/street46.jpg \
  --outdir local_intro \
  --encoder vitb \
  --stride 4
```

This creates `street46.ply` and `street46_depth_official.png`. The second image copy can then be removed; `rgb.jpg` remains the human-view opening frame.

### 9. Render the final 16-state journey

```bash
python scripts/render/render_taiwan_zip_v6_3_model_fix.py \
  --checkpoint results_spatial_v2/taiwan_zip_spatial_v2_best.pt \
  --data-zip phase_2b_spatial_v2/taiwan_zip_spatial_v2_bundle.zip \
  --dataset-dir local_intro \
  --output-dir outputs/taiwan_zip_final \
  --fps 20 \
  --seconds 50 \
  --width 1280 \
  --height 720 \
  --num-states 16 \
  --anchor-indices 0,6,12,18,24,31 \
  --seed 42
```

Expected outputs include:

```text
outputs/taiwan_zip_final/
├─ taiwan_zip_v6_3_MODEL_FINAL.mp4
├─ render_manifest_MODEL_FINAL.json
├─ hero_main_visual.png
└─ stills/
```

All 16 main states are encoded from regression samples, transformed in spatial latent space, and decoded through the trained checkpoint.

### 10. Export a generated state to PLY / Rhino

The trainer's generated folders contain paired depth and semantic files. Export one pair with:

```bash
python scripts/export/taiwan_zip_generated_to_ply.py \
  --depth path/to/generated_depth.npy \
  --semantics path/to/generated_semantics.npz \
  --output outputs/generated_pointcloud \
  --width 500 \
  --depth-strength 300 \
  --threshold 0.50 \
  --stride 1
```

The exporter writes category-specific and combined point clouds using the documented Rhino coordinate convention.

### 11. Verify reproducibility

For a comparable run, record and preserve:

- dataset summary and manifest
- checkpoint epoch and validation loss
- random seed
- anchor indices
- latent and base channel counts
- rendering parameters
- generated JSON manifests

The reference configuration uses seed `42`, seven input/output channels, a spatial latent shape of `16 × 16 × 16`, and 16 presentation states. Exact pixel output can still vary with PyTorch, CUDA, driver, and FFmpeg versions.

### Common problems

| Problem | Check |
|---|---|
| CUDA out of memory | reduce training batch size; do not keep parser models and the VAE loaded together |
| Hugging Face connection error | confirm required models exist in local cache; then use offline environment flags |
| `ffmpeg` not found | install FFmpeg and add it to `PATH` |
| checkpoint key mismatch | use the Spatial VAE V2 checkpoint with the V2 renderer, not the legacy global VAE |
| `regression.npz not found` | confirm the supplied ZIP is the Spatial V2 bundle produced in step 5 |
| final renderer cannot find local files | verify `rgb.jpg`, `street46.ply`, and `street46_depth_official.png` are together in `--dataset-dir` |

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
- local `dataset/scenes/` when running the complete pipeline

Keep large artifacts in release storage, cloud drive, or another dataset/model registry rather than Git history.

## Technical conventions

- Training channels: `depth_norm`, `facade`, `window`, `signboard`, `vegetation`, `person`, `vehicle`
- Relative depth: `0 = farther`, `1 = nearer`; it is not metric depth
- Semantics are independent multi-label fields
- Rhino convention: `+X = image right`, `+Y = forward/depth`, `+Z = up`

## Image provenance

Every image embedded in this README is an existing Taiwan.zip project output. The README uses the project’s recorded hero image, 16-state grid, and four diagrams rendered from local RGB, relative-depth, semantic-mask, generated-field, and point-cloud data. No new illustrative or externally sourced image was generated for this README.

See [`docs/PLAYING_MODELS_CONTEXT.md`](docs/PLAYING_MODELS_CONTEXT.md) and [`GITHUB_INTEGRATION_REPORT.md`](GITHUB_INTEGRATION_REPORT.md) before reproducing the complete workflow.
