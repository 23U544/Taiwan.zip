# Taiwan.zip Local Package Report

Generated from `D:\GIA\3rd Up\DepthAnything\Depth-Anything-V2` without modifying the source project.

## Result

- Selected source/config/document files: 44
- Missing selected paths: 0
- Binary datasets, checkpoints, point clouds and media copied: 0

## Final workflow discovery

- `img2pointclouds.py`: found and copied to `scripts/depth/img2pointclouds.py`.
- Frozen parser: `street_parser_v1.py` found and copied to `scripts/parsing/street_parser_v1.py`.
- Corpus/manifest workflow: `corpus_core.py`, `corpus_ingest.py`, `corpus_depth.py`, and `corpus_pipeline.py` found.
- Spatial V2 dataset builder: `build_taiwan_zip_spatial_v2.py` identified and copied.
- Generated-to-PLY exporter: `taiwan_zip_generated_to_ply.py` identified and copied.
- Latest final renderer: `render_taiwan_zip_v6_3_final.py` identified and copied.

## Trainer caveat requiring manual review

`taiwan_zip_colab_train.py` contains the complete seven-channel `TaiwanZipSpatialVAE` model, training loop, checkpoint writing, reconstruction, interpolation, perturbation, and extrapolation. However, its header and internal extraction path still name the prototype bundle and it writes `taiwan_zip_best.pt`, not `taiwan_zip_spatial_v2_best.pt`. It is therefore included as the closest recovered trainer candidate, but cannot be proven from the current filesystem to be the exact final Spatial V2 trainer.

The requested `results_spatial_v2/taiwan_zip_spatial_v2_best.pt` was not present locally during packaging. Checkpoint binaries would be excluded from this GitHub source pack even if present.

## Dependencies

The full project-local Python portion of `depth_anything_v2/` was copied because `img2pointclouds.py`, `build_dataset.py`, and `corpus_depth.py` import it. `corpus_core.py` was copied because the corpus workers import it. Other imports listed in `inventories/python_imports.txt` are standard-library or third-party packages and were not vendored.

## Excluded development history

The frozen parser replaces earlier detector experiments. Representative excluded duplicates/obsolete versions:

architectural_elements_v2.py, architectural_elements_v3.py, architectural_elements_v4.py, architectural_elements_v4_1.py,
window_detector_v5.py, window_detector_v5_1.py, window_detector_v5_1_fixed.py, window_detector_v6.py, window_detector_v6_1.py,
taiwan_streetscape_finder.py, taiwan_streetscape_finder_v2.py, taiwan_streetscape_finder_v2_1.py,
screen_taiwan_streetscape_candidates_v1.py, screen_taiwan_streetscape_candidates_v1_1.py

Generated datasets, 604 scene directories, masks, NPY/NPZ files, model weights, PLY files, previews, screenshots, videos, archives, and caches were excluded. Important large exclusions are recorded in `inventories/excluded_large_files.csv`.

## Manual review paths

- Confirm whether `taiwan_zip_colab_train.py` was the exact script used for the Spatial V2 checkpoint or whether a later trainer exists elsewhere.
- Restore checkpoint/run-guide artifacts separately through release storage, not this source-only GitHub archive.
- Review hard-coded original Windows/Colab paths before executing from a new checkout.

## Checklist

[x] img2pointclouds
[x] final street parser
[x] corpus / manifest
[x] Spatial V2 dataset builder
[ ] Spatial V2 trainer — candidate recovered; exact final provenance unconfirmed
[x] generated-to-PLY exporter
[x] final curation scripts
[x] local dependencies
