# GitHub Integration Report

## Sources merged

- `D:\GIA\TaiwanZip_GitHub_Local_Pack`
- `D:\GIA\FINAL_GITHUB_COLAB_PACK-20260901T084331Z-1-001\FINAL_GITHUB_COLAB_PACK`

Both source folders were preserved unchanged. Their contents were copied into `D:\GIA\Taiwan.zip`.

## Integration decisions

- The extra `FINAL_GITHUB_COLAB_PACK` wrapper directory was removed during integration.
- `scripts/training/taiwan_zip_spatial_vae_v2.py` is the final Spatial V2 trainer. It defines the spatial latent tensor, writes `taiwan_zip_spatial_v2_best.pt`, and exports interpolation, extrapolation, and local-perturbation grids.
- `scripts/render/render_taiwan_zip_v6_3_model_fix.py` is the primary checkpoint-driven final renderer.
- `scripts/render/render_taiwan_zip_morph.py` is retained as the compact latent morph renderer.
- The earlier global 128-D VAE trainer was retained as `scripts/legacy/taiwan_zip_global_vae_v1.py`.
- The dataset-bundle fallback renderer was retained as `scripts/legacy/render_taiwan_zip_v6_3_bundle_fallback.py`.
- No differing same-path conflicts occurred.

## Included presentation artifacts

- `media/hero_main_visual.png`
- `media/final_16state_grid.png`
- `metadata/run_summary.json`
- `notebooks/Present.ipynb`

The recorded run summary describes a 79-scene prototype run (`taiwan-zip-spatial-vae-v1.1`). It is retained as historical run evidence and should not be presented as the 604-scene Spatial V2 corpus summary. The current Spatial V2 dataset summary is separately stored under `configs/spatial_v2_dataset_summary.json`.

## External artifacts not included

- trained `.pt` checkpoint
- Spatial V2 NPZ dataset bundle
- 604-scene source dataset
- generated point-cloud and full video outputs
- downloaded Depth Anything / Hugging Face weights

## Validation

- Copied source files from both packs: 56
- Differing same-path conflicts: 0
- Python sources are syntax-checked after integration.
- Git-blocked data/model extensions are checked before completion.
