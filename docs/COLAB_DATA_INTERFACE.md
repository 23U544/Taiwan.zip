# Colab Data Interface

The canonical corpus is `dataset/` on persistent storage. Colab is a compute worker; `/content` may be used only as temporary scratch space.

## Required persistent inputs

- `corpus_manifest.csv`: provenance, source grouping, state, versions, and split assignments.
- `scenes/<scene_id>/rgb.jpg`, `depth_raw.npy`, `depth_norm.npy`, and `metadata.json` for parsing workers.
- Frozen `street_parser_v1.py` plus its pinned model cache/checkpoints.
- For training workers: a versioned export containing `channel_schema.json`, `split_manifest.csv`, `samples.csv`, and `samples/*.npz`.

## Batch parsing

Mount persistent storage, install the environment, and select non-overlapping zero-based shards:

```text
python corpus_pipeline.py --dataset-root /persistent/dataset --shard 0 --num-shards 4
```

The pipeline validates existing Parser V1 outputs and skips complete scenes. It records failures in `pipeline_failures.jsonl`; rerunning the same shard retries only missing, failed, or version-mismatched scenes. Use `--force` only for an intentional rebuild. Never write the only copy of canonical output to `/content`; synchronize each completed `parsing_v1` directory and refreshed manifest to persistent storage.

## Training export

Create exports from the canonical manifest, never from a hand-picked scene list:

```text
python build_training_dataset.py --dataset-root /persistent/dataset --output /persistent/exports/training_v1 --patch-size 256
python validate_training_export.py /persistent/exports/training_v1
```

Scene/source-group split happens before patch extraction. Consumers must read channel order from `channel_schema.json`, tensor layout is `[C,H,W]`, and `samples.csv` preserves scene and crop traceability. Default V1 channels are normalized relative depth plus stable semantic channels; experimental grille, wire, and arcade channels require `--include-experimental`. RGB crops are optional through `--include-rgb`.

## Resume and version policy

- Corpus: `playing-models-corpus-v1`
- Parser: `playing-models-street-parser-v1.0`
- Training export: `playing-models-training-v1`

Do not merge outputs with mismatched parser or channel-schema versions. Shards may be processed by separate sessions, but manifest reconciliation must retain one row per stable `scene_id` and a single split for every `source_group_id`/`split_group`.
