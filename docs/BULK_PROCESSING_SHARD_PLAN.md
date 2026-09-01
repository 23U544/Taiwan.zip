# Bulk Processing Shard Plan

## Measured local basis

- Parser V1: 41 scenes in 41.3 minutes; **60.4 s/scene average** on RTX 4050 6 GB.
- Deep QA: approximately **3.0 s/scene** from the 49-scene full-mask audit/validation runs.
- Existing depth metadata write intervals: approximately **0.45 s/scene median-to-mean range**. This is a filesystem-timestamp proxy from the prior batch, not a controlled fresh benchmark; future `corpus_depth.py` records explicit per-scene timing in `depth_benchmark.jsonl`.
- Canonical storage: **16.48 MB/scene average**.

| New scenes | Depth estimate | Parsing estimate | QA estimate | Total wall-time estimate | Storage |
|---:|---:|---:|---:|---:|---:|
| 500 | 3.8 min | 8.39 h | 25 min | 8.87 h | 8.04 GB |
| 750 | 5.7 min | 12.58 h | 37.5 min | 13.30 h | 12.07 GB |
| 1000 | 7.5 min | 16.78 h | 50 min | 17.74 h | 16.09 GB |

These are planning estimates; 4K images such as current scene 000025 are materially slower and should be distributed across shards.

## Recommended placement

- Local: approval, provenance reconciliation, ingestion, duplicate review, manifest merge, deep QA, visual review, statistics, export, and Rhino-facing work.
- Local or Colab: Depth Anything inference. It is fast locally, so transfer overhead may exceed compute unless images are already on persistent cloud storage.
- Colab/persistent GPU worker: bulk Parser V1 inference. This dominates runtime and benefits most from multiple independent sessions.
- Colab later: training, outside Phase 1.6A.

## Shards

- 500 scenes: 5 shards × approximately 100 scenes.
- 750 scenes: 8 shards (seven × 94, one × 92) or a simple 10-shard assignment.
- 1000 scenes: 10 shards × approximately 100 scenes.

Use `--shard N --num-shards K`; shard numbering is zero-based and deterministic over manifest order. Each shard receives a frozen input manifest snapshot, corpus/parser version, source files or persistent paths, model/cache declaration, and checksum list.

Each completed scene must return RGB reference, both depth arrays, depth preview/metadata, complete `parsing_v1`, and per-scene logs. The worker validates before marking complete and synchronizes immediately to persistent storage. Rerunning the same shard skips valid version-matched scenes and retries missing/failed ones. `/content` is scratch only.

After all workers finish, local reconciliation recomputes hashes/status, rejects conflicting duplicate scene IDs, performs deep QA, checks that every source group remains intact, and writes the single canonical manifest. Never merge two workers by accepting whichever manifest was written last.
