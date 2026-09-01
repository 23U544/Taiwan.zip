"""Resumable, shardable corpus orchestration around frozen parser V1."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from corpus_core import PARSER_VERSION, load_manifest, scan_corpus, utc_now, validate_scene, write_duplicate_report, write_manifest

p = argparse.ArgumentParser()
p.add_argument("--dataset-root", default="dataset")
p.add_argument("--python", default=sys.executable)
p.add_argument("--parser", default="street_parser_v1.py")
p.add_argument("--start", type=int)
p.add_argument("--end", type=int)
p.add_argument("--shard", type=int, help="zero-based shard index")
p.add_argument("--num-shards", type=int)
p.add_argument("--scenes", nargs="*")
p.add_argument("--force", action="store_true")
p.add_argument("--inventory-only", action="store_true")
p.add_argument("--dry-run", action="store_true")
args = p.parse_args()
root = Path(args.dataset_root).resolve(); manifest_path = root / "corpus_manifest.csv"
rows, validations = scan_corpus(root, load_manifest(manifest_path)); write_manifest(manifest_path, rows); write_duplicate_report(rows, root / "duplicate_report.csv")
selected = rows
if args.scenes: selected = [row for row in selected if row["scene_id"] in set(args.scenes)]
if args.start is not None: selected = [row for row in selected if int(row["scene_id"].split("_")[-1]) >= args.start]
if args.end is not None: selected = [row for row in selected if int(row["scene_id"].split("_")[-1]) <= args.end]
if args.shard is not None:
    if not args.num_shards or not 0 <= args.shard < args.num_shards: p.error("--shard requires valid --num-shards")
    selected = [row for index,row in enumerate(selected) if index % args.num_shards == args.shard]
queue = [row for row in selected if row["depth_status"] == "complete" and (args.force or row["parsing_status"] != "complete" or row["parser_version"] != PARSER_VERSION)]
print(json.dumps({"inventory":len(rows), "selected":len(selected), "parse_queue":len(queue), "skipped_complete":len(selected)-len(queue)}, ensure_ascii=False))
if args.inventory_only or args.dry_run: raise SystemExit(0)
failure_path = root / "pipeline_failures.jsonl"
benchmark_path = root / "parsing_benchmark.jsonl"
env = os.environ.copy(); env["HF_HUB_OFFLINE"] = "1"; env["TRANSFORMERS_OFFLINE"] = "1"
for row in queue:
    scene_id = row["scene_id"]
    command = [args.python, args.parser, "--scene", str(root/"scenes"/scene_id), "--dataset-root", str(root/"scenes")]
    started = time.perf_counter()
    result = subprocess.run(command, env=env, text=True)
    elapsed = time.perf_counter() - started
    validation = validate_scene(root/"scenes"/scene_id, require_parsing=True)
    if result.returncode or not validation["valid"]:
        prior = 0
        if failure_path.exists(): prior = sum(1 for line in failure_path.read_text(encoding="utf-8").splitlines() if f'"scene_id": "{scene_id}"' in line)
        event = {"scene_id":scene_id, "stage":"parsing", "timestamp":utc_now(), "exception":validation["errors"] or f"returncode={result.returncode}", "parser_version":PARSER_VERSION, "retry_count":prior+1}
        with failure_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(event, ensure_ascii=False)+"\n")
    event = {"scene_id":scene_id, "stage":"parsing", "timestamp":utc_now(), "seconds":elapsed, "parser_version":PARSER_VERSION, "returncode":result.returncode, "valid":validation["valid"]}
    with benchmark_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(event, ensure_ascii=False)+"\n")
rows, _ = scan_corpus(root, load_manifest(manifest_path)); write_manifest(manifest_path, rows)
