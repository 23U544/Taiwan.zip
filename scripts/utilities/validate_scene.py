"""Validate one scene or a corpus without running any model."""
import argparse
import json
from pathlib import Path
from corpus_core import validate_scene

p = argparse.ArgumentParser()
p.add_argument("--dataset-root", default="dataset")
p.add_argument("--scene")
p.add_argument("--require-parsing", action="store_true")
p.add_argument("--json-output")
args = p.parse_args()
root = Path(args.dataset_root) / "scenes"
scenes = [root / args.scene] if args.scene else sorted(root.glob("scene_*"))
results = [validate_scene(scene, args.require_parsing) for scene in scenes]
summary = {"scenes":len(results), "valid":sum(item["valid"] for item in results), "failed":sum(not item["valid"] for item in results), "results":results}
text = json.dumps(summary, indent=2, ensure_ascii=False)
if args.json_output: Path(args.json_output).write_text(text, encoding="utf-8")
print(text)
raise SystemExit(0 if summary["failed"] == 0 else 1)
