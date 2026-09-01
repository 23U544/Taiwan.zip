"""Idempotent image-folder ingestion into the persistent Playing Models corpus."""
import argparse
import json
from pathlib import Path
from PIL import Image
from corpus_core import load_manifest, scan_corpus, sha256_file, utc_now, write_duplicate_report, write_manifest

p = argparse.ArgumentParser()
p.add_argument("input", nargs="?", help="Folder containing source images; omit with --refresh-only")
p.add_argument("--dataset-root", default="dataset")
p.add_argument("--source-type", default="folder")
p.add_argument("--source-dataset", default="")
p.add_argument("--source-group-id", default="", help="Shared group for all ingested files; default is one group per file")
p.add_argument("--license", default="")
p.add_argument("--attribution", default="")
p.add_argument("--refresh-only", action="store_true")
p.add_argument("--dry-run", action="store_true")
args = p.parse_args()
root = Path(args.dataset_root)
manifest_path = root / "corpus_manifest.csv"
previous = load_manifest(manifest_path)
existing_hashes = {row.get("image_hash"): row["scene_id"] for row in previous if row.get("image_hash")}
next_id = max([int(row["scene_id"].split("_")[-1]) for row in previous] + [int(path.name.split("_")[-1]) for path in (root/"scenes").glob("scene_*")], default=0) + 1
created, duplicates, invalid = [], [], []
if not args.refresh_only:
    if not args.input: p.error("input is required unless --refresh-only is used")
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    for source in sorted(path for path in Path(args.input).rglob("*") if path.suffix.lower() in extensions):
        try:
            with Image.open(source) as image: image.verify()
            digest = sha256_file(source)
            if digest in existing_hashes:
                duplicates.append({"source":str(source), "existing_scene":existing_hashes[digest]}); continue
            scene_id = f"scene_{next_id:06d}"; next_id += 1
            if not args.dry_run:
                scene = root / "scenes" / scene_id; scene.mkdir(parents=True)
                with Image.open(source) as image: image.convert("RGB").save(scene / "rgb.jpg", quality=95)
                metadata = {"scene_id":scene_id, "source_image":str(source), "source_type":args.source_type, "source_dataset":args.source_dataset, "source_group_id":args.source_group_id or scene_id, "license":args.license or None, "attribution":args.attribution or None, "ingest_timestamp":utc_now()}
                (scene / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            created.append({"scene_id":scene_id, "source":str(source)}); existing_hashes[digest] = scene_id
        except Exception as exc: invalid.append({"source":str(source), "error":repr(exc)})
if not args.dry_run:
    rows, _ = scan_corpus(root, load_manifest(manifest_path))
    write_manifest(manifest_path, rows)
    write_duplicate_report(rows, root / "duplicate_report.csv")
print(json.dumps({"created":created, "exact_duplicates_skipped":duplicates, "invalid":invalid, "dry_run":args.dry_run}, indent=2, ensure_ascii=False))
