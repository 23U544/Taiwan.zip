"""Shared corpus inventory, provenance, validation, and split primitives."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CORPUS_VERSION = "playing-models-corpus-v1"
PARSER_VERSION = "playing-models-street-parser-v1.0"
TRAINING_VERSION = "playing-models-training-v1"
SEMANTIC_TAGS = [
    "facade", "window", "door", "balcony", "railing", "grille", "awning",
    "storefront", "rolling_shutter", "signboard", "air_conditioner",
    "utility_pole", "wire", "vegetation", "person", "vehicle",
    "street_object", "arcade_candidate",
]
EXPERIMENTAL_TAGS = {"grille", "wire", "arcade_candidate"}
REGRESSION_SCENES = {
    "scene_000001", "scene_000002", "scene_000010", "scene_000027",
    "scene_000037", "scene_000044", "scene_000046", "scene_000049",
}
MANIFEST_FIELDS = [
    "corpus_version", "scene_id", "source_type", "source_dataset", "source_id",
    "source_group_id", "source_reference", "source_filename", "license",
    "license_url", "attribution", "creator", "city", "search_mode", "image_url",
    "source_image_hash", "ingest_timestamp", "original_width", "original_height",
    "image_hash", "perceptual_hash", "depth_status", "parsing_status",
    "qa_status", "qa_visual_status", "parser_version", "pipeline_status",
    "split_group", "split", "notes",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def perceptual_hash(path: Path) -> str:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    dct = cv2.dct(gray)
    low = dct[:8, :8]
    median = np.median(low.ravel()[1:])
    bits = (low > median).ravel()
    return f"{int(''.join('1' if value else '0' for value in bits), 2):016x}"


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["scene_id"]):
            writer.writerow({key: row.get(key, "") for key in MANIFEST_FIELDS})


def validate_scene(scene_dir: Path, require_parsing: bool = False) -> dict:
    errors, warnings = [], []
    rgb_path = scene_dir / "rgb.jpg"
    raw_path, norm_path = scene_dir / "depth_raw.npy", scene_dir / "depth_norm.npy"
    shape = None
    try:
        with Image.open(rgb_path) as image:
            image.verify()
        with Image.open(rgb_path) as image:
            shape = (image.height, image.width)
    except Exception as exc:
        errors.append(f"rgb_unreadable: {exc!r}")
    depths = {}
    for name, path in (("depth_raw", raw_path), ("depth_norm", norm_path)):
        try:
            array = np.load(path, mmap_mode="r")
            depths[name] = array
            if shape and array.shape != shape: errors.append(f"{name}_shape:{array.shape}!={shape}")
            if not np.isfinite(array).all(): errors.append(f"{name}_non_finite")
        except Exception as exc:
            errors.append(f"{name}_invalid: {exc!r}")
    parsing = scene_dir / "parsing_v1"
    parser_version = ""
    parsing_complete = False
    instance_count = None
    if parsing.is_dir():
        required = ["semantic_masks.npz", "instance_masks.npz", "instances.json", "relationships.json", "scene_features.json", "parser_metadata.json"]
        for name in required:
            if not (parsing / name).is_file(): errors.append(f"missing_parsing_file:{name}")
        try:
            meta = read_json(parsing / "parser_metadata.json")
            parser_version = str(meta.get("parser_version", ""))
            if parser_version != PARSER_VERSION: errors.append(f"parser_version:{parser_version}")
            if meta.get("completion_status") != "complete": errors.append("parser_not_complete")
            if meta.get("semantic_tags") != SEMANTIC_TAGS: errors.append("semantic_tag_schema_mismatch")
        except Exception as exc: errors.append(f"parser_metadata_invalid:{exc!r}")
        try:
            with np.load(parsing / "semantic_masks.npz") as masks:
                if list(masks.files) != SEMANTIC_TAGS: errors.append("semantic_npz_keys_mismatch")
                for key in masks.files:
                    value = masks[key]
                    if shape and value.shape != shape: errors.append(f"semantic_shape:{key}:{value.shape}")
                    if not np.isin(value, (0, 1)).all(): errors.append(f"semantic_non_binary:{key}")
        except Exception as exc: errors.append(f"semantic_masks_invalid:{exc!r}")
        try:
            instances = read_json(parsing / "instances.json")
            relationships = read_json(parsing / "relationships.json")
            features = read_json(parsing / "scene_features.json")
            if not isinstance(instances, list): errors.append("instances_not_list")
            if not isinstance(relationships.get("relationships"), list): errors.append("relationships_not_list")
            if features.get("scene_id") != scene_dir.name: errors.append("scene_features_id_mismatch")
            instance_count = len(instances)
            with np.load(parsing / "instance_masks.npz") as masks:
                stack, ids = masks["masks"], masks["instance_ids"]
                if stack.ndim != 3 or (shape and tuple(stack.shape[1:]) != shape): errors.append(f"instance_mask_shape:{stack.shape}")
                if stack.shape[0] != len(instances) or len(ids) != len(instances): errors.append("instance_count_mismatch")
                if not np.isin(stack, (0, 1)).all(): errors.append("instance_masks_non_binary")
                for index, item in enumerate(instances):
                    if item.get("mask_index") != index or str(ids[index]) != item.get("instance_id"):
                        errors.append(f"instance_index_mismatch:{index}"); break
            parsing_complete = not any("parsing" in value or "semantic" in value or "instance" in value or "parser" in value or "relationships" in value or "scene_features" in value for value in errors)
        except Exception as exc: errors.append(f"parsing_json_or_instances_invalid:{exc!r}")
    elif require_parsing:
        errors.append("parsing_missing")
    return {
        "scene_id": scene_dir.name, "valid": not errors, "errors": errors, "warnings": warnings,
        "rgb_shape": list(shape) if shape else None,
        "depth_complete": shape is not None and set(depths) == {"depth_raw", "depth_norm"} and not any(value.startswith("depth_") for value in errors),
        "parsing_present": parsing.is_dir(), "parsing_complete": parsing_complete,
        "parser_version": parser_version, "instance_count": instance_count,
    }


def infer_pipeline_status(validation: dict) -> str:
    if validation["errors"] and validation["parsing_present"]: return "FAILED"
    if not validation["rgb_shape"]: return "FAILED"
    if not validation["depth_complete"]: return "DEPTH_PENDING"
    if not validation["parsing_present"]: return "PARSING_PENDING"
    if not validation["parsing_complete"]: return "FAILED"
    return "READY"


def scan_corpus(dataset_root: Path, previous: list[dict] | None = None) -> tuple[list[dict], list[dict]]:
    previous_by_id = {row["scene_id"]: row for row in (previous or [])}
    rows, validations = [], []
    for scene in sorted(path for path in (dataset_root / "scenes").glob("scene_*") if path.is_dir()):
        validation = validate_scene(scene)
        validations.append(validation)
        prior = previous_by_id.get(scene.name, {})
        rgb = scene / "rgb.jpg"
        width = validation["rgb_shape"][1] if validation["rgb_shape"] else ""
        height = validation["rgb_shape"][0] if validation["rgb_shape"] else ""
        metadata = {}
        try: metadata = read_json(scene / "metadata.json")
        except Exception: pass
        source = metadata.get("source_reference") or metadata.get("source_page") or metadata.get("source_image") or metadata.get("source_path") or metadata.get("source") or ""
        source_path = Path(source)
        if source and not source_path.is_absolute(): source_path = dataset_root.parent / source_path
        source_image_hash = prior.get("source_image_hash") or metadata.get("source_image_hash") or (sha256_file(source_path) if source_path.is_file() else "")
        row = {
            "corpus_version": CORPUS_VERSION, "scene_id": scene.name,
            "source_type": prior.get("source_type") or metadata.get("source_type") or "existing_local",
            "source_dataset": prior.get("source_dataset") or metadata.get("source_dataset") or "",
            "source_id": prior.get("source_id") or metadata.get("source_id") or "",
            "source_group_id": prior.get("source_group_id") or metadata.get("source_group_id") or scene.name,
            "source_reference": prior.get("source_reference") or source,
            "source_filename": prior.get("source_filename") or metadata.get("source_filename") or Path(source).name,
            "license": prior.get("license") or metadata.get("license") or "",
            "license_url": prior.get("license_url") or metadata.get("license_url") or "",
            "attribution": prior.get("attribution") or metadata.get("attribution") or "",
            "creator": prior.get("creator") or metadata.get("creator") or "",
            "city": prior.get("city") or metadata.get("city") or "",
            "search_mode": prior.get("search_mode") or metadata.get("search_mode") or "",
            "image_url": prior.get("image_url") or metadata.get("image_url") or "",
            "source_image_hash": source_image_hash,
            "ingest_timestamp": prior.get("ingest_timestamp") or utc_now(),
            "original_width": width, "original_height": height,
            "image_hash": sha256_file(rgb) if rgb.is_file() else "",
            "perceptual_hash": perceptual_hash(rgb) if rgb.is_file() else "",
            "depth_status": "complete" if validation["depth_complete"] else "pending",
            "parsing_status": "complete" if validation["parsing_complete"] else ("incomplete" if validation["parsing_present"] else "pending"),
            "qa_status": "pass" if validation["valid"] and validation["parsing_complete"] else ("pending" if not validation["parsing_present"] else "failed"),
            "qa_visual_status": prior.get("qa_visual_status") or "unchecked",
            "parser_version": validation["parser_version"], "pipeline_status": infer_pipeline_status(validation),
            "split_group": prior.get("split_group") or prior.get("source_group_id") or scene.name,
            "split": prior.get("split") or ("regression" if scene.name in REGRESSION_SCENES else ""),
            "notes": prior.get("notes") or metadata.get("notes") or "legacy source grouping unknown; singleton split_group is provisional",
        }
        rows.append(row)
    return rows, validations


def write_duplicate_report(rows: list[dict], path: Path, near_threshold: int = 8) -> list[dict]:
    report = []
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            exact = bool(left["image_hash"] and left["image_hash"] == right["image_hash"])
            distance = hamming_hex(left["perceptual_hash"], right["perceptual_hash"])
            if exact or distance <= near_threshold:
                report.append({"scene_id_a":left["scene_id"], "scene_id_b":right["scene_id"], "exact_duplicate":str(exact).lower(), "phash_distance":distance, "status":"review"})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["scene_id_a", "scene_id_b", "exact_duplicate", "phash_distance", "status"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(report)
    return report
