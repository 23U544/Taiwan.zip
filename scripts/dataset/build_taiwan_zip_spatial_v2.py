#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taiwan.zip Spatial V2 Dataset Builder

目的：
- 冻結目前 parser 已完成的 scene snapshot
- 只使用 center / lead / trail 三種 square crops
- 建立 7-channel dataset:
    depth_norm + facade + window + signboard + vegetation + person + vehicle
- 產出 train / val / test / regression splits
- 壓成 zip 給 Colab 訓練 Spatial VAE

預設建議：
- scene-start = 1
- scene-end   = 357
- size        = 256
- views       = center, lead, trail

執行範例：
D:\\Miniconda3\\python.exe build_taiwan_zip_spatial_v2.py ^
    --dataset-root dataset ^
    --manifest dataset\\corpus_manifest.csv ^
    --output phase_2b_spatial_v2\\training_taiwan_zip_spatial_v2 ^
    --scene-start 1 ^
    --scene-end 357 ^
    --size 256 ^
    --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

CHANNELS = [
    "depth_norm",
    "facade",
    "window",
    "signboard",
    "vegetation",
    "person",
    "vehicle",
]
SEMANTIC_CHANNELS = CHANNELS[1:]
VIEWS = ["center", "lead", "trail"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="dataset")
    p.add_argument("--manifest", default="dataset/corpus_manifest.csv")
    p.add_argument("--output", default="phase_2b_spatial_v2/training_taiwan_zip_spatial_v2")
    p.add_argument("--scene-start", type=int, default=1)
    p.add_argument("--scene-end", type=int, default=357)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_manifest(path: Path):
    rows = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sid = (row.get("scene_id") or "").strip()
            if sid:
                rows[sid] = row
    return rows


def scene_dir(dataset_root: Path, sid: str) -> Path:
    p = dataset_root / "scenes" / sid
    if p.exists():
        return p
    p2 = dataset_root / sid
    return p2


def is_parser_complete(sd: Path) -> bool:
    return (
        (sd / "depth_norm.npy").exists()
        and (sd / "parsing_v1" / "semantic_masks.npz").exists()
        and (sd / "parsing_v1" / "parser_metadata.json").exists()
    )


def group_id(row: dict | None, sid: str) -> str:
    if row:
        for k in ("source_group_id", "group_id", "source_group"):
            v = (row.get(k) or "").strip()
            if v:
                return v
    return sid


def resolve_semantic(npz_obj, key: str):
    if key in npz_obj.files:
        return np.asarray(npz_obj[key])

    aliases = {
        "facade": ["facade_mask", "building_facade"],
        "window": ["windows", "window_mask"],
        "signboard": ["sign", "signage", "signboard_mask"],
        "vegetation": ["plant", "plants", "vegetation_mask"],
        "person": ["people", "pedestrian", "person_mask"],
        "vehicle": ["vehicles", "car", "vehicle_mask"],
    }

    lower_map = {k.lower(): k for k in npz_obj.files}
    for candidate in [key] + aliases.get(key, []):
        if candidate in npz_obj.files:
            return np.asarray(npz_obj[candidate])
        if candidate.lower() in lower_map:
            return np.asarray(npz_obj[lower_map[candidate.lower()]])

    raise KeyError(f"Missing semantic '{key}'. Available keys: {npz_obj.files}")


def ensure_hw(a, name="array"):
    a = np.squeeze(np.asarray(a))
    if a.ndim != 2:
        raise ValueError(f"{name}: expected HxW, got shape={a.shape}")
    return a


def resize_float(a, wh):
    return np.asarray(
        Image.fromarray(a.astype(np.float32), mode="F").resize(wh, Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


def resize_mask(a, wh):
    img = Image.fromarray((a > 0).astype(np.uint8) * 255, mode="L")
    img = img.resize(wh, Image.Resampling.NEAREST)
    return (np.asarray(img) > 127).astype(np.uint8)


def square_crop(a, mode: str):
    h, w = a.shape
    s = min(h, w)
    if w >= h:
        y = 0
        if mode == "center":
            x = (w - s) // 2
        elif mode == "lead":
            x = 0
        elif mode == "trail":
            x = w - s
        else:
            raise ValueError(mode)
    else:
        x = 0
        if mode == "center":
            y = (h - s) // 2
        elif mode == "lead":
            y = 0
        elif mode == "trail":
            y = h - s
        else:
            raise ValueError(mode)
    return a[y:y+s, x:x+s]


def make_view(a, mode: str, size: int, mask: bool = False):
    cropped = square_crop(a, mode)
    if mask:
        return resize_mask(cropped, (size, size))
    return resize_float(cropped, (size, size))


def split_groups(groups, seed=42):
    uniq = sorted(set(groups))
    random.Random(seed).shuffle(uniq)
    n = len(uniq)

    n_train = max(1, round(n * 0.70))
    n_val = max(1, round(n * 0.10)) if n >= 4 else 0
    n_test = max(1, round(n * 0.10)) if n >= 5 else 0

    if n >= 6 and (n - n_train - n_val - n_test) < 1:
        n_train -= 1

    out = {}
    i = 0
    for g in uniq[i:i+n_train]:
        out[g] = "train"
    i += n_train

    for g in uniq[i:i+n_val]:
        out[g] = "val"
    i += n_val

    for g in uniq[i:i+n_test]:
        out[g] = "test"
    i += n_test

    for g in uniq[i:]:
        out[g] = "regression"

    return out


def save_split(out_dir: Path, split_name: str, items: list[dict]):
    if not items:
        return
    np.savez_compressed(
        out_dir / f"{split_name}.npz",
        depth=np.stack([x["depth"] for x in items]).astype(np.float16),
        semantics=np.stack([x["semantics"] for x in items]).astype(np.uint8),
        scene_ids=np.array([x["scene_id"] for x in items], dtype="<U32"),
        sample_ids=np.array([x["sample_id"] for x in items], dtype="<U64"),
        source_group_ids=np.array([x["source_group_id"] for x in items], dtype="<U128"),
        views=np.array([x["view"] for x in items], dtype="<U16"),
    )


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    manifest_rows = read_manifest(Path(args.manifest))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    usable_scenes = []
    skipped = []

    for i in range(args.scene_start, args.scene_end + 1):
        sid = f"scene_{i:06d}"
        sd = scene_dir(dataset_root, sid)
        if not sd.exists():
            skipped.append({"scene_id": sid, "reason": "scene_dir_missing"})
            continue
        if not is_parser_complete(sd):
            skipped.append({"scene_id": sid, "reason": "parser_not_complete"})
            continue
        row = manifest_rows.get(sid)
        gid = group_id(row, sid)
        usable_scenes.append((sid, sd, row, gid))

    if not usable_scenes:
        raise SystemExit("No parser-complete scenes found in the requested range.")

    group_split = split_groups([x[3] for x in usable_scenes], seed=args.seed)

    buckets = {"train": [], "val": [], "test": [], "regression": []}
    manifest_export_rows = []

    for idx, (sid, sd, row, gid) in enumerate(usable_scenes, start=1):
        depth = ensure_hw(np.load(sd / "depth_norm.npy"), f"{sid}:depth_norm")

        with np.load(sd / "parsing_v1" / "semantic_masks.npz") as z:
            sems = []
            for key in SEMANTIC_CHANNELS:
                arr = ensure_hw(resolve_semantic(z, key), f"{sid}:{key}")
                if arr.shape != depth.shape:
                    raise ValueError(
                        f"{sid}: semantic '{key}' shape={arr.shape} != depth shape={depth.shape}"
                    )
                sems.append(arr)

        split_name = group_split[gid]

        for view_name in VIEWS:
            depth_view = np.clip(make_view(depth, view_name, args.size, mask=False), 0.0, 1.0)
            semantic_view = np.stack(
                [make_view(m, view_name, args.size, mask=True) for m in sems],
                axis=0,
            )

            sample_id = f"{sid}__{view_name}"
            item = {
                "sample_id": sample_id,
                "scene_id": sid,
                "source_group_id": gid,
                "split": split_name,
                "view": view_name,
                "depth": depth_view,
                "semantics": semantic_view,
            }

            buckets[split_name].append(item)
            manifest_export_rows.append(item)

        print(f"[{idx:03d}/{len(usable_scenes):03d}] {sid} -> {split_name}")

    for split_name, items in buckets.items():
        save_split(out_dir, split_name, items)

    manifest_csv = out_dir / "dataset_manifest.csv"
    with manifest_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "scene_id", "source_group_id", "split", "view"])
        for item in manifest_export_rows:
            w.writerow([
                item["sample_id"],
                item["scene_id"],
                item["source_group_id"],
                item["split"],
                item["view"],
            ])

    train_sem = np.stack([x["semantics"] for x in buckets["train"]]).astype(np.float32)
    sem_frac = train_sem.mean(axis=(0, 2, 3))

    summary = {
        "version": "taiwan-zip-spatial-v2-dataset",
        "frozen_scene_range": {
            "scene_start": args.scene_start,
            "scene_end": args.scene_end,
        },
        "ready_scene_count": len(usable_scenes),
        "sample_count": len(manifest_export_rows),
        "views_per_scene": len(VIEWS),
        "views": VIEWS,
        "image_size": args.size,
        "channels": CHANNELS,
        "semantic_channels": SEMANTIC_CHANNELS,
        "split_sample_counts": {k: len(v) for k, v in buckets.items()},
        "split_scene_counts": {
            k: len(set(x["scene_id"] for x in v)) for k, v in buckets.items()
        },
        "semantic_positive_fraction_train": {
            k: float(v) for k, v in zip(SEMANTIC_CHANNELS, sem_frac)
        },
        "skipped_count": len(skipped),
        "skipped_preview": skipped[:50],
    }

    summary_path = out_dir / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    zip_path = out_dir.parent / "taiwan_zip_spatial_v2_bundle.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in out_dir.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(out_dir.parent))

    print("\nDONE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"ZIP: {zip_path.resolve()}")


if __name__ == "__main__":
    main()