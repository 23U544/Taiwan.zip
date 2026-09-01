#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taiwan.zip — Generated Depth + Semantics -> Rhino-ready Point Cloud

Purpose
-------
Convert generated outputs from Taiwan.zip Spatial VAE V2:

    *_depth.npy
    *_semantics.npz

into:

    combined_semantic.ply
    combined_semantic.xyz
    layers/
        facade.ply / .xyz
        window.ply / .xyz
        signboard.ply / .xyz
        vegetation.ply / .xyz
        person.ply / .xyz
        vehicle.ply / .xyz

Coordinate convention
---------------------
The generated image is treated as a pseudo-3D spatial field, not metric geometry.

Camera-like axes:
    X_cam = horizontal image axis
    Y_cam = vertical image axis (positive upward)
    Z_cam = pseudo-depth

Rhino convention:
    X_rhino = X_cam
    Y_rhino = Z_cam
    Z_rhino = Y_cam

This keeps the project convention:
    X_rhino = X_cam
    Y_rhino = Z_cam
    Z_rhino = Y_cam

Depth definition:
    depth_norm = 0 -> farther
    depth_norm = 1 -> nearer

Pseudo-depth:
    Y_rhino = (1 - depth_norm) * depth_strength

Therefore near points stay near Y=0, and farther points extend deeper in +Y.

Example
-------
D:\\Miniconda3\\python.exe taiwan_zip_generated_to_ply.py ^
  --depth "04_depth.npy" ^
  --semantics "04_semantics.npz" ^
  --output "taiwan_unzip_t1_5" ^
  --width 500 ^
  --depth-strength 350 ^
  --threshold 0.50 ^
  --stride 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEMANTIC_CHANNELS = [
    "facade",
    "window",
    "signboard",
    "vegetation",
    "person",
    "vehicle",
]

# RGB colors only for visualization/export.
COLORS = {
    "facade":     (205, 205, 205),
    "window":     (70, 145, 235),
    "signboard":  (255, 125, 35),
    "vegetation": (55, 200, 90),
    "person":     (240, 70, 170),
    "vehicle":    (245, 205, 45),
    "unlabeled":  (75, 75, 75),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", required=True, help="Generated *_depth.npy")
    p.add_argument("--semantics", required=True, help="Generated *_semantics.npz")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--width", type=float, default=500.0,
                   help="Rhino X width in arbitrary units.")
    p.add_argument("--depth-strength", type=float, default=300.0,
                   help="Maximum pseudo-depth along Rhino +Y.")
    p.add_argument("--threshold", type=float, default=0.50,
                   help="Semantic probability threshold for layer PLYs.")
    p.add_argument("--combined-min-prob", type=float, default=0.20,
                   help="Minimum dominant semantic probability to color; otherwise unlabeled.")
    p.add_argument("--stride", type=int, default=1,
                   help="Pixel sampling stride. 1 keeps every point, 2 keeps 1/4, etc.")
    p.add_argument("--min-depth", type=float, default=0.0,
                   help="Ignore points with depth_norm below this.")
    p.add_argument("--max-depth", type=float, default=1.0,
                   help="Ignore points with depth_norm above this.")
    return p.parse_args()


def load_data(depth_path: Path, semantics_path: Path):
    depth = np.asarray(np.load(depth_path), dtype=np.float32)
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"Depth must be HxW. Got: {depth.shape}")

    with np.load(semantics_path) as z:
        missing = [k for k in SEMANTIC_CHANNELS if k not in z.files]
        if missing:
            raise KeyError(
                f"Missing semantic channels: {missing}. "
                f"Available: {z.files}"
            )
        semantics = np.stack(
            [np.asarray(z[k], dtype=np.float32) for k in SEMANTIC_CHANNELS],
            axis=0
        )

    if semantics.ndim != 3:
        raise ValueError(f"Semantics must be [6,H,W]. Got: {semantics.shape}")

    if semantics.shape[1:] != depth.shape:
        raise ValueError(
            f"Shape mismatch: depth={depth.shape}, semantics={semantics.shape}"
        )

    if not np.isfinite(depth).all():
        raise ValueError("Depth contains NaN/Inf.")
    if not np.isfinite(semantics).all():
        raise ValueError("Semantics contain NaN/Inf.")

    depth = np.clip(depth, 0.0, 1.0)
    semantics = np.clip(semantics, 0.0, 1.0)
    return depth, semantics


def make_coordinates(depth: np.ndarray, width: float, depth_strength: float, stride: int):
    """
    Returns flattened:
        xyz: [N,3] in Rhino coordinates
        pixel_y, pixel_x: original pixel indices for each generated point
    """
    h, w = depth.shape

    ys = np.arange(0, h, stride, dtype=np.int32)
    xs = np.arange(0, w, stride, dtype=np.int32)
    xx, yy = np.meshgrid(xs, ys)

    # Keep image aspect ratio in X/Z plane.
    height = width * (h / w)

    # Horizontal axis.
    x_rhino = (xx.astype(np.float32) / max(w - 1, 1) - 0.5) * width

    # Vertical axis: positive upward.
    y_cam = (0.5 - yy.astype(np.float32) / max(h - 1, 1)) * height

    sampled_depth = depth[yy, xx]

    # depth_norm: 1 near, 0 far.
    # Rhino +Y becomes the pseudo-depth axis.
    y_rhino = (1.0 - sampled_depth) * depth_strength

    # Fixed project convention:
    # X_rhino = X_cam
    # Y_rhino = Z_cam
    # Z_rhino = Y_cam
    z_rhino = y_cam

    xyz = np.stack([x_rhino, y_rhino, z_rhino], axis=-1).reshape(-1, 3)
    return xyz, yy.reshape(-1), xx.reshape(-1), sampled_depth.reshape(-1)


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray, scalar: np.ndarray | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(xyz)

    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment Taiwan.zip generated semantic point cloud\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        if scalar is not None:
            f.write("property float probability\n")
        f.write("end_header\n")

        if scalar is None:
            for (x, y, z), (r, g, b) in zip(xyz, rgb):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        else:
            for (x, y, z), (r, g, b), p in zip(xyz, rgb, scalar):
                f.write(
                    f"{x:.6f} {y:.6f} {z:.6f} "
                    f"{int(r)} {int(g)} {int(b)} {float(p):.6f}\n"
                )


def write_xyz(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    """
    XYZRGB plain text:
        X Y Z R G B
    Easy to inspect/import into many 3D workflows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.column_stack([xyz, rgb.astype(np.float32)])
    np.savetxt(
        path,
        arr,
        fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"],
    )


def main():
    args = parse_args()

    depth_path = Path(args.depth)
    sem_path = Path(args.semantics)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    depth, semantics = load_data(depth_path, sem_path)
    xyz, py, px, sampled_depth = make_coordinates(
        depth,
        width=args.width,
        depth_strength=args.depth_strength,
        stride=max(1, args.stride),
    )

    probs = semantics[:, py, px].T  # [N,6]

    depth_valid = (
        (sampled_depth >= args.min_depth)
        & (sampled_depth <= args.max_depth)
        & np.isfinite(sampled_depth)
    )

    # ---------------------------------------------------------
    # Combined dominant-semantic cloud
    # ---------------------------------------------------------
    dominant_idx = np.argmax(probs, axis=1)
    dominant_prob = probs[np.arange(len(probs)), dominant_idx]

    combined_rgb = np.empty((len(xyz), 3), dtype=np.uint8)

    for i, channel in enumerate(SEMANTIC_CHANNELS):
        mask = dominant_idx == i
        combined_rgb[mask] = COLORS[channel]

    low_conf = dominant_prob < args.combined_min_prob
    combined_rgb[low_conf] = COLORS["unlabeled"]

    combined_mask = depth_valid

    write_ply(
        out_dir / "combined_semantic.ply",
        xyz[combined_mask],
        combined_rgb[combined_mask],
        dominant_prob[combined_mask],
    )
    write_xyz(
        out_dir / "combined_semantic.xyz",
        xyz[combined_mask],
        combined_rgb[combined_mask],
    )

    # ---------------------------------------------------------
    # Semantic layers
    # ---------------------------------------------------------
    layer_dir = out_dir / "layers"
    layer_stats = {}

    for c, channel in enumerate(SEMANTIC_CHANNELS):
        probability = probs[:, c]
        mask = depth_valid & (probability >= args.threshold)

        color = np.tile(
            np.array(COLORS[channel], dtype=np.uint8)[None, :],
            (int(mask.sum()), 1),
        )

        write_ply(
            layer_dir / f"{channel}.ply",
            xyz[mask],
            color,
            probability[mask],
        )
        write_xyz(
            layer_dir / f"{channel}.xyz",
            xyz[mask],
            color,
        )

        layer_stats[channel] = {
            "point_count": int(mask.sum()),
            "fraction_of_sampled_pixels": float(mask.mean()),
            "threshold": float(args.threshold),
            "mean_probability_selected": (
                float(probability[mask].mean()) if mask.any() else None
            ),
        }

    metadata = {
        "version": "taiwan-zip-generated-to-pointcloud-v1",
        "source_depth": str(depth_path),
        "source_semantics": str(sem_path),
        "input_shape": [int(depth.shape[0]), int(depth.shape[1])],
        "semantic_channels": SEMANTIC_CHANNELS,
        "coordinate_convention": {
            "X_rhino": "X_cam / horizontal image axis",
            "Y_rhino": "Z_cam / pseudo-depth",
            "Z_rhino": "Y_cam / vertical image axis",
        },
        "depth_definition": {
            "depth_norm_0": "farther",
            "depth_norm_1": "nearer",
            "pseudo_depth": "(1 - depth_norm) * depth_strength",
            "metric_depth": False,
        },
        "parameters": {
            "width": float(args.width),
            "depth_strength": float(args.depth_strength),
            "threshold": float(args.threshold),
            "combined_min_prob": float(args.combined_min_prob),
            "stride": int(args.stride),
            "min_depth": float(args.min_depth),
            "max_depth": float(args.max_depth),
        },
        "combined_point_count": int(combined_mask.sum()),
        "layers": layer_stats,
    }

    (out_dir / "pointcloud_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("DONE")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("OUTPUT:", out_dir.resolve())


if __name__ == "__main__":
    main()
