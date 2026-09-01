#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taiwan.zip — Latent Morph Point-Cloud Renderer V1

Creates a true model-generated morph video from Taiwan.zip Spatial VAE V2.
Each frame is decoded from the learned latent space, converted to a semantic
pseudo-3D point cloud, rendered, then encoded to MP4 with ffmpeg.

Modes
-----
1) interpolation:
   z(t) = z_A * (1-t) + z_B * t
   t may extend beyond [0,1], so t=1.5 is extrapolation.

2) local_perturbation:
   z(sigma) = z_A + sigma * fixed_noise * local_mask

Depth convention
----------------
depth_norm = 0 farther, 1 nearer.
Rhino-style pseudo 3D:
X = image horizontal
Y = (1-depth_norm) * depth_strength
Z = image vertical

Example — interpolation through extrapolation
---------------------------------------------
!python render_taiwan_zip_morph.py \
  --checkpoint "/content/drive/MyDrive/ColabNotebooks/PlayingModels/TaiwanZip/results_spatial_v2/taiwan_zip_spatial_v2_best.pt" \
  --data-zip "/content/drive/MyDrive/ColabNotebooks/PlayingModels/TaiwanZip/taiwan_zip_spatial_v2_bundle.zip" \
  --output "/content/drive/MyDrive/ColabNotebooks/PlayingModels/TaiwanZip/morph_interpolation.mp4" \
  --mode interpolation \
  --index-a 0 --index-b 1 \
  --t-start 0.0 --t-end 1.5 \
  --frames 91 --fps 30 \
  --stride 2 --point-size 2.2 \
  --width 500 --depth-strength 350

Example — local mutation
------------------------
!python render_taiwan_zip_morph.py \
  --checkpoint ".../taiwan_zip_spatial_v2_best.pt" \
  --data-zip ".../taiwan_zip_spatial_v2_bundle.zip" \
  --output ".../morph_local_mutation.mp4" \
  --mode local_perturbation \
  --index-a 0 \
  --sigma-start 0.0 --sigma-end 0.65 \
  --frames 61 --fps 30
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

CHANNELS = [
    "depth_norm", "facade", "window", "signboard",
    "vegetation", "person", "vehicle",
]

# Semantic visualization palette only; not learned RGB.
COLORS = np.array([
    [0.80, 0.80, 0.80],  # facade
    [0.20, 0.55, 1.00],  # window
    [1.00, 0.50, 0.10],  # signboard
    [0.20, 0.80, 0.35],  # vegetation
    [0.95, 0.25, 0.65],  # person
    [0.95, 0.80, 0.15],  # vehicle
], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-zip", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["interpolation", "local_perturbation"], default="interpolation")
    p.add_argument("--index-a", type=int, default=0)
    p.add_argument("--index-b", type=int, default=1)
    p.add_argument("--t-start", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=1.5)
    p.add_argument("--sigma-start", type=float, default=0.0)
    p.add_argument("--sigma-end", type=float, default=0.65)
    p.add_argument("--mutation-radius", type=float, default=0.25)
    p.add_argument("--frames", type=int, default=91)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--hold-start", type=int, default=12)
    p.add_argument("--hold-end", type=int, default=18)
    p.add_argument("--latent-channels", type=int, default=16)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--width", type=float, default=500.0)
    p.add_argument("--depth-strength", type=float, default=350.0)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--semantic-min-prob", type=float, default=0.18)
    p.add_argument("--elev", type=float, default=18.0)
    p.add_argument("--azim", type=float, default=-60.0)
    p.add_argument("--camera-drift", type=float, default=8.0)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--work-dir", default="/content/taiwan_zip_morph_work")
    p.add_argument("--keep-frames", action="store_true")
    return p.parse_args()


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p),
            nn.GroupNorm(min(8, cout), cout),
            nn.SiLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(cin, cout, k=4, s=2, p=1),
            ConvBlock(cout, cout),
        )
    def forward(self, x): return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(min(8, cout), cout),
            nn.SiLU(inplace=True),
            ConvBlock(cout, cout),
        )
    def forward(self, x): return self.net(x)


class TaiwanZipSpatialVAEV2(nn.Module):
    def __init__(self, in_channels=7, latent_channels=16, base_channels=32):
        super().__init__()
        b = base_channels
        self.encoder = nn.Sequential(
            DownBlock(in_channels, b),
            DownBlock(b, b * 2),
            DownBlock(b * 2, b * 4),
            DownBlock(b * 4, b * 4),
            ConvBlock(b * 4, b * 4),
        )
        self.mu_head = nn.Conv2d(b * 4, latent_channels, 1)
        self.logvar_head = nn.Conv2d(b * 4, latent_channels, 1)
        self.decoder_in = ConvBlock(latent_channels, b * 4)
        self.decoder = nn.Sequential(
            UpBlock(b * 4, b * 4),
            UpBlock(b * 4, b * 2),
            UpBlock(b * 2, b),
            UpBlock(b, b),
            ConvBlock(b, b),
        )
        self.depth_head = nn.Sequential(nn.Conv2d(b, 1, 3, padding=1), nn.Sigmoid())
        self.semantic_head = nn.Conv2d(b, 6, 3, padding=1)

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def decode(self, z):
        h = self.decoder_in(z)
        h = self.decoder(h)
        return self.depth_head(h), self.semantic_head(h)


class NPZDataset(Dataset):
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.depth = d["depth"]
        self.sem = d["semantics"]
        self.scene_ids = d["scene_ids"]
        self.sample_ids = d["sample_ids"]

    def __len__(self): return len(self.depth)

    def __getitem__(self, i):
        dep = torch.from_numpy(self.depth[i].astype(np.float32))[None]
        sem = torch.from_numpy(self.sem[i].astype(np.float32))
        return {
            "x": torch.cat([dep, sem], dim=0),
            "scene_id": str(self.scene_ids[i]),
            "sample_id": str(self.sample_ids[i]),
        }


def smoothstep01(x):
    return x * x * (3.0 - 2.0 * x)


def find_dataset_root(work):
    for p in [work / "training_taiwan_zip_spatial_v2", work / "training_taiwan_zip_prototype"]:
        if p.exists(): return p
    raise FileNotFoundError("Could not locate extracted Taiwan.zip dataset folder.")


@torch.no_grad()
def encode_mu(model, ds, idx, device):
    item = ds[idx]
    mu, _ = model.encode(item["x"][None].to(device))
    return item, mu[0]


@torch.no_grad()
def decode_fields(model, z, device):
    depth, sem_logits = model.decode(z[None].to(device))
    depth = depth[0, 0].float().cpu().numpy()
    sem = torch.sigmoid(sem_logits[0]).float().cpu().numpy()
    return depth, sem


def make_local_mask(z, radius_ratio, device):
    c, h, w = z.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    radius = min(h, w) * radius_ratio
    mask2d = (((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2).astype(np.float32)
    mask = np.repeat(mask2d[None, ...], c, axis=0)
    return torch.from_numpy(mask).to(device)


def fields_to_pointcloud(depth, sem, width, depth_strength, stride, min_prob):
    h, w = depth.shape
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    xx, yy = np.meshgrid(xs, ys)

    height = width * (h / w)
    xr = (xx.astype(np.float32) / max(w - 1, 1) - 0.5) * width
    zr = (0.5 - yy.astype(np.float32) / max(h - 1, 1)) * height
    d = depth[yy, xx]
    yr = (1.0 - d) * depth_strength

    probs = sem[:, yy, xx].transpose(1, 2, 0)
    dominant = probs.argmax(axis=-1)
    conf = probs.max(axis=-1)
    rgb = COLORS[dominant]
    alpha = np.clip((conf - min_prob) / max(1.0 - min_prob, 1e-6), 0.0, 1.0)

    xyz = np.stack([xr, yr, zr], axis=-1).reshape(-1, 3)
    rgb = rgb.reshape(-1, 3)
    alpha = alpha.reshape(-1)
    keep = alpha > 0.01
    return xyz[keep], rgb[keep], alpha[keep]


def render_frame(xyz, rgb, alpha, out_path, width, depth_strength, elev, azim, point_size, dpi, title):
    fig = plt.figure(figsize=(10.67, 6.0), facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")
    colors = rgb * (0.35 + 0.65 * alpha[:, None])
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=point_size,
               linewidths=0, depthshade=False, rasterized=True)
    ax.set_xlim(-width / 2, width / 2)
    ax.set_ylim(0, depth_strength)
    ax.set_zlim(-width / 2, width / 2)
    ax.set_box_aspect((width, depth_strength, width))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.text2D(0.035, 0.94, title, transform=ax.transAxes,
              color="white", fontsize=12, family="monospace")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=dpi, facecolor="black")
    plt.close(fig)


def encode_video(frame_dir, fps, output_path):
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. It is normally available in Colab.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "17",
        "-preset", "medium",
        "-movflags", "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    work = Path(args.work_dir)
    frames_dir = work / "frames"
    if work.exists(): shutil.rmtree(work)
    frames_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.data_zip, "r") as z:
        z.extractall(work)

    data_root = find_dataset_root(work)
    reg = NPZDataset(data_root / "regression.npz")

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ck_args = ck.get("args", {})
    latent_channels = int(ck_args.get("latent_channels", args.latent_channels))
    base_channels = int(ck_args.get("base_channels", args.base_channels))

    model = TaiwanZipSpatialVAEV2(7, latent_channels, base_channels).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    item_a, za = encode_mu(model, reg, args.index_a, device)
    item_b, zb = encode_mu(model, reg, args.index_b, device)
    print("A:", item_a["sample_id"])
    print("B:", item_b["sample_id"])
    print("Latent shape:", tuple(za.shape))

    u = np.linspace(0.0, 1.0, args.frames, dtype=np.float32)
    eased = smoothstep01(u)
    latent_frames, labels = [], []

    if args.mode == "interpolation":
        ts = args.t_start + (args.t_end - args.t_start) * eased
        for t in ts:
            latent_frames.append(za * (1.0 - float(t)) + zb * float(t))
            labels.append(f"TAIWAN.ZIP  |  latent t = {t:.3f}")
    else:
        mask = make_local_mask(za, args.mutation_radius, device)
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed + 1000)
        fixed_noise = torch.randn(za.shape, generator=gen, device=device, dtype=za.dtype)
        sigmas = args.sigma_start + (args.sigma_end - args.sigma_start) * eased
        for sigma in sigmas:
            latent_frames.append(za + float(sigma) * fixed_noise * mask)
            labels.append(f"TAIWAN.ZIP  |  local mutation sigma = {sigma:.3f}")

    if args.hold_start > 0:
        latent_frames = [latent_frames[0]] * args.hold_start + latent_frames
        labels = [labels[0]] * args.hold_start + labels
    if args.hold_end > 0:
        latent_frames += [latent_frames[-1]] * args.hold_end
        labels += [labels[-1]] * args.hold_end

    print("Frames:", len(latent_frames))
    for i, (z, label) in enumerate(zip(latent_frames, labels)):
        depth, sem = decode_fields(model, z, device)
        xyz, rgb, alpha = fields_to_pointcloud(
            depth, sem, args.width, args.depth_strength,
            max(1, args.stride), args.semantic_min_prob
        )
        progress = i / max(len(latent_frames) - 1, 1)
        azim = args.azim + args.camera_drift * (progress - 0.5)
        render_frame(
            xyz, rgb, alpha, frames_dir / f"frame_{i:05d}.png",
            args.width, args.depth_strength, args.elev, azim,
            args.point_size, args.dpi, label
        )
        if (i + 1) % 10 == 0 or i == len(latent_frames) - 1:
            print(f"Rendered {i+1}/{len(latent_frames)}")

    output_path = Path(args.output)
    encode_video(frames_dir, args.fps, output_path)

    meta = {
        "version": "taiwan-zip-latent-morph-pointcloud-v1",
        "mode": args.mode,
        "sample_a": item_a["sample_id"],
        "sample_b": item_b["sample_id"],
        "latent_shape": list(za.shape),
        "frames_generated": args.frames,
        "frames_with_holds": len(latent_frames),
        "fps": args.fps,
        "output": str(output_path),
        "render": {
            "width": args.width,
            "depth_strength": args.depth_strength,
            "stride": args.stride,
            "point_size": args.point_size,
            "semantic_min_prob": args.semantic_min_prob,
            "elev": args.elev,
            "azim": args.azim,
            "camera_drift": args.camera_drift,
        },
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not args.keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)

    print("DONE")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
