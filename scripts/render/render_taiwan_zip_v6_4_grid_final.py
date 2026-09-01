#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taiwan.zip — V6.3 MODEL FIX
===========================

Correct final pipeline:
LOCAL SOURCE:
    rgb.jpg
    street46_depth_official.png
    street46.ply
        ↓
bridge into trained Spatial VAE V2
        ↓
16 model-derived latent states
        ↓
continuous latent morph
        ↓
final model-generated hero / stills

Important:
- The 16 states are NOT derived from the local PLY.
- The local PLY is used only for the opening / bridge.
- The latent states are encoded from regression samples in the spatial_v2 bundle,
  manipulated in latent space, then decoded by the trained checkpoint.
"""

from __future__ import annotations

import argparse
import io
import json
import math
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
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData


SCRIPT_VERSION = "taiwan_zip_v6_4_grid_final"

SEM_CHANNELS = ["facade", "window", "signboard", "vegetation", "person", "vehicle"]

SEMANTIC_COLORS = np.array([
    [0.78, 0.80, 0.82],
    [0.20, 0.58, 1.00],
    [1.00, 0.47, 0.10],
    [0.18, 0.80, 0.38],
    [0.96, 0.26, 0.66],
    [1.00, 0.82, 0.15],
], dtype=np.float32)

STATE_KEYWORDS = [
    ("TRACE", "sampled architectural residue"),
    ("PARCEL", "fragmented urban units"),
    ("LATTICE", "window-sign-facade ordering"),
    ("DRIFT", "semantic displacement"),
    ("SPLICE", "re-linked street fragments"),
    ("RELAY", "relational transfer across scenes"),
    ("FOLD", "folded depth logic"),
    ("WEAVE", "woven point trajectories"),
    ("ECHO", "architectural afterimage"),
    ("BLOOM", "expanded semantic cloud"),
    ("OFFSET", "shifted spatial hierarchy"),
    ("SPILL", "overflowing categorical edges"),
    ("VECTOR", "directional urban memory"),
    ("HYBRID", "hybridized streetscape body"),
    ("RESONANCE", "recomposed computational perception"),
    ("FIELD", "final hybrid point-cloud field"),
]

BG = np.array([0.015, 0.020, 0.035], dtype=np.float32)
GHOST_TINT_A = np.array([0.22, 0.55, 0.90], dtype=np.float32)
GHOST_TINT_B = np.array([0.85, 0.30, 0.72], dtype=np.float32)


# ============================================================
# Args
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-zip", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--seconds", type=float, default=50.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)

    p.add_argument("--num-states", type=int, default=16)
    p.add_argument("--anchor-indices", default="0,6,12,18,24,31")

    p.add_argument("--point-stride", type=int, default=4)
    p.add_argument("--point-size", type=float, default=2.4)
    p.add_argument("--depth-strength", type=float, default=430.0)
    p.add_argument("--semantic-min-prob", type=float, default=0.16)

    p.add_argument("--perturb-strength", type=float, default=0.85)
    p.add_argument("--perturb-radius", type=float, default=0.27)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--grid-hold-seconds", type=float, default=3.0)
    p.add_argument("--grid-fade-seconds", type=float, default=0.8)
    p.add_argument("--keep-frames", action="store_true")
    return p.parse_args()


# ============================================================
# Exact trained model
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p),
            nn.GroupNorm(num_groups=min(8, cout), num_channels=cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(cin, cout, k=4, s=2, p=1),
            ConvBlock(cout, cout),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=min(8, cout), num_channels=cout),
            nn.SiLU(inplace=True),
            ConvBlock(cout, cout),
        )

    def forward(self, x):
        return self.net(x)


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

        self.depth_head = nn.Sequential(
            nn.Conv2d(b, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.semantic_head = nn.Conv2d(b, 6, 3, padding=1)

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def decode(self, z):
        h = self.decoder_in(z)
        h = self.decoder(h)
        return self.depth_head(h), self.semantic_head(h)


# ============================================================
# Bundle loader
# ============================================================

def find_regression_npz(zip_path: Path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        candidates = [n for n in names if n.endswith("/regression.npz") or n == "regression.npz"]
        if not candidates:
            candidates = [n for n in names if n.endswith("regression.npz")]
        if not candidates:
            raise FileNotFoundError("regression.npz not found in bundle zip.")
        return candidates[0]


def load_regression_from_zip(zip_path: Path):
    inner = find_regression_npz(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        raw = zf.read(inner)
    npz = np.load(io.BytesIO(raw), allow_pickle=False)

    required = ["depth", "semantics", "scene_ids", "sample_ids"]
    for k in required:
        if k not in npz.files:
            raise KeyError(f"Missing {k} in regression npz. Keys={npz.files}")

    depth = np.asarray(npz["depth"], dtype=np.float32)
    sem = np.asarray(npz["semantics"], dtype=np.float32)
    scene_ids = np.asarray(npz["scene_ids"])
    sample_ids = np.asarray(npz["sample_ids"])

    # Expected:
    # depth [N,H,W]
    # semantics [N,6,H,W]
    if depth.ndim != 3:
        raise ValueError(f"Unexpected depth shape: {depth.shape}")
    if sem.ndim != 4 or sem.shape[1] != 6:
        raise ValueError(f"Unexpected semantics shape: {sem.shape}")

    return {
        "depth": depth,
        "semantics": sem,
        "scene_ids": scene_ids,
        "sample_ids": sample_ids,
        "inner_name": inner,
    }


def sample_tensor(reg, idx):
    d = torch.from_numpy(reg["depth"][idx].astype(np.float32))[None]
    s = torch.from_numpy(reg["semantics"][idx].astype(np.float32))
    return torch.cat([d, s], dim=0)


# ============================================================
# Latent path
# ============================================================

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def smootherstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (x * (x * 6.0 - 15.0) + 10.0)


def lerp(a, b, t):
    return a * (1.0 - t) + b * t


def gaussian_mask(shape, cx, cy, radius, device):
    c, h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx /= max(w - 1, 1)
    yy /= max(h - 1, 1)

    sigma = max(radius * 0.42, 1e-4)
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    m = np.exp(-d2 / (2.0 * sigma * sigma))
    m = np.repeat(m[None], c, axis=0).astype(np.float32)
    return torch.from_numpy(m).to(device)


@torch.no_grad()
def encode_index(model, reg, idx, device):
    x = sample_tensor(reg, idx)[None].to(device)
    mu, _ = model.encode(x)
    return mu[0]


@torch.no_grad()
def decode_latent(model, z, device):
    d, s = model.decode(z[None].to(device))
    return (
        d[0, 0].float().cpu().numpy(),
        torch.sigmoid(s[0]).float().cpu().numpy(),
    )


def build_16_states(anchor_latents, noise, device, perturb_strength, perturb_radius, n=16):
    """
    Single continuous latent path.
    Operations are embedded into successive state construction:
    interpolation / extrapolation / perturbation / recombination.
    """
    states = [anchor_latents[0].clone()]
    modes = ["source"]

    for i in range(1, n):
        a = anchor_latents[(i - 1) % len(anchor_latents)]
        b = anchor_latents[i % len(anchor_latents)]
        mode = i % 4

        if mode == 1:
            t = 0.35 + 0.40 * (0.5 + 0.5 * math.sin(i * 0.61))
            z = lerp(a, b, t)
            modes.append("interpolation")

        elif mode == 2:
            t = 1.10 + 0.35 * (0.5 + 0.5 * math.sin(i * 0.73))
            z = lerp(a, b, t)
            modes.append("extrapolation")

        elif mode == 3:
            base = lerp(a, b, 0.45)
            cx = 0.20 + 0.58 * (0.5 + 0.5 * math.sin(i * 0.83))
            cy = 0.28 + 0.44 * (0.5 + 0.5 * math.cos(i * 0.67))
            mask = gaussian_mask(base.shape, cx, cy, perturb_radius, device)
            z = base + perturb_strength * noise * mask
            modes.append("perturbation")

        else:
            prev = states[-1]
            c = anchor_latents[(i + 1) % len(anchor_latents)]
            z = 0.45 * prev + 0.35 * b + 0.20 * c
            modes.append("recombination")

        states.append(z)

    return states, modes


# ============================================================
# Local assets
# ============================================================

def load_local_ply(path: Path):
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    names = v.data.dtype.names

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    if {"red", "green", "blue"}.issubset(set(names)):
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32) / 255.0
    else:
        rgb = np.ones((len(xyz), 3), dtype=np.float32) * 0.85

    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    rgb = rgb[finite]

    # normalize for renderer only
    xyz = xyz - np.median(xyz, axis=0, keepdims=True)
    scale = np.max(np.ptp(xyz, axis=0))
    xyz = xyz / max(scale, 1e-6)
    xyz *= 520.0

    if len(xyz) > 8000:
        idx = np.linspace(0, len(xyz) - 1, 8000).astype(np.int32)
        xyz, rgb = xyz[idx], rgb[idx]

    return xyz, rgb


def resample_cloud(xyz, rgb, n):
    if len(xyz) == 0:
        raise RuntimeError("Empty point cloud.")
    idx = np.linspace(0, len(xyz) - 1, n).round().astype(int)
    return xyz[idx], rgb[idx]


# ============================================================
# Generated fields -> pseudo 3D cloud
# ============================================================

def fields_to_cloud(depth, sem, width, depth_strength, stride, min_prob):
    h, w = depth.shape

    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    xx, yy = np.meshgrid(xs, ys)

    height = width * h / w

    x = (xx.astype(np.float32) / max(w - 1, 1) - 0.5) * width
    z = (0.5 - yy.astype(np.float32) / max(h - 1, 1)) * height
    d = depth[yy, xx]
    y = (1.0 - d) * depth_strength

    probs = sem[:, yy, xx].transpose(1, 2, 0)
    dominant = np.argmax(probs, axis=-1)
    confidence = np.max(probs, axis=-1)

    rgb = SEMANTIC_COLORS[dominant]
    alpha = np.clip(
        (confidence - min_prob) / max(1.0 - min_prob, 1e-6),
        0.0, 1.0
    )

    xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    rgb = rgb.reshape(-1, 3)
    alpha = alpha.reshape(-1)

    keep = alpha > 0.01
    return xyz[keep], rgb[keep], alpha[keep]


# ============================================================
# Render
# ============================================================

def draw_label(ax, title, subtitle):
    ax.text2D(
        0.035, 0.945, title,
        transform=ax.transAxes,
        color="white",
        fontsize=14,
        family="monospace",
        weight="bold",
    )
    if subtitle:
        ax.text2D(
            0.035, 0.905, subtitle,
            transform=ax.transAxes,
            color=(0.75, 0.83, 0.92),
            fontsize=9,
            family="monospace",
        )


def camera_for(progress, intro=False):
    if intro:
        t = smootherstep(progress)
        return {
            "azim": -88 + 40 * t,
            "elev": 7 + 15 * t,
            "zoom": 1.02 - 0.08 * t,
        }

    # mostly fixed isometric with a subtle late walk-in
    if progress < 0.78:
        return {
            "azim": -48 + 2.5 * math.sin(progress * math.pi * 2.0),
            "elev": 22 + 1.2 * math.sin(progress * math.pi * 1.4),
            "zoom": 0.94,
        }

    t = smootherstep((progress - 0.78) / 0.22)
    return {
        "azim": -48 + 10 * t,
        "elev": 22 - 4 * t,
        "zoom": 0.94 - 0.11 * t,
    }


def make_ghost_clouds(decoded_states, stride=6):
    ghosts = []
    picks = np.linspace(0, len(decoded_states) - 1, min(5, len(decoded_states))).round().astype(int)

    for idx in picks:
        d, s = decoded_states[idx]
        xyz, rgb, alpha = fields_to_cloud(
            d, s,
            width=540,
            depth_strength=440,
            stride=stride,
            min_prob=0.10,
        )
        ghosts.append((xyz, rgb, alpha))

    return ghosts


def moving_ghosts(ghosts, progress):
    out = []
    for i, (xyz, rgb, alpha) in enumerate(ghosts):
        phase = progress * math.pi * 2.0 + i * 1.27
        q = xyz.copy()
        q[:, 0] += math.sin(phase) * 55.0
        q[:, 1] += math.cos(phase * 0.72) * 28.0
        q[:, 2] += math.cos(phase * 1.11) * 42.0
        q *= 1.05 + 0.03 * math.sin(phase * 0.5)

        tint = GHOST_TINT_A if i % 2 == 0 else GHOST_TINT_B
        c = rgb * 0.45 + tint[None] * 0.55

        out.append((q, np.clip(c, 0, 1), alpha * 0.12))
    return out


def render_cloud(
    xyz, rgb, alpha,
    out_path,
    W, H,
    camera,
    point_size,
    ghosts=None,
    title="",
    subtitle="",
    dpi=120,
):
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    ax = fig.add_subplot(111, projection="3d", facecolor=BG)

    if ghosts:
        for gx, grgb, ga in ghosts:
            visible = ga > 0.001
            col = grgb[visible] * (0.20 + 0.80 * ga[visible, None])
            ax.scatter(
                gx[visible, 0], gx[visible, 1], gx[visible, 2],
                c=col,
                s=max(0.55, point_size * 0.45),
                linewidths=0,
                depthshade=False,
                rasterized=True,
            )

    visible = alpha > 0.003
    col = rgb[visible] * (0.32 + 0.68 * alpha[visible, None])

    ax.scatter(
        xyz[visible, 0], xyz[visible, 1], xyz[visible, 2],
        c=col,
        s=point_size,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )

    zoom = camera["zoom"]
    width = 520 * 1.75 * zoom
    height = 520 * 1.15 * zoom

    ax.set_xlim(-width / 2, width / 2)
    ax.set_ylim(-30, 500)
    ax.set_zlim(-height / 2, height / 2)
    ax.set_box_aspect((width, 530, height))

    ax.view_init(elev=camera["elev"], azim=camera["azim"])
    ax.set_axis_off()

    if title:
        draw_label(ax, title, subtitle)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)


def render_image(img: Image.Image, out_path: Path, W, H, title, subtitle, blend=None):
    canvas = img.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)

    if blend is not None:
        other, t = blend
        other = other.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
        canvas = Image.blend(canvas, other, float(np.clip(t, 0, 1)))

    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rounded_rectangle((25, 22, 430, 88), radius=14, fill=(0, 0, 0, 105))

    try:
        f1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 27)
        f2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        f1 = ImageFont.load_default()
        f2 = ImageFont.load_default()

    draw.text((40, 31), title, fill=(250, 250, 250, 255), font=f1)
    draw.text((41, 62), subtitle, fill=(195, 210, 230, 255), font=f2)

    canvas.save(out_path)


def render_state_grid_image(state_clouds, out_path: Path, total_w: int, total_h: int, point_size: float):
    """Render a 4x4 final overview board of the 16 decoded states using one shared camera."""
    cols = 4
    rows = 4
    tile_w = total_w // cols
    tile_h = total_h // rows
    temp_dir = out_path.parent / (out_path.stem + "_tiles")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # One stable camera for all 16 states so they can be compared as one family.
    grid_cam = {"azim": -48, "elev": 22, "zoom": 0.92}

    canvas = Image.new("RGB", (total_w, total_h), tuple((BG * 255).astype(np.uint8).tolist()))
    draw = ImageDraw.Draw(canvas, "RGBA")

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(12, int(tile_h * 0.08)))
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(10, int(tile_h * 0.05)))
    except:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    margin = max(4, int(min(tile_w, tile_h) * 0.02))

    for i, (xyz, rgb, alpha) in enumerate(state_clouds[:16]):
        tile_path = temp_dir / f"tile_{i:02d}.png"
        render_cloud(
            xyz, rgb, alpha,
            tile_path,
            tile_w, tile_h,
            grid_cam,
            max(0.95, point_size * 0.85),
            ghosts=None,
            title="",
            subtitle="",
            dpi=120,
        )
        tile = Image.open(tile_path).convert("RGB")
        x = (i % cols) * tile_w
        y = (i // cols) * tile_h
        canvas.paste(tile, (x, y))

        # subtle border
        draw.rounded_rectangle(
            (x + margin, y + margin, x + tile_w - margin, y + tile_h - margin),
            radius=max(6, int(min(tile_w, tile_h) * 0.03)),
            outline=(255, 255, 255, 30),
            width=1,
        )

        keyword, subtitle = STATE_KEYWORDS[i]
        box_w = int(tile_w * 0.66)
        box_h = int(tile_h * 0.20)
        bx0 = x + margin * 2
        by0 = y + tile_h - box_h - margin * 2
        bx1 = bx0 + box_w
        by1 = by0 + box_h
        draw.rounded_rectangle((bx0, by0, bx1, by1), radius=10, fill=(0, 0, 0, 118))
        draw.text((bx0 + 12, by0 + 8), f"{i+1:02d}  {keyword}", fill=(250, 250, 250, 255), font=font_title)
        draw.text((bx0 + 13, by0 + 8 + int(box_h * 0.42)), subtitle, fill=(195, 210, 230, 255), font=font_sub)

    canvas.save(out_path)
    shutil.rmtree(temp_dir, ignore_errors=True)



def encode_video(frame_dir, fps, output):
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%05d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "17",
        "-preset", "medium",
        "-movflags", "+faststart",
        str(output),
    ]
    subprocess.run(cmd, check=True)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("SCRIPT_VERSION:", SCRIPT_VERSION)
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    work = Path("/content/taiwan_zip_v6_3_model_fix_work")
    frames = work / "frames"

    if work.exists():
        shutil.rmtree(work)
    frames.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Local intro
    # --------------------------------------------------------
    rgb_path = dataset_dir / "rgb.jpg"
    ply_path = dataset_dir / "street46.ply"
    depth_img_path = dataset_dir / "street46_depth_official.png"

    if not depth_img_path.exists():
        depth_img_path = dataset_dir / "depth_preview.png"

    rgb_img = Image.open(rgb_path).convert("RGB")
    depth_img = Image.open(depth_img_path).convert("RGB")

    local_xyz, local_rgb = load_local_ply(ply_path)

    # --------------------------------------------------------
    # Regression bundle
    # --------------------------------------------------------
    reg = load_regression_from_zip(Path(args.data_zip))

    print("Regression source:", reg["inner_name"])
    print("depth shape:", reg["depth"].shape)
    print("semantics shape:", reg["semantics"].shape)

    anchor_indices = [int(x.strip()) for x in args.anchor_indices.split(",") if x.strip()]
    for i in anchor_indices:
        if i < 0 or i >= len(reg["depth"]):
            raise IndexError(f"Invalid anchor index {i}, regression size={len(reg['depth'])}")

    print("Anchor samples:")
    for i in anchor_indices:
        print(" ", i, str(reg["sample_ids"][i]))

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ck_args = ck.get("args", {})

    latent_channels = int(ck_args.get("latent_channels", 16))
    base_channels = int(ck_args.get("base_channels", 32))

    model = TaiwanZipSpatialVAEV2(
        in_channels=7,
        latent_channels=latent_channels,
        base_channels=base_channels,
    ).to(device)

    print("Expected first key:", next(iter(model.state_dict().keys())))
    print("Checkpoint first key:", next(iter(ck["model"].keys())))

    model.load_state_dict(ck["model"])
    model.eval()

    anchors = [encode_index(model, reg, i, device) for i in anchor_indices]

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 991)
    noise = torch.randn(anchors[0].shape, generator=generator, device=device, dtype=anchors[0].dtype)

    states, state_modes = build_16_states(
        anchors,
        noise,
        device,
        args.perturb_strength,
        args.perturb_radius,
        n=args.num_states,
    )

    # Pre-decode only states (transition frames are decoded individually later)
    decoded_states = [decode_latent(model, z, device) for z in states]
    ghosts_base = make_ghost_clouds(decoded_states)

    # --------------------------------------------------------
    # State stills
    # --------------------------------------------------------
    still_dir = output_dir / "stills"
    still_dir.mkdir(exist_ok=True)

    state_clouds = []
    for i, (depth, sem) in enumerate(decoded_states):
        xyz, rgb, alpha = fields_to_cloud(
            depth, sem,
            width=520,
            depth_strength=args.depth_strength,
            stride=args.point_stride,
            min_prob=args.semantic_min_prob,
        )
        state_clouds.append((xyz, rgb, alpha))

    selected_stills = [0, 3, 7, 11, 15]
    for i in selected_stills:
        xyz, rgb, alpha = state_clouds[i]
        cam = camera_for(i / max(args.num_states - 1, 1))
        render_cloud(
            xyz, rgb, alpha,
            still_dir / f"{i+1:02d}_{STATE_KEYWORDS[i][0].lower()}.png",
            args.width, args.height,
            cam,
            args.point_size,
            moving_ghosts(ghosts_base, i / max(args.num_states - 1, 1)),
            STATE_KEYWORDS[i][0],
            STATE_KEYWORDS[i][1],
            dpi=140,
        )

    # local PLY still
    local_xyz_r, local_rgb_r = resample_cloud(local_xyz, local_rgb, min(7000, len(local_xyz)))
    local_alpha = np.ones(len(local_xyz_r), dtype=np.float32) * 0.92

    render_cloud(
        local_xyz_r, local_rgb_r, local_alpha,
        still_dir / "00_local_pointcloud.png",
        args.width, args.height,
        camera_for(1.0, intro=True),
        args.point_size,
        moving_ghosts(ghosts_base, 0.0),
        "LIFTED FIELD",
        "local img2pointclouds reconstruction",
        dpi=140,
    )

    # hero from final MODEL state
    hero_path = output_dir / "hero_main_visual.png"
    xyz, rgb, alpha = state_clouds[-1]
    render_cloud(
        xyz, rgb, alpha,
        hero_path,
        1600, 900,
        {"azim": -40, "elev": 20, "zoom": 0.83},
        max(2.6, args.point_size),
        moving_ghosts(ghosts_base, 0.92),
        "FIELD",
        "final model-derived hybrid point-cloud field",
        dpi=160,
    )

    state_grid_path = output_dir / "final_16state_grid.png"
    state_grid_video_path = output_dir / "final_16state_grid_video.png"
    render_state_grid_image(state_clouds, state_grid_path, 2400, 2400, args.point_size)
    render_state_grid_image(state_clouds, state_grid_video_path, args.width, args.height, args.point_size)
    state_grid_img = Image.open(state_grid_video_path).convert("RGB")
    hero_img = Image.open(hero_path).convert("RGB")

    # --------------------------------------------------------
    # Video timeline
    # --------------------------------------------------------
    base_frames = round(args.seconds * args.fps)
    grid_frames = max(1, round(args.grid_hold_seconds * args.fps))
    grid_fade_frames = max(1, min(grid_frames, round(args.grid_fade_seconds * args.fps)))
    total_frames = base_frames + grid_frames

    rgb_frames = round(3.0 * args.fps)
    depth_frames = round(3.0 * args.fps)
    local_cloud_frames = round(4.0 * args.fps)
    bridge_frames = round(4.0 * args.fps)
    final_hold_frames = round(3.0 * args.fps)

    morph_frames = base_frames - rgb_frames - depth_frames - local_cloud_frames - bridge_frames - final_hold_frames
    morph_frames = max(morph_frames, args.num_states - 1)

    seg_count = args.num_states - 1
    seg_lengths = np.full(seg_count, morph_frames // seg_count, dtype=int)
    for i in range(morph_frames % seg_count):
        seg_lengths[i] += 1

    print("Timeline:", {
        "rgb": rgb_frames,
        "depth": depth_frames,
        "local_cloud": local_cloud_frames,
        "bridge": bridge_frames,
        "latent_morph": morph_frames,
        "final_hold": final_hold_frames,
        "final_grid": grid_frames,
        "total": total_frames,
    })

    frame_idx = 0

    # RGB
    for j in range(rgb_frames):
        render_image(
            rgb_img, frames / f"frame_{frame_idx:05d}.png",
            args.width, args.height,
            "OBSERVED STREET",
            "human-visible Taiwanese streetscape"
        )
        frame_idx += 1

    # RGB -> depth
    for j in range(depth_frames):
        t = smootherstep(j / max(depth_frames - 1, 1))
        render_image(
            rgb_img, frames / f"frame_{frame_idx:05d}.png",
            args.width, args.height,
            "MEASURED DEPTH",
            "relative spatial estimation",
            blend=(depth_img, t)
        )
        frame_idx += 1

    # Local point cloud
    for j in range(local_cloud_frames):
        p = j / max(local_cloud_frames - 1, 1)
        render_cloud(
            local_xyz_r, local_rgb_r, local_alpha,
            frames / f"frame_{frame_idx:05d}.png",
            args.width, args.height,
            camera_for(p, intro=True),
            args.point_size,
            moving_ghosts(ghosts_base, frame_idx / max(total_frames - 1, 1)),
            "LIFTED FIELD",
            "local img2pointclouds reconstruction",
            dpi=120,
        )
        frame_idx += 1

    # Bridge local PLY -> first MODEL state
    first_xyz, first_rgb, first_alpha = state_clouds[0]
    bridge_n = min(4500, max(1800, min(len(local_xyz_r), len(first_xyz))))
    ax0, ar0 = resample_cloud(local_xyz_r, local_rgb_r, bridge_n)
    ax1, ar1 = resample_cloud(first_xyz, first_rgb, bridge_n)
    aa0 = np.ones(bridge_n, dtype=np.float32) * 0.92
    aa1 = np.ones(bridge_n, dtype=np.float32) * 0.88

    for j in range(bridge_frames):
        t = smootherstep(j / max(bridge_frames - 1, 1))
        xyz = lerp(ax0, ax1, t)
        rgb = np.clip(lerp(ar0, ar1, t), 0, 1)
        alpha = lerp(aa0, aa1, t)

        render_cloud(
            xyz, rgb, alpha,
            frames / f"frame_{frame_idx:05d}.png",
            args.width, args.height,
            camera_for(0.10 + 0.10 * t),
            args.point_size,
            moving_ghosts(ghosts_base, frame_idx / max(total_frames - 1, 1)),
            "LATENT THRESHOLD",
            "local reconstruction enters trained spatial field",
            dpi=120,
        )
        frame_idx += 1

    # 16-state true latent morph
    rendered = 0
    for seg in range(seg_count):
        za = states[seg]
        zb = states[seg + 1]
        nseg = int(seg_lengths[seg])

        title, subtitle = STATE_KEYWORDS[seg + 1]

        for j in range(nseg):
            t = smootherstep(j / max(nseg - 1, 1))
            z = lerp(za, zb, t)

            # CRITICAL: every frame decoded from the trained model
            d, s = decode_latent(model, z, device)
            xyz, rgb, alpha = fields_to_cloud(
                d, s,
                width=520,
                depth_strength=args.depth_strength,
                stride=args.point_stride,
                min_prob=args.semantic_min_prob,
            )

            progress = rendered / max(morph_frames - 1, 1)

            render_cloud(
                xyz, rgb, alpha,
                frames / f"frame_{frame_idx:05d}.png",
                args.width, args.height,
                camera_for(progress),
                args.point_size,
                moving_ghosts(ghosts_base, frame_idx / max(total_frames - 1, 1)),
                title,
                subtitle + f" / {state_modes[seg + 1]}",
                dpi=120,
            )

            frame_idx += 1
            rendered += 1

            if frame_idx % 50 == 0:
                print(f"Rendered {frame_idx}/{total_frames}")

    # final hold
    final_xyz, final_rgb, final_alpha = state_clouds[-1]

    for j in range(final_hold_frames):
        render_cloud(
            final_xyz, final_rgb, final_alpha,
            frames / f"frame_{frame_idx:05d}.png",
            args.width, args.height,
            camera_for(1.0),
            args.point_size,
            moving_ghosts(ghosts_base, frame_idx / max(total_frames - 1, 1)),
            "FIELD",
            "final model-derived hybrid state",
            dpi=120,
        )
        frame_idx += 1

    # final 4x4 grid overview of all 16 states
    for j in range(grid_frames):
        if j < grid_fade_frames:
            t = smootherstep(j / max(grid_fade_frames - 1, 1))
            frame = Image.blend(hero_img.resize((args.width, args.height)), state_grid_img, t)
        else:
            frame = state_grid_img
        frame.save(frames / f"frame_{frame_idx:05d}.png")
        frame_idx += 1

    # fix if rounding mismatch
    while frame_idx < total_frames:
        shutil.copy2(
            frames / f"frame_{frame_idx-1:05d}.png",
            frames / f"frame_{frame_idx:05d}.png"
        )
        frame_idx += 1

    video_path = output_dir / "taiwan_zip_v6_3_MODEL_FINAL.mp4"
    encode_video(frames, args.fps, video_path)

    manifest = {
        "version": SCRIPT_VERSION,
        "video": str(video_path),
        "hero": str(hero_path),
        "final_16state_grid": str(state_grid_path),
        "final_16state_grid_video": str(state_grid_video_path),
        "stills": str(still_dir),
        "regression_npz": reg["inner_name"],
        "regression_depth_shape": list(reg["depth"].shape),
        "regression_semantics_shape": list(reg["semantics"].shape),
        "anchors": [
            {
                "index": int(i),
                "scene_id": str(reg["scene_ids"][i]),
                "sample_id": str(reg["sample_ids"][i]),
            }
            for i in anchor_indices
        ],
        "states": [
            {
                "index": i + 1,
                "keyword": STATE_KEYWORDS[i][0],
                "mode": state_modes[i],
            }
            for i in range(args.num_states)
        ],
        "local_intro": {
            "rgb": str(rgb_path),
            "depth": str(depth_img_path),
            "ply": str(ply_path),
            "used_for_generated_states": False,
            "purpose": "opening and bridge only",
        },
        "important_note": "All 16 state clouds and all main morph frames are decoded from the trained Spatial VAE V2 checkpoint.",
        "final_grid_note": "The last scene in the video is a 4x4 overview board showing all 16 states in one shared camera view.",
    }

    (output_dir / "render_manifest_MODEL_FINAL.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not args.keep_frames:
        shutil.rmtree(frames, ignore_errors=True)

    print("DONE")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
