import os
import io
import json
import math
import zipfile
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from plyfile import PlyData
from tqdm import tqdm
import imageio.v2 as imageio


# =========================
# 基本設定
# =========================

RNG = np.random.default_rng(42)

STATE_KEYWORDS = [
    "TRACE",
    "PARCEL",
    "LATTICE",
    "DRIFT",
    "SPLICE",
    "RELAY",
    "FOLD",
    "WEAVE",
    "ECHO",
    "BLOOM",
    "OFFSET",
    "SPILL",
    "VECTOR",
    "HYBRID",
    "RESONANCE",
    "FIELD",
]

STATE_SUBTEXT = [
    "sampled architectural residue",
    "fragmented urban units",
    "window-sign-facade ordering",
    "semantic displacement",
    "re-linked street fragments",
    "relational transfer across scenes",
    "folded depth logic",
    "woven point trajectories",
    "architectural afterimage",
    "expanded semantic cloud",
    "shifted spatial hierarchy",
    "overflowing categorical edges",
    "directional urban memory",
    "hybridized streetscape body",
    "recomposed computational perception",
    "final hybrid point-cloud field",
]

SEM_CHANNELS = ["facade", "window", "signboard", "vegetation", "person", "vehicle"]

SEM_PALETTE = {
    "facade":     np.array([0.78, 0.78, 0.80], dtype=np.float32),
    "window":     np.array([0.47, 0.66, 0.95], dtype=np.float32),
    "signboard":  np.array([0.95, 0.54, 0.22], dtype=np.float32),
    "vegetation": np.array([0.55, 0.83, 0.31], dtype=np.float32),
    "person":     np.array([0.92, 0.52, 0.72], dtype=np.float32),
    "vehicle":    np.array([0.98, 0.86, 0.22], dtype=np.float32),
}

BG_COLOR = np.array([7, 10, 18], dtype=np.uint8)
BG_ACCENT_A = np.array([80, 140, 255], dtype=np.float32) / 255.0
BG_ACCENT_B = np.array([255, 120, 170], dtype=np.float32) / 255.0


# =========================
# 工具
# =========================

def ensure_even(x):
    return x if x % 2 == 0 else x - 1


def lerp(a, b, t):
    return a * (1.0 - t) + b * t


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ease_in_out(t):
    return 0.5 - 0.5 * math.cos(math.pi * np.clip(t, 0.0, 1.0))


def normalize_cloud(xyz, target_scale=2.2):
    xyz = xyz.astype(np.float32)
    center = xyz.mean(axis=0, keepdims=True)
    xyz = xyz - center
    span = np.max(np.ptp(xyz, axis=0))
    span = max(span, 1e-6)
    xyz = xyz / span * target_scale
    # 翻一下 y，比較接近視覺慣性
    xyz[:, 1] *= -1.0
    return xyz


def resample_cloud(xyz, rgb, n=3500):
    m = len(xyz)
    if m == 0:
        raise ValueError("Empty cloud cannot be resampled.")
    if m == n:
        return xyz.copy(), rgb.copy()
    if m > n:
        idx = np.linspace(0, m - 1, n).astype(np.int32)
        return xyz[idx], rgb[idx]
    # m < n 時補點
    rep = RNG.choice(m, size=n - m, replace=True)
    xyz2 = np.concatenate([xyz, xyz[rep]], axis=0)
    rgb2 = np.concatenate([rgb, rgb[rep]], axis=0)
    return xyz2, rgb2


def sort_cloud(xyz, rgb):
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    return xyz[order], rgb[order]


def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return img


def load_gray_image(path, size=None):
    img = Image.open(path).convert("L")
    if size is not None:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return img


def load_ply(path):
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    names = v.data.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    if {"red", "green", "blue"}.issubset(set(names)):
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32) / 255.0
    else:
        rgb = np.ones((len(xyz), 3), dtype=np.float32) * 0.88

    return xyz, rgb


def depth_to_rgb(depth01):
    """
    自製簡單 colormap：深處偏深藍，近處偏亮灰。
    """
    d = np.clip(depth01, 0.0, 1.0)
    c1 = np.array([12, 18, 30], dtype=np.float32) / 255.0
    c2 = np.array([80, 120, 180], dtype=np.float32) / 255.0
    c3 = np.array([220, 230, 240], dtype=np.float32) / 255.0

    mid = np.clip(d * 2.0, 0.0, 1.0)
    rgb = np.where(
        (d[..., None] < 0.5),
        lerp(c1, c2, mid[..., None]),
        lerp(c2, c3, ((d - 0.5) * 2.0)[..., None]),
    )
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def add_vignette(img_np, strength=0.25):
    h, w = img_np.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx = w * 0.5
    cy = h * 0.5
    rr = np.sqrt(((xx - cx) / (w * 0.65)) ** 2 + ((yy - cy) / (h * 0.65)) ** 2)
    mask = np.clip((rr - 0.2) / 0.9, 0.0, 1.0)
    factor = 1.0 - strength * mask[..., None]
    out = img_np.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_text_block(img_np, title="", subtitle="", footer=""):
    img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        font_footer = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_footer = ImageFont.load_default()

    w, h = img.size

    # 左上角標題
    if title:
        draw.rounded_rectangle((24, 22, 430, 92), radius=16, fill=(0, 0, 0, 130))
        draw.text((40, 32), title, fill=(245, 245, 250), font=font_title)
        if subtitle:
            draw.text((42, 68), subtitle, fill=(185, 195, 215), font=font_sub)

    # 右下角 footer
    if footer:
        bbox = draw.textbbox((0, 0), footer, font=font_footer)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x0 = w - tw - 34
        y0 = h - th - 28
        draw.rounded_rectangle((x0 - 14, y0 - 10, x0 + tw + 14, y0 + th + 10), radius=12, fill=(0, 0, 0, 120))
        draw.text((x0, y0), footer, fill=(220, 225, 235), font=font_footer)

    return np.array(img)


# =========================
# bundle zip 讀取
# =========================

def list_npz_in_zip(zip_path):
    zf = zipfile.ZipFile(zip_path, "r")
    names = [n for n in zf.namelist() if n.endswith(".npz")]
    zf.close()
    return names


def load_npz_from_zip(zip_path, inner_name):
    with zipfile.ZipFile(zip_path, "r") as zf:
        data = zf.read(inner_name)
    return np.load(io.BytesIO(data), allow_pickle=True)


def infer_depth_semantics(npz_obj):
    keys = list(npz_obj.files)

    # 1) 個別 key
    depth = None
    if "depth_norm" in keys:
        depth = npz_obj["depth_norm"]
    elif "depth" in keys:
        depth = npz_obj["depth"]
    elif "x" in keys:
        x = npz_obj["x"]
        if x.ndim == 3 and x.shape[0] >= 1:
            depth = x[0]
        elif x.ndim == 3 and x.shape[-1] >= 1:
            depth = x[..., 0]
    elif "arr_0" in keys:
        arr = npz_obj["arr_0"]
        if arr.ndim == 3 and arr.shape[0] == 7:
            depth = arr[0]
        elif arr.ndim == 3 and arr.shape[-1] == 7:
            depth = arr[..., 0]

    if depth is None:
        raise ValueError(f"Cannot infer depth from keys: {keys}")

    depth = np.asarray(depth).astype(np.float32)
    if depth.ndim == 3:
        depth = np.squeeze(depth)

    semantics = []
    individual = True
    for k in SEM_CHANNELS:
        if k in keys:
            semantics.append(np.asarray(npz_obj[k]).astype(np.float32))
        else:
            individual = False
            break

    if individual:
        sem = np.stack(semantics, axis=-1)
        return depth, sem

    # 2) 常見 stack key
    for k in ["semantic", "semantics", "semantic_stack", "y"]:
        if k in keys:
            arr = np.asarray(npz_obj[k]).astype(np.float32)
            if arr.ndim == 3 and arr.shape[-1] >= 6:
                return depth, arr[..., :6]
            if arr.ndim == 3 and arr.shape[0] >= 6:
                return depth, np.transpose(arr[:6], (1, 2, 0))

    # 3) x / arr_0
    if "x" in keys:
        x = np.asarray(npz_obj["x"]).astype(np.float32)
        if x.ndim == 3 and x.shape[0] >= 7:
            sem = np.transpose(x[1:7], (1, 2, 0))
            return depth, sem
        if x.ndim == 3 and x.shape[-1] >= 7:
            return depth, x[..., 1:7]

    if "arr_0" in keys:
        arr = np.asarray(npz_obj["arr_0"]).astype(np.float32)
        if arr.ndim == 3 and arr.shape[0] >= 7:
            sem = np.transpose(arr[1:7], (1, 2, 0))
            return depth, sem
        if arr.ndim == 3 and arr.shape[-1] >= 7:
            return depth, arr[..., 1:7]

    raise ValueError(f"Cannot infer semantics from keys: {keys}")


def semantic_color_mix(sem):
    h, w, c = sem.shape
    palette = np.stack([SEM_PALETTE[k] for k in SEM_CHANNELS], axis=0)  # [6,3]
    s = np.clip(sem, 0.0, 1.0)
    denom = np.sum(s, axis=-1, keepdims=True) + 1e-6
    mix = (s @ palette) / denom
    fallback = np.ones((h, w, 3), dtype=np.float32) * 0.65
    mask = (np.sum(s, axis=-1, keepdims=True) > 0.05).astype(np.float32)
    return mix * mask + fallback * (1.0 - mask)


def depth_sem_to_cloud(depth, sem, target_points=4200):
    depth = np.asarray(depth).astype(np.float32)
    sem = np.asarray(sem).astype(np.float32)

    if depth.ndim != 2:
        raise ValueError("depth must be HxW")
    if sem.ndim != 3:
        raise ValueError("sem must be HxWxC")

    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w]
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = np.clip(depth, 0.0, 1.0)

    sem_sum = sem.sum(axis=-1)
    valid = (sem_sum > 0.15) | (depth > 0.02)

    idx = np.argwhere(valid)
    if len(idx) == 0:
        # fallback：全部都取
        idx = np.argwhere(np.ones_like(depth, dtype=bool))

    if len(idx) > target_points * 3:
        sel = np.linspace(0, len(idx) - 1, target_points * 3).astype(np.int32)
        idx = idx[sel]

    ys = idx[:, 0]
    xs = idx[:, 1]

    d = depth[ys, xs]
    z = 1.8 - d * 1.45
    nx = (xs.astype(np.float32) / max(w - 1, 1) - 0.5)
    ny = (ys.astype(np.float32) / max(h - 1, 1) - 0.5)

    X = nx * (1.5 + 1.3 * z)
    Y = ny * (1.3 + 1.0 * z)
    Z = z

    xyz = np.stack([X, Y, Z], axis=1).astype(np.float32)
    rgb_sem = semantic_color_mix(sem)[ys, xs]
    rgb_depth = np.repeat((0.35 + 0.55 * d)[:, None], 3, axis=1)
    rgb = np.clip(0.8 * rgb_sem + 0.2 * rgb_depth, 0.0, 1.0)

    xyz = normalize_cloud(xyz, target_scale=2.2)
    xyz, rgb = sort_cloud(xyz, rgb)
    xyz, rgb = resample_cloud(xyz, rgb, target_points)
    return xyz, rgb


# =========================
# 相機 / 投影 / 繪製
# =========================

def camera_matrices(azim_deg=38.0, elev_deg=24.0, radius=5.8, target=(0, 0, 0)):
    az = math.radians(azim_deg)
    el = math.radians(elev_deg)
    tx, ty, tz = target

    cam = np.array([
        tx + radius * math.cos(el) * math.cos(az),
        ty + radius * math.sin(el),
        tz + radius * math.cos(el) * math.sin(az),
    ], dtype=np.float32)

    target = np.array([tx, ty, tz], dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)

    forward = target - cam
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)

    true_up = np.cross(right, forward)
    true_up = true_up / (np.linalg.norm(true_up) + 1e-8)

    R = np.stack([right, true_up, forward], axis=0)  # world -> cam
    return cam, R


def project_points(xyz, azim=38.0, elev=24.0, radius=5.8, w=1280, h=720, fov_deg=52.0):
    cam, R = camera_matrices(azim, elev, radius)
    rel = xyz - cam[None, :]
    cam_xyz = rel @ R.T

    z = cam_xyz[:, 2]
    valid = z > 0.05
    cam_xyz = cam_xyz[valid]
    z = z[valid]

    if len(cam_xyz) == 0:
        return valid, np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)

    f = (0.5 * w) / math.tan(math.radians(fov_deg) * 0.5)
    u = cam_xyz[:, 0] / z * f + w * 0.5
    v = -cam_xyz[:, 1] / z * f + h * 0.5

    u = u.astype(np.int32)
    v = v.astype(np.int32)

    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    final_mask = np.zeros(len(xyz), dtype=bool)
    final_idx = np.where(valid)[0][inb]
    final_mask[final_idx] = True

    return final_mask, u[inb], v[inb], z[inb]


def splat(canvas, xs, ys, colors, alphas, point_size=1):
    """
    簡單 vectorized splat。
    """
    h, w = canvas.shape[:2]
    flat = canvas.reshape(-1, 3)

    offsets = [(0, 0)]
    if point_size >= 2:
        offsets += [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if point_size >= 3:
        offsets += [(-1, -1), (1, -1), (-1, 1), (1, 1)]

    for dx, dy in offsets:
        xx = xs + dx
        yy = ys + dy
        m = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        if not np.any(m):
            continue
        idx = yy[m] * w + xx[m]
        c = colors[m]
        a = alphas[m][:, None]
        flat[idx] = flat[idx] * (1.0 - a) + c * a

    return flat.reshape(h, w, 3)


def render_cloud_frame(
    xyz_main,
    rgb_main,
    title="",
    subtitle="",
    footer="",
    ghost_xyz=None,
    ghost_rgb=None,
    w=1280,
    h=720,
    azim=38.0,
    elev=24.0,
    radius=5.8,
    point_size_main=2,
    point_size_ghost=1,
    main_alpha=0.92,
    ghost_alpha=0.10,
):
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    canvas[:] = BG_COLOR.astype(np.float32) / 255.0

    # 先畫 ghost
    if ghost_xyz is not None and ghost_rgb is not None and len(ghost_xyz) > 0:
        mask, u, v, z = project_points(ghost_xyz, azim, elev, radius, w, h)
        if len(u) > 0:
            rgb = ghost_rgb[mask]
            zz = (z - z.min()) / (max(z.max() - z.min(), 1e-6))
            a = ghost_alpha * (0.55 + 0.45 * (1.0 - zz))
            order = np.argsort(z)[::-1]
            canvas = splat(canvas, u[order], v[order], rgb[order], a[order], point_size=point_size_ghost)

    # 再畫主點雲
    mask, u, v, z = project_points(xyz_main, azim, elev, radius, w, h)
    if len(u) > 0:
        rgb = rgb_main[mask]
        zz = (z - z.min()) / (max(z.max() - z.min(), 1e-6))
        a = main_alpha * (0.65 + 0.35 * (1.0 - zz))
        order = np.argsort(z)[::-1]
        canvas = splat(canvas, u[order], v[order], rgb[order], a[order], point_size=point_size_main)

    img = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
    img = add_vignette(img, strength=0.18)
    img = draw_text_block(img, title=title, subtitle=subtitle, footer=footer)
    return img


# =========================
# 背景 ghost cloud
# =========================

def build_ghost_cloud(state_xyz_list, state_rgb_list, total_points=12000):
    xyz_all = []
    rgb_all = []
    for xyz, rgb in zip(state_xyz_list[:6], state_rgb_list[:6]):
        xyz_all.append(xyz)
        pale = 0.55 * rgb + 0.45 * (0.5 * BG_ACCENT_A + 0.5 * BG_ACCENT_B)
        rgb_all.append(np.clip(pale, 0, 1))

    xyz = np.concatenate(xyz_all, axis=0)
    rgb = np.concatenate(rgb_all, axis=0)

    if len(xyz) > total_points:
        idx = np.linspace(0, len(xyz) - 1, total_points).astype(np.int32)
        xyz = xyz[idx]
        rgb = rgb[idx]

    return xyz, rgb


def animate_ghost_cloud(xyz, rgb, t_global):
    """
    讓背景不是格線，而是透明、流動、帶漂移感的點雲。
    """
    xyz2 = xyz.copy()
    phase = 2.0 * math.pi * t_global

    xyz2[:, 0] += 0.12 * np.sin(phase + xyz[:, 2] * 1.3)
    xyz2[:, 1] += 0.08 * np.cos(phase * 0.8 + xyz[:, 0] * 1.7)
    xyz2[:, 2] += 0.10 * np.sin(phase * 0.6 + xyz[:, 1] * 1.2)

    # 背景散得更開一點
    xyz2 *= np.array([1.25, 1.15, 1.20], dtype=np.float32)

    # 顏色淡化
    mix = 0.5 + 0.5 * np.sin(phase * 0.7)
    tint = lerp(BG_ACCENT_A, BG_ACCENT_B, mix)
    rgb2 = np.clip(0.35 * rgb + 0.65 * tint[None, :], 0.0, 1.0)
    return xyz2, rgb2


# =========================
# 狀態生成
# =========================

def recolor_local_cloud_by_height(xyz, rgb):
    """
    第一個 state 用 local cloud，但把顏色更設計化，
    讓它順暢接到後面計算點雲狀態。
    """
    zz = xyz[:, 2]
    z01 = (zz - zz.min()) / max(zz.max() - zz.min(), 1e-6)
    c_low = np.array([0.48, 0.70, 0.98], dtype=np.float32)
    c_high = np.array([0.98, 0.55, 0.32], dtype=np.float32)
    tint = lerp(c_low[None, :], c_high[None, :], z01[:, None])
    rgb2 = np.clip(0.45 * rgb + 0.55 * tint, 0.0, 1.0)
    return xyz.copy(), rgb2


def transformed_variant(xyz, rgb, i, total):
    """
    若 bundle 不足，可由 local cloud 直接推演變體。
    """
    t = i / max(total - 1, 1)
    xyz2 = xyz.copy()

    # 幾種輕度幾何擾動
    xyz2[:, 0] *= 1.0 + 0.08 * math.sin(2.0 * math.pi * t)
    xyz2[:, 1] *= 1.0 + 0.10 * math.cos(1.5 * math.pi * t)
    xyz2[:, 2] *= 1.0 + 0.12 * math.sin(1.2 * math.pi * t + 0.4)

    xyz2[:, 0] += 0.14 * np.sin(xyz[:, 2] * 2.2 + t * 4.0)
    xyz2[:, 1] += 0.10 * np.cos(xyz[:, 0] * 2.0 - t * 3.0)
    xyz2[:, 2] += 0.08 * np.sin(xyz[:, 1] * 1.8 + t * 5.0)

    rgb2 = rgb.copy()
    hue_a = np.array([0.50, 0.72, 0.97], dtype=np.float32)
    hue_b = np.array([0.99, 0.59, 0.28], dtype=np.float32)
    mix = 0.5 + 0.5 * math.sin(2.0 * math.pi * t)
    tint = lerp(hue_a, hue_b, mix)
    rgb2 = np.clip(0.55 * rgb2 + 0.45 * tint[None, :], 0.0, 1.0)

    return normalize_cloud(xyz2, target_scale=2.2), rgb2


def build_state_bank_from_bundle(bundle_zip, local_xyz, local_rgb, target_states=16, points_per_state=3600):
    state_xyz = []
    state_rgb = []

    # state 0：由 local cloud 開始，顏色重新設計
    xyz0, rgb0 = recolor_local_cloud_by_height(local_xyz, local_rgb)
    xyz0, rgb0 = sort_cloud(*resample_cloud(xyz0, rgb0, points_per_state))
    state_xyz.append(xyz0)
    state_rgb.append(rgb0)

    usable_clouds = []

    if bundle_zip is not None and Path(bundle_zip).exists():
        npz_names = list_npz_in_zip(bundle_zip)
        npz_names = sorted(npz_names)

        if len(npz_names) > 0:
            picks = np.linspace(0, len(npz_names) - 1, min(10, len(npz_names))).astype(np.int32)
            for p in picks:
                try:
                    npz = load_npz_from_zip(bundle_zip, npz_names[p])
                    depth, sem = infer_depth_semantics(npz)
                    xyz, rgb = depth_sem_to_cloud(depth, sem, target_points=points_per_state)
                    xyz, rgb = sort_cloud(xyz, rgb)
                    usable_clouds.append((xyz, rgb, npz_names[p]))
                except Exception as e:
                    print(f"[warn] skip {npz_names[p]}: {e}")

    # 如果 bundle 有內容，就拿它們來做 state
    if len(usable_clouds) >= 2:
        anchors = []
        for xyz, rgb, name in usable_clouds:
            anchors.append((xyz, rgb))

        # 目標：做滿 16 states
        # 結構：local -> anchor1 -> mix12 -> anchor2 -> mix23 ...
        idx = 0
        while len(state_xyz) < target_states:
            a = anchors[idx % len(anchors)]
            b = anchors[(idx + 1) % len(anchors)]

            # 先放 a
            if len(state_xyz) < target_states:
                state_xyz.append(a[0])
                state_rgb.append(a[1])

            # 再放中間混合
            if len(state_xyz) < target_states:
                mix_t = 0.5
                xyz_mix = lerp(a[0], b[0], mix_t)
                rgb_mix = np.clip(lerp(a[1], b[1], mix_t), 0.0, 1.0)
                xyz_mix = normalize_cloud(xyz_mix, target_scale=2.2)
                xyz_mix, rgb_mix = sort_cloud(xyz_mix, rgb_mix)
                state_xyz.append(xyz_mix)
                state_rgb.append(rgb_mix)

            idx += 1

        state_xyz = state_xyz[:target_states]
        state_rgb = state_rgb[:target_states]

    else:
        # bundle 不夠時，全部由 local cloud 生成
        print("[info] bundle not usable enough, fallback to local-derived 16 states.")
        state_xyz = []
        state_rgb = []
        for i in range(target_states):
            xyz_i, rgb_i = transformed_variant(local_xyz, local_rgb, i, target_states)
            xyz_i, rgb_i = sort_cloud(*resample_cloud(xyz_i, rgb_i, points_per_state))
            state_xyz.append(xyz_i)
            state_rgb.append(rgb_i)

    return state_xyz, state_rgb


# =========================
# Contact sheet
# =========================

def make_contact_sheet(images, labels, out_path, thumb_w=320, thumb_h=180, cols=4, pad=18):
    rows = math.ceil(len(images) / cols)
    W = cols * thumb_w + (cols + 1) * pad
    H = rows * thumb_h + (rows + 1) * pad + rows * 34

    sheet = Image.new("RGB", (W, H), tuple(BG_COLOR.tolist()))
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()

    for i, (img_np, label) in enumerate(zip(images, labels)):
        r = i // cols
        c = i % cols
        x = pad + c * (thumb_w + pad)
        y = pad + r * (thumb_h + pad + 34)

        img = Image.fromarray(img_np).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(img, (x, y))
        draw.text((x, y + thumb_h + 8), label, fill=(235, 240, 245), font=font)

    sheet.save(out_path)


# =========================
# 影片內容
# =========================

def rgb_to_np(img):
    return np.array(img).astype(np.uint8)


def fit_image_cover(img, size):
    w, h = img.size
    W, H = size
    scale = max(W / w, H / h)
    nw = int(w * scale)
    nh = int(h * scale)
    img2 = img.resize((nw, nh), Image.Resampling.LANCZOS)
    x0 = (nw - W) // 2
    y0 = (nh - H) // 2
    return img2.crop((x0, y0, x0 + W, y0 + H))


def intro_rgb_frame(rgb_img, w, h, t):
    """
    t: 0~1
    """
    zoom = 1.0 + 0.04 * t
    W0, H0 = rgb_img.size
    img = rgb_img.resize((int(W0 * zoom), int(H0 * zoom)), Image.Resampling.LANCZOS)
    frame = fit_image_cover(img, (w, h))
    arr = rgb_to_np(frame)
    arr = add_vignette(arr, strength=0.20)
    arr = draw_text_block(
        arr,
        title="HUMAN VIEW",
        subtitle="observed Taiwanese streetscape",
        footer="Taiwan.zip / local source RGB",
    )
    return arr


def intro_depth_frame(rgb_img, depth_img, w, h, t):
    rgb = fit_image_cover(rgb_img, (w, h))
    dep = fit_image_cover(depth_img.convert("RGB"), (w, h))

    rgb_np = np.array(rgb).astype(np.float32)
    dep_np = np.array(dep).astype(np.float32)

    # 由原圖慢慢進入 depth
    mix = smoothstep(t)
    out = rgb_np * (1.0 - 0.55 * mix) + dep_np * (0.55 * mix)

    # 疊一點霧化與亮部
    out = np.clip(out, 0, 255).astype(np.uint8)
    out = add_vignette(out, strength=0.16)
    out = draw_text_block(
        out,
        title="RELATIVE DEPTH",
        subtitle="spatial estimation from a single street image",
        footer="Depth Anything based relative depth",
    )
    return out


def intro_cloud_transition_frame(rgb_img, cloud_xyz, cloud_rgb, ghost_xyz, ghost_rgb, w, h, t):
    """
    原圖逐漸轉為真實 local point cloud。
    """
    base = fit_image_cover(rgb_img, (w, h))
    base_np = np.array(base).astype(np.float32)

    az = lerp(8.0, 38.0, smoothstep(t))
    el = lerp(6.0, 24.0, smoothstep(t))
    rad = lerp(6.8, 5.8, smoothstep(t))

    cloud_np = render_cloud_frame(
        cloud_xyz,
        cloud_rgb,
        title="ESTIMATED CLOUD",
        subtitle="local point cloud reconstructed from depth inference",
        footer="street46.ply",
        ghost_xyz=ghost_xyz,
        ghost_rgb=ghost_rgb,
        w=w,
        h=h,
        azim=az,
        elev=el,
        radius=rad,
        point_size_main=2,
        point_size_ghost=1,
        main_alpha=0.95,
        ghost_alpha=0.06,
    ).astype(np.float32)

    # 由原圖 crossfade 到點雲
    a = smoothstep(t)
    out = base_np * (1.0 - a) + cloud_np * a
    return np.clip(out, 0, 255).astype(np.uint8)


# =========================
# 主程式
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--bundle-zip", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seconds", type=float, default=50.0)
    parser.add_argument("--points-per-state", type=int, default=3600)
    parser.add_argument("--ghost-points", type=int, default=12000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(args.dataset_dir)
    bundle_zip = Path(args.bundle_zip) if args.bundle_zip else None

    W = ensure_even(args.width)
    H = ensure_even(args.height)
    fps = args.fps
    total_frames = int(round(args.seconds * fps))

    # ---------------------------------
    # local assets
    # ---------------------------------
    rgb_path = dataset_dir / "rgb.jpg"
    ply_path = dataset_dir / "street46.ply"

    depth_img_path = dataset_dir / "street46_depth_official.png"
    if not depth_img_path.exists():
        depth_img_path = dataset_dir / "depth_preview.png"

    if not rgb_path.exists():
        raise FileNotFoundError(f"Missing rgb.jpg: {rgb_path}")
    if not ply_path.exists():
        raise FileNotFoundError(f"Missing street46.ply: {ply_path}")
    if not depth_img_path.exists():
        raise FileNotFoundError(f"Missing depth preview image: {depth_img_path}")

    rgb_img = load_image(rgb_path)
    depth_img = load_gray_image(depth_img_path)

    local_xyz, local_rgb = load_ply(ply_path)
    local_xyz = normalize_cloud(local_xyz, target_scale=2.2)
    local_xyz, local_rgb = sort_cloud(local_xyz, local_rgb)
    local_xyz, local_rgb = resample_cloud(local_xyz, local_rgb, args.points_per_state)

    # ---------------------------------
    # 16-state bank
    # ---------------------------------
    state_xyz, state_rgb = build_state_bank_from_bundle(
        bundle_zip,
        local_xyz,
        local_rgb,
        target_states=16,
        points_per_state=args.points_per_state
    )

    # ghost cloud
    ghost_xyz, ghost_rgb = build_ghost_cloud(state_xyz, state_rgb, total_points=args.ghost_points)

    # ---------------------------------
    # 時間配置
    # ---------------------------------
    intro_rgb_frames = int(3.0 * fps)      # 3 秒
    intro_depth_frames = int(3.0 * fps)    # 3 秒
    intro_cloud_frames = int(4.0 * fps)    # 4 秒
    outro_hold_frames = int(2.0 * fps)     # 2 秒

    state_count = 16
    transitions = state_count - 1
    remaining = total_frames - (intro_rgb_frames + intro_depth_frames + intro_cloud_frames + outro_hold_frames)

    # 每個 state 先停一下，再做 transition
    hold_each = max(8, int(0.45 * fps))          # 約 0.45 秒
    trans_each = max(18, (remaining - state_count * hold_each) // transitions)

    timeline_total = intro_rgb_frames + intro_depth_frames + intro_cloud_frames + state_count * hold_each + transitions * trans_each + outro_hold_frames

    # 若不剛好，調整最後一段
    extra = total_frames - timeline_total
    last_hold_extra = max(0, extra)

    print("SCRIPT_VERSION: taiwan_zip_v6_3_final")
    print(f"Frames target: {total_frames}")
    print(f"Timeline => intro_rgb={intro_rgb_frames}, intro_depth={intro_depth_frames}, intro_cloud={intro_cloud_frames}, hold_each={hold_each}, trans_each={trans_each}, outro={outro_hold_frames}, last_extra={last_hold_extra}")

    # ---------------------------------
    # 預覽 still 圖
    # ---------------------------------
    preview_dir = out_dir / "stills"
    preview_dir.mkdir(exist_ok=True)

    # 先存幾張代表圖
    still_indices = [0, 3, 7, 11, 15]
    still_labels = []

    # local cloud still
    local_still = render_cloud_frame(
        local_xyz, local_rgb,
        title="ESTIMATED CLOUD",
        subtitle="local point cloud reconstructed from the source image",
        footer="local intro cloud",
        ghost_xyz=ghost_xyz, ghost_rgb=ghost_rgb,
        w=W, h=H, azim=38.0, elev=24.0, radius=5.8,
        point_size_main=2, point_size_ghost=1,
        main_alpha=0.95, ghost_alpha=0.06,
    )
    Image.fromarray(local_still).save(preview_dir / "00_local_estimated_cloud.png")

    still_images = [local_still]
    still_labels.append("00 / ESTIMATED CLOUD")

    for idx in still_indices:
        img = render_cloud_frame(
            state_xyz[idx], state_rgb[idx],
            title=STATE_KEYWORDS[idx],
            subtitle=STATE_SUBTEXT[idx],
            footer=f"state {idx+1:02d} / 16",
            ghost_xyz=ghost_xyz, ghost_rgb=ghost_rgb,
            w=W, h=H,
            azim=38.0 + 4.0 * math.sin(idx * 0.6),
            elev=24.0,
            radius=5.8,
            point_size_main=2,
            point_size_ghost=1,
            main_alpha=0.95,
            ghost_alpha=0.07,
        )
        Image.fromarray(img).save(preview_dir / f"{idx+1:02d}_{STATE_KEYWORDS[idx].lower()}.png")
        still_images.append(img)
        still_labels.append(f"{idx+1:02d} / {STATE_KEYWORDS[idx]}")

    # 主視覺：最後一個 state 再做一版更強
    hero = render_cloud_frame(
        state_xyz[-1], state_rgb[-1],
        title="FIELD",
        subtitle="final hybrid point-cloud still",
        footer="Taiwan.zip / hero visual",
        ghost_xyz=ghost_xyz, ghost_rgb=ghost_rgb,
        w=1600, h=900,
        azim=41.0, elev=26.0, radius=5.4,
        point_size_main=2,
        point_size_ghost=1,
        main_alpha=0.98,
        ghost_alpha=0.09,
    )
    Image.fromarray(hero).save(out_dir / "hero_main_visual.png")

    # 16-state contact sheet
    sheet_frames = []
    sheet_labels = []
    for i in range(16):
        thumb = render_cloud_frame(
            state_xyz[i], state_rgb[i],
            title="",
            subtitle="",
            footer="",
            ghost_xyz=ghost_xyz, ghost_rgb=ghost_rgb,
            w=960, h=540,
            azim=38.0, elev=24.0, radius=5.8,
            point_size_main=2,
            point_size_ghost=1,
            main_alpha=0.95,
            ghost_alpha=0.06,
        )
        sheet_frames.append(thumb)
        sheet_labels.append(f"{i+1:02d} / {STATE_KEYWORDS[i]}")
    make_contact_sheet(sheet_frames, sheet_labels, out_dir / "state_contact_sheet.png", thumb_w=300, thumb_h=169, cols=4, pad=18)

    # ---------------------------------
    # 影片
    # ---------------------------------
    video_path = out_dir / "taiwan_zip_v6_3_final.mp4"
    writer = imageio.get_writer(str(video_path), fps=fps, codec="libx264", quality=8)

    frame_counter = 0

    # 1) HUMAN VIEW
    for i in tqdm(range(intro_rgb_frames), desc="intro_rgb"):
        t = i / max(intro_rgb_frames - 1, 1)
        frame = intro_rgb_frame(rgb_img, W, H, t)
        writer.append_data(frame)
        frame_counter += 1

    # 2) RELATIVE DEPTH
    for i in tqdm(range(intro_depth_frames), desc="intro_depth"):
        t = i / max(intro_depth_frames - 1, 1)
        frame = intro_depth_frame(rgb_img, depth_img, W, H, t)
        writer.append_data(frame)
        frame_counter += 1

    # 3) ESTIMATED CLOUD
    for i in tqdm(range(intro_cloud_frames), desc="intro_cloud"):
        t = i / max(intro_cloud_frames - 1, 1)
        gt, gr = animate_ghost_cloud(ghost_xyz, ghost_rgb, frame_counter / max(total_frames, 1))
        frame = intro_cloud_transition_frame(rgb_img, local_xyz, local_rgb, gt, gr, W, H, t)
        writer.append_data(frame)
        frame_counter += 1

    # 4) 16-state morph
    for s in range(16):
        # 先 hold
        for k in tqdm(range(hold_each), desc=f"hold_{s+1:02d}", leave=False):
            g_t = frame_counter / max(total_frames, 1)
            gt, gr = animate_ghost_cloud(ghost_xyz, ghost_rgb, g_t)

            az = 38.0 + 2.0 * math.sin(2.0 * math.pi * g_t)
            el = 24.0 + 1.0 * math.sin(2.0 * math.pi * g_t * 0.6)
            rad = 5.8 + 0.15 * math.sin(2.0 * math.pi * g_t * 0.35)

            frame = render_cloud_frame(
                state_xyz[s], state_rgb[s],
                title=STATE_KEYWORDS[s],
                subtitle=STATE_SUBTEXT[s],
                footer=f"state {s+1:02d} / 16",
                ghost_xyz=gt, ghost_rgb=gr,
                w=W, h=H,
                azim=az, elev=el, radius=rad,
                point_size_main=2,
                point_size_ghost=1,
                main_alpha=0.95,
                ghost_alpha=0.08,
            )
            writer.append_data(frame)
            frame_counter += 1

        # state 之間 transition
        if s < 15:
            A_xyz = state_xyz[s]
            A_rgb = state_rgb[s]
            B_xyz = state_xyz[s+1]
            B_rgb = state_rgb[s+1]

            for j in tqdm(range(trans_each), desc=f"morph_{s+1:02d}_{s+2:02d}", leave=False):
                u = j / max(trans_each - 1, 1)
                u = ease_in_out(u)

                main_xyz = lerp(A_xyz, B_xyz, u)
                main_rgb = np.clip(lerp(A_rgb, B_rgb, u), 0.0, 1.0)

                g_t = frame_counter / max(total_frames, 1)
                gt, gr = animate_ghost_cloud(ghost_xyz, ghost_rgb, g_t)

                az = 38.0 + 3.5 * math.sin(2.0 * math.pi * g_t * 0.85)
                el = 24.0 + 1.8 * math.sin(2.0 * math.pi * g_t * 0.45 + 0.3)
                rad = 5.8 + 0.20 * math.sin(2.0 * math.pi * g_t * 0.30)

                frame = render_cloud_frame(
                    main_xyz, main_rgb,
                    title=STATE_KEYWORDS[s+1],
                    subtitle=STATE_SUBTEXT[s+1],
                    footer=f"morph {s+1:02d} → {s+2:02d}",
                    ghost_xyz=gt, ghost_rgb=gr,
                    w=W, h=H,
                    azim=az, elev=el, radius=rad,
                    point_size_main=2,
                    point_size_ghost=1,
                    main_alpha=0.95,
                    ghost_alpha=0.08,
                )
                writer.append_data(frame)
                frame_counter += 1

    # 5) final hold
    final_total_hold = outro_hold_frames + last_hold_extra
    for i in tqdm(range(final_total_hold), desc="outro_hold"):
        g_t = frame_counter / max(total_frames, 1)
        gt, gr = animate_ghost_cloud(ghost_xyz, ghost_rgb, g_t)

        frame = render_cloud_frame(
            state_xyz[-1], state_rgb[-1],
            title="FIELD",
            subtitle="final hybrid point-cloud state",
            footer="Taiwan.zip / final frame",
            ghost_xyz=gt, ghost_rgb=gr,
            w=W, h=H,
            azim=40.0, elev=25.0, radius=5.5,
            point_size_main=2,
            point_size_ghost=1,
            main_alpha=0.98,
            ghost_alpha=0.09,
        )
        writer.append_data(frame)
        frame_counter += 1

    writer.close()

    # ---------------------------------
    # manifest
    # ---------------------------------
    manifest = {
        "script_version": "taiwan_zip_v6_3_final",
        "dataset_dir": str(dataset_dir),
        "bundle_zip": str(bundle_zip) if bundle_zip else "",
        "output_dir": str(out_dir),
        "video_path": str(video_path),
        "hero_main_visual": str(out_dir / "hero_main_visual.png"),
        "state_contact_sheet": str(out_dir / "state_contact_sheet.png"),
        "stills_dir": str(preview_dir),
        "states": [
            {
                "index": i + 1,
                "keyword": STATE_KEYWORDS[i],
                "subtitle": STATE_SUBTEXT[i],
            }
            for i in range(16)
        ],
        "local_intro_assets": {
            "rgb": str(rgb_path),
            "ply": str(ply_path),
            "depth_preview": str(depth_img_path),
        },
        "timeline": {
            "fps": fps,
            "width": W,
            "height": H,
            "target_seconds": args.seconds,
            "total_frames_written": frame_counter,
            "intro_rgb_frames": intro_rgb_frames,
            "intro_depth_frames": intro_depth_frames,
            "intro_cloud_frames": intro_cloud_frames,
            "hold_each": hold_each,
            "transition_each": trans_each,
            "outro_hold": final_total_hold,
        },
    }

    with open(out_dir / "render_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("DONE")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()