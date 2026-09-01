#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taiwan.zip — Spatial VAE V2

重點：
- 真正的 spatial latent，不是單一 global 128-D vector
- latent shape 預設：16 x 16 x 16
- input channels = 7
- output = reconstructed depth + reconstructed semantics
- 會輸出：
    - training_curve.png
    - reconstruction_grid.png
    - latent_interpolation_grid.png
    - latent_extrapolation_grid.png
    - latent_local_perturbation_grid.png
    - best / last checkpoints
    - run_summary.json

Colab 執行範例：
!python /content/drive/MyDrive/PlayingModels/TaiwanZip/taiwan_zip_spatial_vae_v2.py \
    --data-zip /content/drive/MyDrive/PlayingModels/TaiwanZip/taiwan_zip_spatial_v2_bundle.zip \
    --output /content/drive/MyDrive/PlayingModels/TaiwanZip/results_spatial_v2 \
    --epochs 100 \
    --batch-size 16 \
    --latent-channels 16 \
    --base-channels 32 \
    --lr 2e-4
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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

SEM_COLORS = np.array(
    [
        [0.80, 0.80, 0.80],  # facade
        [0.20, 0.55, 1.00],  # window
        [1.00, 0.50, 0.10],  # signboard
        [0.20, 0.80, 0.35],  # vegetation
        [0.95, 0.25, 0.65],  # person
        [0.95, 0.80, 0.15],  # vehicle
    ],
    dtype=np.float32,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-zip", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", default="/content/taiwan_zip_spatial_v2_work")
    p.add_argument("--checkpoint-dir", default="/content/taiwan_zip_spatial_v2_ckpts")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--latent-channels", type=int, default=16)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--kl-beta", type=float, default=0.002)
    p.add_argument("--kl-warmup", type=int, default=20)
    p.add_argument("--depth-weight", type=float, default=1.0)
    p.add_argument("--bce-weight", type=float, default=1.0)
    p.add_argument("--dice-weight", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--min-delta", type=float, default=0.002)
    return p.parse_args()


class NPZDataset(Dataset):
    def __init__(self, npz_path, augment=False):
        data = np.load(npz_path, allow_pickle=False)
        self.depth = data["depth"]
        self.semantics = data["semantics"]
        self.scene_ids = data["scene_ids"]
        self.sample_ids = data["sample_ids"]
        self.group_ids = data["source_group_ids"]
        self.augment = augment

    def __len__(self):
        return len(self.depth)

    def __getitem__(self, idx):
        depth = torch.from_numpy(self.depth[idx].astype(np.float32))[None, ...]
        sem = torch.from_numpy(self.semantics[idx].astype(np.float32))
        x = torch.cat([depth, sem], dim=0)  # 7 x H x W

        if self.augment and random.random() < 0.5:
            x = torch.flip(x, dims=[2])  # horizontal flip

        return {
            "x": x,
            "scene_id": str(self.scene_ids[idx]),
            "sample_id": str(self.sample_ids[idx]),
            "group_id": str(self.group_ids[idx]),
        }


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
    """
    256x256 input
      -> 128
      -> 64
      -> 32
      -> 16
    latent = [B, latent_channels, 16, 16]
    """

    def __init__(self, in_channels=7, latent_channels=16, base_channels=32):
        super().__init__()
        b = base_channels

        self.encoder = nn.Sequential(
            DownBlock(in_channels, b),       # 256 -> 128
            DownBlock(b, b * 2),             # 128 -> 64
            DownBlock(b * 2, b * 4),         # 64 -> 32
            DownBlock(b * 4, b * 4),         # 32 -> 16
            ConvBlock(b * 4, b * 4),
        )

        self.mu_head = nn.Conv2d(b * 4, latent_channels, kernel_size=1)
        self.logvar_head = nn.Conv2d(b * 4, latent_channels, kernel_size=1)

        self.decoder_in = ConvBlock(latent_channels, b * 4)
        self.decoder = nn.Sequential(
            UpBlock(b * 4, b * 4),           # 16 -> 32
            UpBlock(b * 4, b * 2),           # 32 -> 64
            UpBlock(b * 2, b),               # 64 -> 128
            UpBlock(b, b),                   # 128 -> 256
            ConvBlock(b, b),
        )

        self.depth_head = nn.Sequential(
            nn.Conv2d(b, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.semantic_head = nn.Conv2d(b, 6, kernel_size=3, padding=1)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_in(z)
        h = self.decoder(h)
        depth = self.depth_head(h)
        semantics = self.semantic_head(h)
        return depth, semantics

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        depth, semantics = self.decode(z)
        return depth, semantics, mu, logvar, z


def dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    dims = (0, 2, 3)
    intersection = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def semantic_composite(prob_6chw):
    """
    prob_6chw: [6, H, W]
    """
    h, w = prob_6chw.shape[1:]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)
    for i in range(6):
        p = prob_6chw[i][..., None]
        rgb += p * SEM_COLORS[i]
        weight += p
    rgb = rgb / np.maximum(weight, 1.0)
    return np.clip(rgb, 0.0, 1.0)


def compute_losses(x, outputs, pos_weight, args):
    pred_depth, pred_sem, mu, logvar, _ = outputs
    true_depth = x[:, :1]
    true_sem = x[:, 1:]

    loss_depth = F.l1_loss(pred_depth, true_depth)
    loss_bce = F.binary_cross_entropy_with_logits(
        pred_sem, true_sem, pos_weight=pos_weight
    )
    loss_dice = dice_loss(pred_sem, true_sem)

    # spatial KL
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return loss_depth, loss_bce, loss_dice, kl


def save_checkpoint(path: Path, obj: dict):
    torch.save(obj, path)
    if (not path.exists()) or path.stat().st_size == 0:
        raise RuntimeError(f"Checkpoint write failed: {path}")


def save_generated_sample(model, z, out_dir: Path, prefix: str, metadata: dict, device):
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        d, s = model.decode(z[None].to(device))
        d = d[0, 0].float().cpu().numpy()
        s = torch.sigmoid(s[0]).float().cpu().numpy()

    np.save(out_dir / f"{prefix}_depth.npy", d.astype(np.float32))
    np.savez_compressed(
        out_dir / f"{prefix}_semantics.npz",
        **{k: s[i].astype(np.float32) for i, k in enumerate(SEMANTIC_CHANNELS)}
    )
    (out_dir / f"{prefix}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return d, s


def latent_circle_mask(z_shape, radius_ratio=0.25):
    """
    z_shape: [C, H, W]
    回傳 shape [1, C, H, W] 的 binary mask
    """
    c, h, w = z_shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = min(h, w) * radius_ratio

    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask2d = (dist <= radius).astype(np.float32)

    mask = np.repeat(mask2d[None, ...], c, axis=0)
    return torch.from_numpy(mask)[None, ...]


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Device: cuda {torch.cuda.get_device_name(0)}")
    else:
        print("Device: cpu")

    work_dir = Path(args.work_dir)
    out_dir = Path(args.output)
    ckpt_dir = Path(args.checkpoint_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # unzip dataset
    with zipfile.ZipFile(args.data_zip, "r") as z:
        z.extractall(work_dir)

    data_root_candidates = [
        work_dir / "training_taiwan_zip_spatial_v2",
        work_dir / "training_taiwan_zip_prototype",
    ]
    data_root = None
    for p in data_root_candidates:
        if p.exists():
            data_root = p
            break
    if data_root is None:
        raise FileNotFoundError("Cannot locate extracted dataset folder.")

    summary_path_candidates = [
        data_root / "dataset_summary.json",
        data_root / "prototype_dataset_summary.json",
    ]
    summary_path = None
    for p in summary_path_candidates:
        if p.exists():
            summary_path = p
            break
    if summary_path is None:
        raise FileNotFoundError("Cannot locate dataset summary JSON.")

    dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(dataset_summary, ensure_ascii=False, indent=2))

    train_ds = NPZDataset(data_root / "train.npz", augment=True)
    val_ds = NPZDataset(data_root / "val.npz", augment=False)
    test_ds = NPZDataset(data_root / "test.npz", augment=False)
    reg_ds = NPZDataset(data_root / "regression.npz", augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    print("samples:", len(train_ds), len(val_ds), len(test_ds), len(reg_ds))

    # class balancing
    frac = dataset_summary["semantic_positive_fraction_train"]
    posw = []
    for key in SEMANTIC_CHANNELS:
        p = max(float(frac[key]), 1e-6)
        weight = np.clip((1.0 - p) / p, 1.0, 20.0)
        posw.append(float(weight))
    pos_weight = torch.tensor(posw, device=device).view(1, 6, 1, 1)
    print("pos_weight:", dict(zip(SEMANTIC_CHANNELS, posw)))

    model = TaiwanZipSpatialVAEV2(
        in_channels=7,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print("params:", num_params)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history = {
        "train_total": [],
        "val_total": [],
        "train_depth": [],
        "train_bce": [],
        "train_dice": [],
        "train_kl": [],
        "beta": [],
    }

    best_val = float("inf")
    best_epoch = None
    bad_epochs = 0

    best_ckpt = ckpt_dir / "taiwan_zip_spatial_v2_best.pt"
    last_ckpt = ckpt_dir / "taiwan_zip_spatial_v2_last.pt"

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        beta = args.kl_beta * min(1.0, epoch / max(args.kl_warmup, 1))
        model.train()

        train_sums = np.zeros(5, dtype=np.float64)
        train_n = 0

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(x)
                ld, lb, ldi, lkl = compute_losses(x, outputs, pos_weight, args)
                total = (
                    args.depth_weight * ld
                    + args.bce_weight * lb
                    + args.dice_weight * ldi
                    + beta * lkl
                )

            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = x.size(0)
            train_n += bs
            train_sums += np.array(
                [
                    float(total.detach()),
                    float(ld.detach()),
                    float(lb.detach()),
                    float(ldi.detach()),
                    float(lkl.detach()),
                ]
            ) * bs

        train_mean = train_sums / max(train_n, 1)

        model.eval()
        val_total = 0.0
        val_n = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    outputs = model(x)
                    ld, lb, ldi, lkl = compute_losses(x, outputs, pos_weight, args)
                    total = (
                        args.depth_weight * ld
                        + args.bce_weight * lb
                        + args.dice_weight * ldi
                        + beta * lkl
                    )
                val_total += float(total) * x.size(0)
                val_n += x.size(0)

        val_mean = val_total / max(val_n, 1)

        history["train_total"].append(float(train_mean[0]))
        history["val_total"].append(float(val_mean))
        history["train_depth"].append(float(train_mean[1]))
        history["train_bce"].append(float(train_mean[2]))
        history["train_dice"].append(float(train_mean[3]))
        history["train_kl"].append(float(train_mean[4]))
        history["beta"].append(float(beta))

        ck = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "args": vars(args),
            "channels": CHANNELS,
            "dataset_summary": dataset_summary,
        }
        save_checkpoint(last_ckpt, ck)

        improved = (best_epoch is None) or (val_mean < (best_val - args.min_delta))
        if improved:
            best_val = float(val_mean)
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(best_ckpt, ck)
            print(f"  -> BEST saved: epoch={epoch}, val={best_val:.4f}")
        else:
            bad_epochs += 1

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train={train_mean[0]:.4f} "
            f"val={val_mean:.4f} "
            f"depth={train_mean[1]:.4f} "
            f"bce={train_mean[2]:.4f} "
            f"dice={train_mean[3]:.4f} "
            f"kl={train_mean[4]:.4f} "
            f"beta={beta:.6f} "
            f"patience={bad_epochs}/{args.patience}"
        )

        if bad_epochs >= args.patience:
            print(f"EARLY STOP at epoch {epoch}; best epoch={best_epoch}, best val={best_val:.4f}")
            break

    if not best_ckpt.exists():
        raise RuntimeError(f"Best checkpoint missing: {best_ckpt}")

    shutil.copy2(best_ckpt, out_dir / "taiwan_zip_spatial_v2_best.pt")
    shutil.copy2(last_ckpt, out_dir / "taiwan_zip_spatial_v2_last.pt")

    ck = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    # save history
    (out_dir / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # training curve
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_total"], label="train")
    plt.plot(history["val_total"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curve.png", dpi=180)
    plt.close()

    # reconstruction grid
    nrec = min(4, len(reg_ds))
    fig, axes = plt.subplots(nrec, 4, figsize=(12, 3 * nrec))
    axes = np.atleast_2d(axes)

    with torch.no_grad():
        for r in range(nrec):
            item = reg_ds[r]
            x = item["x"][None].to(device)
            pred_depth, pred_sem, mu, logvar, z = model(x)
            pred_sem_prob = torch.sigmoid(pred_sem[0]).cpu().numpy()

            axes[r, 0].imshow(x[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[r, 0].set_title(f"{item['sample_id']}\ninput depth")
            axes[r, 1].imshow(semantic_composite(x[0, 1:].cpu().numpy()))
            axes[r, 1].set_title("input semantics")
            axes[r, 2].imshow(pred_depth[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[r, 2].set_title("recon depth")
            axes[r, 3].imshow(semantic_composite(pred_sem_prob))
            axes[r, 3].set_title("recon semantics")

            for c in range(4):
                axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "reconstruction_grid.png", dpi=180)
    plt.close()

    # choose two regression samples
    def encode_mu(ds, idx):
        item = ds[idx]
        x = item["x"][None].to(device)
        with torch.no_grad():
            mu, _ = model.encode(x)
        return item, mu[0]

    item_a, z_a = encode_mu(reg_ds, 0)
    item_b, z_b = encode_mu(reg_ds, min(1, len(reg_ds) - 1))

    pair_meta = {
        "sample_a": item_a["sample_id"],
        "sample_b": item_b["sample_id"],
        "latent_shape": list(z_a.shape),
    }
    (out_dir / "sample_pair_metadata.json").write_text(
        json.dumps(pair_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # interpolation
    interp_dir = out_dir / "generated_interpolation"
    interp_results = []
    for i, t in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        z = z_a * (1.0 - t) + z_b * t
        d, s = save_generated_sample(
            model,
            z,
            interp_dir,
            f"{i:02d}",
            {
                "type": "interpolation",
                "sample_a": item_a["sample_id"],
                "sample_b": item_b["sample_id"],
                "t": t,
                "depth_note": "0=farther, 1=nearer",
            },
            device,
        )
        interp_results.append((t, d, s))

    fig, axes = plt.subplots(2, len(interp_results), figsize=(15, 6))
    for i, (t, d, s) in enumerate(interp_results):
        axes[0, i].imshow(d, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"t={t}")
        axes[1, i].imshow(semantic_composite(s))
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "latent_interpolation_grid.png", dpi=200)
    plt.close()

    # extrapolation
    extra_dir = out_dir / "generated_extrapolation"
    extra_results = []
    direction = z_b - z_a
    for i, t in enumerate([-0.5, 0.0, 0.5, 1.0, 1.5]):
        z = z_a + t * direction
        d, s = save_generated_sample(
            model,
            z,
            extra_dir,
            f"{i:02d}",
            {
                "type": "extrapolation",
                "sample_a": item_a["sample_id"],
                "sample_b": item_b["sample_id"],
                "t": t,
                "depth_note": "0=farther, 1=nearer",
            },
            device,
        )
        extra_results.append((t, d, s))

    fig, axes = plt.subplots(2, len(extra_results), figsize=(15, 6))
    for i, (t, d, s) in enumerate(extra_results):
        axes[0, i].imshow(d, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"t={t}")
        axes[1, i].imshow(semantic_composite(s))
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "latent_extrapolation_grid.png", dpi=200)
    plt.close()

    # local perturbation
    pert_dir = out_dir / "generated_local_perturbation"
    pert_results = []
    mask = latent_circle_mask(z_a.shape, radius_ratio=0.25).to(device)  # [1,C,H,W]

    for i, sigma in enumerate([0.0, 0.15, 0.30, 0.50]):
        z = z_a[None].clone().to(device)
        if sigma > 0:
            noise = torch.randn_like(z) * sigma
            z = z + noise * mask
        z0 = z[0]
        d, s = save_generated_sample(
            model,
            z0,
            pert_dir,
            f"{i:02d}",
            {
                "type": "local_perturbation",
                "source_sample": item_a["sample_id"],
                "sigma": sigma,
                "mask": "center_circle_radius_0.25",
                "depth_note": "0=farther, 1=nearer",
            },
            device,
        )
        pert_results.append((sigma, d, s))

    fig, axes = plt.subplots(2, len(pert_results), figsize=(12, 6))
    for i, (sigma, d, s) in enumerate(pert_results):
        axes[0, i].imshow(d, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"sigma={sigma}")
        axes[1, i].imshow(semantic_composite(s))
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "latent_local_perturbation_grid.png", dpi=200)
    plt.close()

    run_summary = {
        "version": "taiwan-zip-spatial-vae-v2",
        "best_epoch": best_epoch,
        "best_val": best_val,
        "epochs_completed": len(history["train_total"]),
        "runtime_seconds": time.time() - t0,
        "device": str(device),
        "latent_shape": [args.latent_channels, 16, 16],
        "base_channels": args.base_channels,
        "channels": CHANNELS,
        "dataset_summary": dataset_summary,
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("DONE. Results:", out_dir)


if __name__ == "__main__":
    main()