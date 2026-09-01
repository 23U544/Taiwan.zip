#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Playing Models — Taiwan Streetscape Candidate Screener V2
==========================================================

Quality-focused second-stage screener.

Input
-----
Use the `screened_candidates.csv` produced by Screener V1.1.
This script REUSES its cached thumbnails. It does not download images.

Why V2 exists
-------------
V1.1 proved that a single "streetscape vs non-streetscape" CLIP score is useful,
but some aerial views, isolated landmarks, temples, public buildings, and
object-centric scenes can still receive very high scores.

V2 therefore evaluates several independent CLIP axes:

1. baseline_street      — reuse V1.1 score
2. eye_level            — street/pedestrian viewpoint vs aerial/high viewpoint
3. spatial_context      — facade + street context vs isolated landmark/detail
4. playing_models_value — useful urban spatial configuration vs object/event/railway/interior

The axes are combined into a final suitability score, but all component scores
are exported so thresholds remain auditable.

This is a corpus triage tool, not ground-truth annotation.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

VERSION = "2.0"

EYE_LEVEL_POS = [
    "an eye-level street photograph viewed from pedestrian height",
    "a street-level photograph looking along a road or alley",
    "a frontal building facade photographed from the sidewalk or street",
    "an urban street scene photographed from ground level",
    "a narrow alley viewed from ground level between buildings",
]
EYE_LEVEL_NEG = [
    "an aerial bird's-eye view of a city",
    "a skyline panorama photographed from a high viewpoint",
    "a drone photograph looking down at buildings",
    "a rooftop or observation-deck view looking down over a city",
    "a distant landscape panorama without ground-level street perspective",
]

SPATIAL_CONTEXT_POS = [
    "a street scene showing neighboring building facades together",
    "a building facade together with sidewalk road alley or storefront context",
    "a Taiwanese urban scene with architecture signs scooters and public space together",
    "a street edge with multiple architectural and attached elements visible",
    "a residential or commercial street where spatial relationships between buildings and street are visible",
]
SPATIAL_CONTEXT_NEG = [
    "a close-up architectural detail with no surrounding street context",
    "a single isolated monument or landmark filling the photograph",
    "a close-up temple shrine gate or religious object without surrounding street context",
    "a park garden plaza or open field with little street-edge architecture",
    "a single public building photographed as an isolated object with little surrounding urban context",
]

PLAYING_MODELS_POS = [
    "a complex Taiwanese streetscape with facades windows signs air conditioners scooters and street elements",
    "an urban street image useful for studying relationships between architecture and street objects",
    "a mixed-use street facade with storefronts signs balconies windows and attached elements",
    "a residential alley with layered building facades utilities vehicles and everyday urban objects",
    "a street-level spatial configuration with multiple semantic urban elements",
]
PLAYING_MODELS_NEG = [
    "a close-up photograph of a single object product tool food plant or animal",
    "an indoor room exhibition classroom lobby or religious interior",
    "a train railway platform rail track or railway vehicle as the main subject",
    "a crowd ceremony performance portrait or event as the main subject",
    "a natural landscape ocean mountain park or garden with little urban street structure",
    "a sign poster artwork document diagram or information board as the main subject",
]

# An explicit "hard nuisance" axis catches the strongest unwanted image families.
NUISANCE_POS = [
    "a useful Taiwanese street-level urban scene with buildings and street context",
    "a useful street-level building facade with surrounding urban context",
]
NUISANCE_NEG = [
    "an aerial city panorama",
    "an indoor scene",
    "a close-up of an object",
    "a railway or train photograph",
    "a ceremony crowd or event photograph",
    "a landscape nature photograph",
    "an isolated temple shrine monument or landmark close-up",
]


def parse_args():
    p = argparse.ArgumentParser(description="Second-stage CLIP quality screener using cached V1.1 thumbnails.")
    p.add_argument("--input-screened-csv", required=True,
                   help="V1.1 screened_candidates.csv containing screen_thumbnail_path and clip_street_probability.")
    p.add_argument("--output", default="taiwan_streetscape_screening_v2")
    p.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    p.add_argument("--clip-local-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--limit", type=int, default=0)

    # Final triage thresholds. These are deliberately conservative defaults.
    p.add_argument("--keep-score", type=float, default=0.72)
    p.add_argument("--review-score", type=float, default=0.48)
    p.add_argument("--min-eye-keep", type=float, default=0.55)
    p.add_argument("--min-context-keep", type=float, default=0.45)
    p.add_argument("--min-value-keep", type=float, default=0.45)
    p.add_argument("--severe-nuisance-threshold", type=float, default=0.78,
                   help="If nuisance probability is this high, a candidate cannot be auto-kept.")

    p.add_argument("--contact-sheet-count", type=int, default=36)
    p.add_argument("--thumbnail-box", type=int, default=260)
    return p.parse_args()


def feature_tensor(x):
    if hasattr(x, "image_embeds") and x.image_embeds is not None:
        return x.image_embeds
    if hasattr(x, "text_embeds") and x.text_embeds is not None:
        return x.text_embeds
    if hasattr(x, "pooler_output") and x.pooler_output is not None:
        return x.pooler_output
    if isinstance(x, tuple):
        return x[0]
    return x


def pair_probability(sim_row, pos_slice, neg_slice):
    """Softmax only within a concept axis, then sum probability over positive prompts."""
    vals = np.concatenate([sim_row[pos_slice], sim_row[neg_slice]]) * 100.0
    vals = vals - vals.max()
    exp = np.exp(vals)
    prob = exp / max(exp.sum(), 1e-12)
    return float(prob[: pos_slice.stop - pos_slice.start].sum())


def load_model(args):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = CLIPProcessor.from_pretrained(
        args.clip_model,
        local_files_only=args.clip_local_only,
    )
    model = CLIPModel.from_pretrained(
        args.clip_model,
        local_files_only=args.clip_local_only,
    ).to(device).eval()
    return torch, processor, model, device


def score(df, args):
    torch, processor, model, device = load_model(args)

    groups = [
        ("eye_pos", EYE_LEVEL_POS),
        ("eye_neg", EYE_LEVEL_NEG),
        ("context_pos", SPATIAL_CONTEXT_POS),
        ("context_neg", SPATIAL_CONTEXT_NEG),
        ("value_pos", PLAYING_MODELS_POS),
        ("value_neg", PLAYING_MODELS_NEG),
        ("nuisance_pos", NUISANCE_POS),
        ("nuisance_neg", NUISANCE_NEG),
    ]

    texts = []
    slices = {}
    cursor = 0
    for name, prompts in groups:
        start = cursor
        texts.extend(prompts)
        cursor += len(prompts)
        slices[name] = slice(start, cursor)

    text_inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    use_fp16 = bool(args.fp16 and device == "cuda")
    amp = lambda: torch.autocast(device_type="cuda", dtype=torch.float16) if use_fp16 else contextlib.nullcontext()

    with torch.inference_mode(), amp():
        tf = feature_tensor(model.get_text_features(**text_inputs))
        tf = tf / tf.norm(dim=-1, keepdim=True)

    valid_indices = []
    for i in range(len(df)):
        p = str(df.at[i, "screen_thumbnail_path"] or "")
        if p and Path(p).exists():
            valid_indices.append(i)
        else:
            df.at[i, "v2_recommendation"] = "review"
            df.at[i, "v2_reason"] = "missing_cached_thumbnail"

    total = len(valid_indices)
    for start in range(0, total, args.batch_size):
        inds = valid_indices[start:start + args.batch_size]
        images, good = [], []
        for i in inds:
            try:
                with Image.open(str(df.at[i, "screen_thumbnail_path"])) as im:
                    images.append(im.convert("RGB").copy())
                good.append(i)
            except Exception as exc:
                df.at[i, "v2_recommendation"] = "review"
                df.at[i, "v2_reason"] = f"thumbnail_open_error:{type(exc).__name__}"

        if not good:
            continue

        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode(), amp():
            imf = feature_tensor(model.get_image_features(**inputs))
            imf = imf / imf.norm(dim=-1, keepdim=True)
            sims = (imf @ tf.T).float().cpu().numpy()

        for j, idx in enumerate(good):
            s = sims[j]
            eye = pair_probability(s, slices["eye_pos"], slices["eye_neg"])
            context = pair_probability(s, slices["context_pos"], slices["context_neg"])
            value = pair_probability(s, slices["value_pos"], slices["value_neg"])
            useful_vs_nuisance = pair_probability(s, slices["nuisance_pos"], slices["nuisance_neg"])
            nuisance = 1.0 - useful_vs_nuisance

            baseline = pd.to_numeric(
                pd.Series([df.at[idx, "clip_street_probability"]]), errors="coerce"
            ).iloc[0]
            if pd.isna(baseline):
                baseline = 0.5
            baseline = float(np.clip(baseline, 0.0, 1.0))

            # Baseline still matters, but cannot dominate a clear aerial/object/landmark failure.
            final = (
                0.35 * baseline
                + 0.25 * eye
                + 0.20 * context
                + 0.20 * value
            )

            # Explicit penalties for strong nuisance evidence and high-viewpoint failures.
            if nuisance > 0.80:
                final -= 0.16 * (nuisance - 0.80) / 0.20
            if eye < 0.25:
                final -= 0.12 * (0.25 - eye) / 0.25
            final = float(np.clip(final, 0.0, 1.0))

            df.at[idx, "v2_eye_level_score"] = eye
            df.at[idx, "v2_spatial_context_score"] = context
            df.at[idx, "v2_playing_models_value_score"] = value
            df.at[idx, "v2_nuisance_probability"] = nuisance
            df.at[idx, "v2_final_score"] = final

            reasons = []
            severe_nuisance = nuisance >= args.severe_nuisance_threshold
            if eye < args.min_eye_keep:
                reasons.append("weak_eye_level")
            if context < args.min_context_keep:
                reasons.append("weak_spatial_context")
            if value < args.min_value_keep:
                reasons.append("weak_playing_models_value")
            if severe_nuisance:
                reasons.append("strong_nuisance_evidence")

            if (
                final >= args.keep_score
                and eye >= args.min_eye_keep
                and context >= args.min_context_keep
                and value >= args.min_value_keep
                and not severe_nuisance
            ):
                rec = "keep"
                reason = "strong_multiaxis_streetscape"
            elif (
                final >= args.review_score
                or baseline >= 0.80
                or (eye >= 0.50 and value >= 0.45)
            ):
                rec = "review"
                reason = ",".join(reasons) if reasons else "borderline_multiaxis"
            else:
                rec = "reject"
                reason = ",".join(reasons) if reasons else "low_multiaxis_score"

            # Never auto-keep V1.1 exact duplicates/errors if those fields exist.
            old_dup = str(df.at[idx, "screen_duplicate_reason"] or "") if "screen_duplicate_reason" in df.columns else ""
            old_err = str(df.at[idx, "screen_error"] or "") if "screen_error" in df.columns else ""
            if old_dup:
                rec = "review" if rec == "keep" else rec
                reason += ",v1_duplicate_flag"
            if old_err:
                rec = "review"
                reason += ",v1_screen_error"

            df.at[idx, "v2_recommendation"] = rec
            df.at[idx, "v2_reason"] = reason

        del inputs, imf, sims
        done = min(start + args.batch_size, total)
        print(f"[V2 CLIP] {done}/{total} batch={len(good)} fp16={use_fp16}")

    del model, processor, tf, text_inputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return df


def truncate(text, n=38):
    s = str(text or "").replace("\n", " ")
    return s if len(s) <= n else s[:n-1] + "…"


def make_sheet(rows, path, title, score_col, count, box):
    if rows.empty:
        return
    rows = rows.head(count)
    cols = 6
    cell_w = box
    cell_h = box + 76
    out = Image.new("RGB", (cols * cell_w, math.ceil(len(rows) / cols) * cell_h + 36), "white")
    draw = ImageDraw.Draw(out)
    draw.text((8, 8), title, fill="black")

    for n, (_, r) in enumerate(rows.iterrows()):
        x = (n % cols) * cell_w
        y = (n // cols) * cell_h + 36
        try:
            with Image.open(str(r["screen_thumbnail_path"])) as src:
                im = src.convert("RGB")
            im.thumbnail((box - 8, box - 8), Image.Resampling.BILINEAR)
            canvas = Image.new("RGB", (box - 8, box - 8), "#e6e6e6")
            canvas.paste(im, ((canvas.width-im.width)//2, (canvas.height-im.height)//2))
            out.paste(canvas, (x+4, y+4))
        except Exception:
            pass

        sc = pd.to_numeric(pd.Series([r.get(score_col)]), errors="coerce").iloc[0]
        sc_text = "nan" if pd.isna(sc) else f"{float(sc):.3f}"
        old = pd.to_numeric(pd.Series([r.get("clip_street_probability")]), errors="coerce").iloc[0]
        old_text = "nan" if pd.isna(old) else f"{float(old):.2f}"
        city = str(r.get("guessed_city") or "")
        eye = r.get("v2_eye_level_score", np.nan)
        ctx = r.get("v2_spatial_context_score", np.nan)
        val = r.get("v2_playing_models_value_score", np.nan)
        draw.text(
            (x+6, y+box+3),
            f"V2 {sc_text} | old {old_text} | {city}\n"
            f"E {float(eye):.2f} C {float(ctx):.2f} V {float(val):.2f}\n"
            f"{truncate(r.get('title',''))}",
            fill="black",
        )
    out.save(path, quality=88)


def make_contact_sheets(df, out_dir, args):
    scored = df[pd.to_numeric(df["v2_final_score"], errors="coerce").notna()].copy()
    scored["v2_final_score"] = pd.to_numeric(scored["v2_final_score"], errors="coerce")
    scored["clip_street_probability"] = pd.to_numeric(scored["clip_street_probability"], errors="coerce")

    make_sheet(
        scored.sort_values("v2_final_score", ascending=False),
        out_dir / "v2_contact_sheet_top.jpg",
        "V2 highest final suitability",
        "v2_final_score", args.contact_sheet_count, args.thumbnail_box,
    )

    keep_cut = args.keep_score
    border = scored.iloc[(scored["v2_final_score"] - keep_cut).abs().argsort()]
    make_sheet(
        border,
        out_dir / "v2_contact_sheet_keep_boundary.jpg",
        "V2 around KEEP boundary",
        "v2_final_score", args.contact_sheet_count, args.thumbnail_box,
    )

    review_cut = args.review_score
    border2 = scored.iloc[(scored["v2_final_score"] - review_cut).abs().argsort()]
    make_sheet(
        border2,
        out_dir / "v2_contact_sheet_review_boundary.jpg",
        "V2 around REVIEW boundary",
        "v2_final_score", args.contact_sheet_count, args.thumbnail_box,
    )

    make_sheet(
        scored.sort_values("v2_final_score", ascending=True),
        out_dir / "v2_contact_sheet_bottom.jpg",
        "V2 lowest final suitability",
        "v2_final_score", args.contact_sheet_count, args.thumbnail_box,
    )

    # Most important diagnostic: V1.1 was highly confident, V2 says not keep.
    conflict = scored[
        (scored["clip_street_probability"] >= 0.90)
        & (scored["v2_recommendation"] != "keep")
    ].sort_values("clip_street_probability", ascending=False)
    make_sheet(
        conflict,
        out_dir / "v2_contact_sheet_high_v1_conflicts.jpg",
        "High V1.1 score but V2 does not KEEP",
        "v2_final_score", args.contact_sheet_count, args.thumbnail_box,
    )


def write_report(df, out_dir, args, elapsed):
    rec = df["v2_recommendation"].fillna("unprocessed").value_counts().to_dict()
    score = pd.to_numeric(df["v2_final_score"], errors="coerce")
    eye = pd.to_numeric(df["v2_eye_level_score"], errors="coerce")
    context = pd.to_numeric(df["v2_spatial_context_score"], errors="coerce")
    value = pd.to_numeric(df["v2_playing_models_value_score"], errors="coerce")
    nuisance = pd.to_numeric(df["v2_nuisance_probability"], errors="coerce")

    summary = {
        "version": VERSION,
        "rows": int(len(df)),
        "recommendations": {str(k): int(v) for k, v in rec.items()},
        "thresholds": {
            "keep_score": args.keep_score,
            "review_score": args.review_score,
            "min_eye_keep": args.min_eye_keep,
            "min_context_keep": args.min_context_keep,
            "min_value_keep": args.min_value_keep,
            "severe_nuisance_threshold": args.severe_nuisance_threshold,
        },
        "score_distribution": {
            "final_median": float(score.median()),
            "final_p25": float(score.quantile(.25)),
            "final_p75": float(score.quantile(.75)),
            "eye_median": float(eye.median()),
            "context_median": float(context.median()),
            "value_median": float(value.median()),
            "nuisance_median": float(nuisance.median()),
        },
        "performance": {"clip_rescore_seconds": float(elapsed)},
    }
    (out_dir / "v2_screening_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if "guessed_city" in df.columns:
        df.groupby("guessed_city")["v2_recommendation"].value_counts().unstack(fill_value=0).to_csv(
            out_dir / "v2_city_distribution.csv", encoding="utf-8-sig"
        )

    md = [
        "# Taiwan Streetscape Screening V2",
        "",
        f"- Rows: {len(df)}",
        f"- KEEP: {rec.get('keep', 0)}",
        f"- REVIEW: {rec.get('review', 0)}",
        f"- REJECT: {rec.get('reject', 0)}",
        f"- CLIP rescore seconds: {elapsed:.2f}",
        "",
        "## Purpose",
        "",
        "V2 is a multi-axis quality triage layer. It reuses V1.1 cached thumbnails and adds",
        "eye-level viewpoint, spatial-context, Playing Models usefulness, and nuisance evidence.",
        "It does not download or alter source images.",
        "",
        "## Median component scores",
        "",
        f"- final: {score.median():.4f}",
        f"- eye_level: {eye.median():.4f}",
        f"- spatial_context: {context.median():.4f}",
        f"- playing_models_value: {value.median():.4f}",
        f"- nuisance_probability: {nuisance.median():.4f}",
        "",
        "Thresholds are triage settings, not ground truth. Inspect the boundary and conflict contact sheets before full-resolution download.",
    ]
    (out_dir / "V2_SCREENING_REPORT.md").write_text("\n".join(md), encoding="utf-8")


def main():
    args = parse_args()
    t0 = time.perf_counter()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_screened_csv)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()
    else:
        df = df.copy()
    df = df.reset_index(drop=True)

    required = ["screen_thumbnail_path", "clip_street_probability"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            "Input must be V1.1 screened_candidates.csv. Missing columns: " + ", ".join(missing)
        )

    defaults = {
        "v2_eye_level_score": np.nan,
        "v2_spatial_context_score": np.nan,
        "v2_playing_models_value_score": np.nan,
        "v2_nuisance_probability": np.nan,
        "v2_final_score": np.nan,
        "v2_recommendation": "",
        "v2_reason": "",
    }
    for col, val in defaults.items():
        df[col] = val

    df = score(df, args)
    elapsed = time.perf_counter() - t0

    df.to_csv(out_dir / "v2_screened_candidates.csv", index=False, encoding="utf-8-sig")
    df[df["v2_recommendation"] == "keep"].to_csv(out_dir / "v2_keep_candidates.csv", index=False, encoding="utf-8-sig")
    df[df["v2_recommendation"] == "review"].to_csv(out_dir / "v2_review_candidates.csv", index=False, encoding="utf-8-sig")
    df[df["v2_recommendation"] == "reject"].to_csv(out_dir / "v2_reject_candidates.csv", index=False, encoding="utf-8-sig")

    make_contact_sheets(df, out_dir, args)
    write_report(df, out_dir, args, elapsed)

    summary = json.loads((out_dir / "v2_screening_summary.json").read_text(encoding="utf-8"))
    print("DONE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
