#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Playing Models — Taiwan Streetscape Screening V2.1 Reclassifier
================================================================

Purpose
-------
Repair the decision layer of Screening V2 WITHOUT rerunning CLIP and WITHOUT
downloading thumbnails again.

V2's component ranking is useful, but its `nuisance_probability` hard veto is
miscalibrated: many excellent street scenes receive high nuisance probability,
which caused all 2000 images to become REVIEW.

V2.1 therefore:
- reuses V2 component scores already stored in `v2_screened_candidates.csv`
- completely removes nuisance_probability from acceptance/rejection decisions
- uses a transparent four-tier policy:
    STRONG_KEEP / KEEP / REVIEW / REJECT
- generates boundary/conflict contact sheets for calibration
- does not modify source images or V2 outputs

This is still corpus triage, not ground-truth annotation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

VERSION = "2.1"


def parse_args():
    p = argparse.ArgumentParser(
        description="Reclassify Screening V2 scores without rerunning CLIP."
    )
    p.add_argument("--input-v2-csv", required=True)
    p.add_argument("--output", default="taiwan_streetscape_screening_v2_1")

    # Strong keep = highly reliable automatic corpus candidate.
    p.add_argument("--strong-keep-score", type=float, default=0.72)
    p.add_argument("--strong-min-eye", type=float, default=0.45)
    p.add_argument("--strong-min-context", type=float, default=0.30)
    p.add_argument("--strong-min-value", type=float, default=0.30)

    # Keep = likely useful, but slightly looser than Strong Keep.
    p.add_argument("--keep-score", type=float, default=0.62)
    p.add_argument("--keep-min-baseline", type=float, default=0.55)
    p.add_argument("--keep-min-eye", type=float, default=0.38)
    p.add_argument("--keep-min-context-or-value", type=float, default=0.32)

    # Review / reject boundary.
    p.add_argument("--review-score", type=float, default=0.32)
    p.add_argument("--review-baseline", type=float, default=0.43)

    p.add_argument("--contact-sheet-count", type=int, default=36)
    p.add_argument("--thumbnail-box", type=int, default=260)
    return p.parse_args()


def num(x, default=np.nan):
    try:
        value = float(x)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def classify_row(r, args):
    final = num(r.get("v2_final_score"))
    eye = num(r.get("v2_eye_level_score"))
    context = num(r.get("v2_spatial_context_score"))
    value = num(r.get("v2_playing_models_value_score"))
    baseline = num(r.get("clip_street_probability"), 0.0)

    if not all(np.isfinite(v) for v in [final, eye, context, value]):
        return "review", "missing_v2_component_score"

    # V2.1 intentionally DOES NOT use v2_nuisance_probability.
    if (
        final >= args.strong_keep_score
        and eye >= args.strong_min_eye
        and context >= args.strong_min_context
        and value >= args.strong_min_value
    ):
        return "strong_keep", "high_multiaxis_spatial_quality"

    if (
        final >= args.keep_score
        and baseline >= args.keep_min_baseline
        and eye >= args.keep_min_eye
        and max(context, value) >= args.keep_min_context_or_value
    ):
        return "keep", "good_spatial_quality"

    if final >= args.review_score or baseline >= args.review_baseline:
        reasons = []
        if final < args.keep_score:
            reasons.append("borderline_final_score")
        if eye < args.keep_min_eye:
            reasons.append("weak_eye_level")
        if context < args.keep_min_context_or_value:
            reasons.append("weak_spatial_context")
        if value < args.keep_min_context_or_value:
            reasons.append("weak_playing_models_value")
        return "review", ",".join(reasons) if reasons else "borderline"

    reasons = []
    if eye < 0.30:
        reasons.append("low_eye_level")
    if context < 0.22:
        reasons.append("low_spatial_context")
    if value < 0.22:
        reasons.append("low_playing_models_value")
    if baseline < args.review_baseline:
        reasons.append("low_v1_street_score")
    return "reject", ",".join(reasons) if reasons else "low_combined_quality"


def truncate(text, n=38):
    s = str(text or "").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def make_sheet(rows, path, title, count, box):
    if rows.empty:
        return
    rows = rows.head(count)
    cols = 6
    cell_w = box
    cell_h = box + 78
    canvas_h = math.ceil(len(rows) / cols) * cell_h + 36
    out = Image.new("RGB", (cols * cell_w, canvas_h), "white")
    draw = ImageDraw.Draw(out)
    draw.text((8, 8), title, fill="black")

    for n, (_, r) in enumerate(rows.iterrows()):
        x = (n % cols) * cell_w
        y = (n // cols) * cell_h + 36
        thumb = str(r.get("screen_thumbnail_path") or "")
        try:
            with Image.open(thumb) as src:
                im = src.convert("RGB")
            im.thumbnail((box - 8, box - 8), Image.Resampling.BILINEAR)
            bg = Image.new("RGB", (box - 8, box - 8), "#e8e8e8")
            bg.paste(im, ((bg.width - im.width) // 2, (bg.height - im.height) // 2))
            out.paste(bg, (x + 4, y + 4))
        except Exception:
            pass

        final = num(r.get("v2_final_score"), 0)
        old = num(r.get("clip_street_probability"), 0)
        eye = num(r.get("v2_eye_level_score"), 0)
        ctx = num(r.get("v2_spatial_context_score"), 0)
        val = num(r.get("v2_playing_models_value_score"), 0)
        city = str(r.get("guessed_city") or "")
        tier = str(r.get("v2_1_recommendation") or "")

        draw.text(
            (x + 6, y + box + 3),
            f"{tier} | V2 {final:.2f} old {old:.2f} | {city}\n"
            f"E {eye:.2f} C {ctx:.2f} V {val:.2f}\n"
            f"{truncate(r.get('title', ''))}",
            fill="black",
        )
    out.save(path, quality=88)


def write_outputs(df, out_dir, args):
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "v2_1_screened_candidates.csv", index=False, encoding="utf-8-sig")

    file_map = {
        "strong_keep": "v2_1_strong_keep_candidates.csv",
        "keep": "v2_1_keep_candidates.csv",
        "review": "v2_1_review_candidates.csv",
        "reject": "v2_1_reject_candidates.csv",
    }
    for tier, name in file_map.items():
        df[df["v2_1_recommendation"] == tier].to_csv(
            out_dir / name, index=False, encoding="utf-8-sig"
        )

    counts = df["v2_1_recommendation"].value_counts().to_dict()

    # City × tier distribution.
    if "guessed_city" in df.columns:
        df.groupby("guessed_city")["v2_1_recommendation"].value_counts().unstack(
            fill_value=0
        ).to_csv(out_dir / "v2_1_city_distribution.csv", encoding="utf-8-sig")

    # Diagnostics around each decision boundary.
    scored = df[pd.to_numeric(df["v2_final_score"], errors="coerce").notna()].copy()
    scored["v2_final_score"] = pd.to_numeric(scored["v2_final_score"], errors="coerce")
    scored["clip_street_probability"] = pd.to_numeric(
        scored["clip_street_probability"], errors="coerce"
    )

    strong = scored[scored["v2_1_recommendation"] == "strong_keep"].sort_values(
        "v2_final_score", ascending=False
    )
    make_sheet(
        strong,
        out_dir / "v2_1_contact_sheet_strong_keep.jpg",
        "V2.1 STRONG KEEP",
        args.contact_sheet_count,
        args.thumbnail_box,
    )

    keep = scored[scored["v2_1_recommendation"].isin(["strong_keep", "keep"])].copy()
    keep_boundary = keep.iloc[
        (keep["v2_final_score"] - args.keep_score).abs().argsort()
    ]
    make_sheet(
        keep_boundary,
        out_dir / "v2_1_contact_sheet_keep_boundary.jpg",
        "V2.1 around KEEP boundary",
        args.contact_sheet_count,
        args.thumbnail_box,
    )

    review = scored[scored["v2_1_recommendation"] == "review"].copy()
    review_boundary = review.iloc[
        (review["v2_final_score"] - args.review_score).abs().argsort()
    ]
    make_sheet(
        review_boundary,
        out_dir / "v2_1_contact_sheet_review_boundary.jpg",
        "V2.1 around REVIEW boundary",
        args.contact_sheet_count,
        args.thumbnail_box,
    )

    reject = scored[scored["v2_1_recommendation"] == "reject"].sort_values(
        "v2_final_score", ascending=True
    )
    make_sheet(
        reject,
        out_dir / "v2_1_contact_sheet_reject.jpg",
        "V2.1 REJECT",
        args.contact_sheet_count,
        args.thumbnail_box,
    )

    # Key regression test: old V1.1 high-confidence images should mostly be retained
    # if V2 spatial scores are genuinely strong.
    old_high = scored[scored["clip_street_probability"] >= 0.90].copy()
    old_high_bad = old_high[
        ~old_high["v2_1_recommendation"].isin(["strong_keep", "keep"])
    ].sort_values("clip_street_probability", ascending=False)
    make_sheet(
        old_high_bad,
        out_dir / "v2_1_contact_sheet_high_v1_not_kept.jpg",
        "High V1.1 confidence but V2.1 still not kept",
        args.contact_sheet_count,
        args.thumbnail_box,
    )

    nuisance = pd.to_numeric(df.get("v2_nuisance_probability"), errors="coerce")
    final = pd.to_numeric(df["v2_final_score"], errors="coerce")
    eye = pd.to_numeric(df["v2_eye_level_score"], errors="coerce")
    ctx = pd.to_numeric(df["v2_spatial_context_score"], errors="coerce")
    val = pd.to_numeric(df["v2_playing_models_value_score"], errors="coerce")

    summary = {
        "version": VERSION,
        "rows": int(len(df)),
        "recommendations": {k: int(v) for k, v in counts.items()},
        "usable_without_manual_review": int(
            counts.get("strong_keep", 0) + counts.get("keep", 0)
        ),
        "thresholds": {
            "strong_keep_score": args.strong_keep_score,
            "keep_score": args.keep_score,
            "review_score": args.review_score,
        },
        "score_distribution": {
            "final_median": float(final.median()),
            "eye_median": float(eye.median()),
            "context_median": float(ctx.median()),
            "value_median": float(val.median()),
            "nuisance_median_diagnostic_only": float(nuisance.median())
            if nuisance.notna().any()
            else None,
        },
        "decision_policy": {
            "uses_nuisance_probability": False,
            "reason": (
                "V2 nuisance probability was empirically miscalibrated and "
                "incorrectly vetoed high-quality street scenes."
            ),
        },
    }

    (out_dir / "v2_1_screening_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        "# Taiwan Streetscape Screening V2.1",
        "",
        "V2.1 repairs the V2 decision layer without rerunning CLIP.",
        "",
        "## Decision",
        "",
        "The V2 `nuisance_probability` is retained only for diagnostics and is not used",
        "for KEEP/REVIEW/REJECT decisions because the full 2000-image run showed it",
        "systematically vetoing excellent street scenes.",
        "",
        "## Counts",
        "",
        f"- STRONG_KEEP: {counts.get('strong_keep', 0)}",
        f"- KEEP: {counts.get('keep', 0)}",
        f"- REVIEW: {counts.get('review', 0)}",
        f"- REJECT: {counts.get('reject', 0)}",
        f"- Automatic usable pool: {counts.get('strong_keep', 0) + counts.get('keep', 0)}",
        "",
        "## Next action",
        "",
        "Inspect STRONG_KEEP, KEEP boundary, REVIEW boundary, REJECT, and high-V1-not-kept",
        "contact sheets before full-resolution download.",
    ]
    (out_dir / "V2_1_SCREENING_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    return summary


def main():
    args = parse_args()
    out_dir = Path(args.output)

    df = pd.read_csv(args.input_v2_csv)
    required = [
        "v2_final_score",
        "v2_eye_level_score",
        "v2_spatial_context_score",
        "v2_playing_models_value_score",
        "clip_street_probability",
        "screen_thumbnail_path",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            "Input must be V2 v2_screened_candidates.csv. Missing: "
            + ", ".join(missing)
        )

    results = df.apply(lambda r: classify_row(r, args), axis=1)
    df["v2_1_recommendation"] = [x[0] for x in results]
    df["v2_1_reason"] = [x[1] for x in results]

    summary = write_outputs(df, out_dir, args)

    print("DONE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
