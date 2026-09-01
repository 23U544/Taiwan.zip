"""Playing Models Taiwanese streetscape semantic parser V1.

The parser intentionally uses multi-label masks.  Grounding DINO decides semantic
instances, SAM2 refines accepted instance boundaries, and existing relative depth
is sampled only after masks are restored to the original RGB resolution.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoImageProcessor,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    Sam2Model,
    Sam2Processor,
    SegformerForSemanticSegmentation,
)

VERSION = "playing-models-street-parser-v1.0"
SEGFORMER_ID = "nvidia/segformer-b2-finetuned-ade-512-512"
DINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM_ID = "facebook/sam2.1-hiera-small"
CLIP_ID = "openai/clip-vit-base-patch32"

TAGS = [
    "facade", "window", "door", "balcony", "railing", "grille",
    "awning", "storefront", "rolling_shutter", "signboard",
    "air_conditioner", "utility_pole", "wire", "vegetation", "person",
    "vehicle", "street_object", "arcade_candidate",
]

CATEGORIES = {
    "facade": "architectural", "window": "architectural", "window_group": "architectural",
    "door": "architectural", "balcony": "architectural", "railing": "architectural",
    "grille": "architectural", "awning": "architectural", "storefront": "architectural",
    "rolling_shutter": "architectural", "signboard": "attached_object",
    "air_conditioner": "attached_object", "utility_pole": "attached_object",
    "wire": "attached_object", "vegetation": "environment", "person": "street_object",
    "vehicle": "street_object", "street_object": "street_object",
    "arcade_candidate": "derived_spatial",
}

PROMPTS = {
    "window": ["window", "building window", "apartment window", "residential window", "window opening"],
    "door": ["building door", "entrance door", "doorway"],
    "balcony": ["building balcony", "apartment balcony"],
    "railing": ["balcony railing", "metal railing"],
    "grille": ["window metal grille", "security grille", "metal window bars"],
    "awning": ["shop awning", "building awning", "canopy"],
    "storefront": ["storefront", "shopfront", "shop entrance"],
    "rolling_shutter": ["rolling shutter", "metal shop shutter", "garage shutter"],
    "signboard": ["shop sign", "store sign", "signboard", "commercial sign", "billboard"],
    "air_conditioner": ["air conditioner", "air conditioning unit", "outdoor AC unit"],
    "utility_pole": ["utility pole", "electric pole", "telephone pole"],
    "person": ["person", "pedestrian"],
    "vehicle": ["car", "truck", "bus", "motorcycle", "scooter", "bicycle", "vehicle"],
    "street_object": ["street furniture", "traffic sign", "traffic light", "bench", "vending machine"],
}

THRESHOLDS = {
    "window": 0.18, "door": 0.24, "balcony": 0.22, "railing": 0.22,
    "grille": 0.22, "awning": 0.24, "storefront": 0.24,
    "rolling_shutter": 0.24, "signboard": 0.22, "air_conditioner": 0.22,
    "utility_pole": 0.25, "person": 0.25, "vehicle": 0.24, "street_object": 0.27,
}

RELIABILITY = {
    "facade": "HIGH", "window": "MEDIUM", "door": "MEDIUM", "balcony": "MEDIUM",
    "railing": "MEDIUM", "grille": "EXPERIMENTAL", "awning": "MEDIUM",
    "storefront": "MEDIUM", "rolling_shutter": "MEDIUM", "signboard": "HIGH",
    "air_conditioner": "HIGH", "utility_pole": "MEDIUM", "wire": "EXPERIMENTAL",
    "vegetation": "HIGH", "person": "HIGH", "vehicle": "HIGH",
    "street_object": "MEDIUM", "arcade_candidate": "EXPERIMENTAL",
}

COLORS = {
    "facade": (50, 180, 90), "window": (30, 180, 255), "window_group": (10, 110, 240),
    "door": (180, 110, 40), "balcony": (30, 220, 190), "railing": (230, 210, 60),
    "grille": (220, 220, 220), "awning": (210, 80, 230), "storefront": (255, 150, 20),
    "rolling_shutter": (160, 100, 60), "signboard": (255, 220, 20),
    "air_conditioner": (230, 70, 60), "utility_pole": (150, 90, 40),
    "wire": (255, 255, 255), "vegetation": (20, 130, 35), "person": (255, 50, 140),
    "vehicle": (80, 220, 80), "street_object": (150, 150, 255), "arcade_candidate": (120, 60, 220),
}

SCENE_TEXTS = {
    "dense_facade": "a close frontal building facade with many visible windows",
    "street_perspective": "a street perspective with buildings, road, vehicles and signs",
    "mixed_urban": "a mixed urban streetscape with buildings, signs and street activity",
    "landscape": "a landscape scene with vegetation and little building facade",
}


def parse_args():
    p = argparse.ArgumentParser(description="Playing Models multi-label street parser V1")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene")
    group.add_argument("--scenes", nargs="+")
    group.add_argument("--all", action="store_true")
    p.add_argument("--dataset-root", default="dataset/scenes")
    p.add_argument("--output-name", default="parsing_v1")
    p.add_argument("--window-tile-sizes", nargs="+", type=int, default=[384, 512, 768])
    p.add_argument("--sam-batch-size", type=int, default=12)
    p.add_argument("--max-instances-per-tag", type=int, default=120)
    return p.parse_args()


def unload(*objects):
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter_area(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    inter = inter_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union else 0.0


def containment(a, b):
    return inter_area(a, b) / max(box_area(a), 1.0)


def center(b):
    return [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]


def nms(items, threshold=0.55):
    kept = []
    for item in sorted(items, key=lambda x: x["score"], reverse=True):
        if all(iou(item["bbox"], k["bbox"]) < threshold for k in kept):
            kept.append(item)
    return kept


def iter_tiles(width, height, size, overlap=0.28):
    step = max(1, int(size * (1 - overlap)))
    xs = list(range(0, max(1, width - size + 1), step))
    ys = list(range(0, max(1, height - size + 1), step))
    if not xs or xs[-1] != max(0, width - size): xs.append(max(0, width - size))
    if not ys or ys[-1] != max(0, height - size): ys.append(max(0, height - size))
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            yield x, y, min(width, x + size), min(height, y + size)


def detect(processor, model, image, prompts, threshold, offset=(0, 0), source="global"):
    w, h = image.size
    inputs = processor(images=image, text=[prompts], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs, threshold=threshold, text_threshold=threshold, target_sizes=[(h, w)]
    )[0]
    labels = result.get("text_labels", result.get("labels", []))
    found = []
    for idx, (box, score) in enumerate(zip(result["boxes"].cpu().numpy(), result["scores"].cpu().numpy())):
        b = [float(box[0] + offset[0]), float(box[1] + offset[1]),
             float(box[2] + offset[0]), float(box[3] + offset[1])]
        if box_area(b) >= 48:
            found.append({"bbox": b, "score": float(score), "prompt": str(labels[idx]) if idx < len(labels) else "", "source": source})
    return found


def run_context(image, device):
    processor = AutoImageProcessor.from_pretrained(SEGFORMER_ID, local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_ID, local_files_only=True).to(device).eval()
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode(): logits = model(**inputs).logits
    logits = F.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
    labels = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    facade = np.isin(labels, [0, 1])
    vegetation = np.isin(labels, [4, 9, 17])
    sky = labels == 2
    road = labels == 6
    facade = cv2.morphologyEx(facade.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8)) > 0
    unload(model, processor, inputs, logits)
    return labels, facade, vegetation, sky, road


def run_scene_clip(image, device):
    processor = CLIPProcessor.from_pretrained(CLIP_ID, local_files_only=True)
    model = CLIPModel.from_pretrained(CLIP_ID, local_files_only=True).to(device).eval()
    names = list(SCENE_TEXTS)
    inputs = processor(text=[SCENE_TEXTS[n] for n in names], images=image, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode(): outputs = model(**inputs)
    probs = torch.softmax(outputs.logits_per_image[0], dim=0).cpu().numpy()
    scores = {n: float(probs[i]) for i, n in enumerate(names)}
    unload(model, processor, inputs, outputs)
    return scores


def run_dino(image, facade, args, device):
    processor = AutoProcessor.from_pretrained(DINO_ID, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_ID, local_files_only=True).to(device).eval()
    raw = defaultdict(list)
    for tag, prompts in PROMPTS.items():
        if tag != "window":
            raw[tag].extend(detect(processor, model, image, prompts, THRESHOLDS[tag]))
    # Local context, rather than a global scene label, decides window search.
    w, h = image.size
    searched, skipped = [], []
    for size in args.window_tile_sizes:
        for tid, b in enumerate(iter_tiles(w, h, size)):
            x1, y1, x2, y2 = b
            fr = float(facade[y1:y2, x1:x2].mean())
            if fr < 0.075:
                skipped.append({"bbox": b, "scale": size, "facade_ratio": fr})
                continue
            searched.append({"bbox": b, "scale": size, "facade_ratio": fr})
            crop = image.crop(b)
            raw["window"].extend(detect(processor, model, crop, PROMPTS["window"], THRESHOLDS["window"], (x1, y1), f"tile_{size}_{tid}"))
    # Global pass supplies large openings, but local support is retained as evidence.
    raw["window"].extend(detect(processor, model, image, PROMPTS["window"], 0.20, source="global"))
    unload(model, processor)
    return raw, searched, skipped


def facade_overlap(box, facade):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(facade.shape[1], x2), min(facade.shape[0], y2)
    return float(facade[y1:y2, x1:x2].mean()) if x2 > x1 and y2 > y1 else 0.0


def consolidate(raw, facade, width, height, max_per_tag):
    accepted, rejected = defaultdict(list), []
    for tag, candidates in raw.items():
        # First remove near-identical same-class proposals; no transitive union-find.
        candidates = nms(candidates, 0.38 if tag == "window" else 0.48)
        for c in candidates:
            c["tag"] = tag
            c["facade_overlap"] = facade_overlap(c["bbox"], facade)
            c["area_ratio"] = box_area(c["bbox"]) / (width * height)
            if tag in {"window", "door", "balcony", "railing", "grille", "awning", "storefront", "rolling_shutter", "air_conditioner"} and c["facade_overlap"] < 0.04:
                c["rejection_reason"] = "insufficient_local_facade"
                rejected.append(c); continue
            aspect = (c["bbox"][2]-c["bbox"][0]) / max(1.0, c["bbox"][3]-c["bbox"][1])
            c["aspect_ratio"] = float(aspect)
            if tag == "window" and c["score"] < 0.20:
                c["rejection_reason"] = "weak_window_evidence"
                rejected.append(c); continue
            if tag == "window" and c["area_ratio"] > 0.035:
                c["rejection_reason"] = "implausibly_large_window"
                rejected.append(c); continue
            if tag == "window" and not (0.22 <= aspect <= 4.5):
                c["rejection_reason"] = "implausible_window_aspect"
                rejected.append(c); continue
            accepted[tag].append(c)
        accepted[tag] = accepted[tag][:max_per_tag]

    # Instance-level competition: only a stronger contained competing box may down-rank a window.
    competitors = sum((accepted[t] for t in ["signboard", "air_conditioner", "vehicle", "person", "street_object"]), [])
    final_windows = []
    for w in accepted["window"]:
        conflict = None
        for other in competitors:
            if containment(w["bbox"], other["bbox"]) > 0.82 and other["score"] > w["score"] * 1.20 and box_area(other["bbox"]) < box_area(w["bbox"]) * 2.5:
                conflict = other; break
        if conflict:
            w["rejection_reason"] = "stronger_contained_instance_competition"
            w["competing_tag"] = conflict["tag"]
            rejected.append(w)
        else:
            final_windows.append(w)
    accepted["window"] = final_windows

    # Hierarchy: a large window box containing several separate small boxes becomes a group.
    groups, children = [], accepted["window"][:]
    for candidate in sorted(children, key=lambda x: box_area(x["bbox"]), reverse=True):
        inside = [x for x in children if x is not candidate and containment(x["bbox"], candidate["bbox"]) > 0.80 and box_area(x["bbox"]) < box_area(candidate["bbox"]) * 0.55]
        distinct = [x for i, x in enumerate(inside) if all(iou(x["bbox"], y["bbox"]) < 0.25 for y in inside[:i])]
        if len(distinct) >= 2 and candidate.get("area_ratio", 1.0) <= 0.075:
            candidate = dict(candidate); candidate["tag"] = "window_group"; candidate["children_refs"] = distinct
            groups.append(candidate)
    group_boxes = [g["bbox"] for g in groups]
    accepted["window"] = [w for w in children if not any(iou(w["bbox"], gb) > 0.80 for gb in group_boxes)]
    accepted["window_group"] = nms(groups, 0.5)
    return accepted, rejected


def refine_sam(image, detections, batch_size, device):
    flat = [d for items in detections.values() for d in items]
    if not flat: return [], []
    processor = Sam2Processor.from_pretrained(SAM_ID, local_files_only=True)
    model = Sam2Model.from_pretrained(SAM_ID, local_files_only=True).to(device).eval()
    masks, failed = [], []
    for start in range(0, len(flat), batch_size):
        batch = flat[start:start + batch_size]
        inputs = processor(images=image, input_boxes=[[d["bbox"] for d in batch]], return_tensors="pt").to(device)
        try:
            with torch.inference_mode(): outputs = model(**inputs, multimask_output=False)
            result = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
            result = result.numpy()
            while result.ndim > 3:
                result = np.squeeze(result, axis=0 if result.shape[0] == 1 else 1)
            if result.ndim == 2:
                result = result[None]
            for m in result: masks.append(m > 0)
        except Exception as exc:
            failed.append({"start": start, "error": repr(exc)})
            for d in batch:
                mask = np.zeros((image.height, image.width), bool)
                x1, y1, x2, y2 = map(int, d["bbox"]); mask[max(0,y1):min(image.height,y2),max(0,x1):min(image.width,x2)] = True
                masks.append(mask)
        del inputs
    unload(model, processor)
    return flat, masks, failed


def derive_wire(rgb, facade):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=70, minLineLength=60, maxLineGap=12)
    mask = np.zeros_like(gray, np.uint8)
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            length = math.hypot(x2-x1, y2-y1)
            angle = abs(math.degrees(math.atan2(y2-y1, x2-x1)))
            if length > 70 and (angle < 25 or angle > 155 or 25 < angle < 80 or 100 < angle < 155):
                cv2.line(mask, (x1,y1), (x2,y2), 255, 1)
    # Thin evidence only; never a veto and explicitly experimental.
    return (mask > 0) & ~cv2.erode(facade.astype(np.uint8), np.ones((3,3),np.uint8)).astype(bool)


def depth_stats(mask, raw_depth, norm_depth):
    valid = mask & np.isfinite(raw_depth) & np.isfinite(norm_depth)
    if not valid.any(): return {k: None for k in ["depth_mean","depth_median","depth_std","depth_p10","depth_p90","normalized_depth_mean","normalized_depth_median","relative_depth_rank"]}
    r, n = raw_depth[valid], norm_depth[valid]
    return {
        "depth_mean": float(r.mean()), "depth_median": float(np.median(r)), "depth_std": float(r.std()),
        "depth_p10": float(np.percentile(r,10)), "depth_p90": float(np.percentile(r,90)),
        "normalized_depth_mean": float(n.mean()), "normalized_depth_median": float(np.median(n)),
        "relative_depth_rank": float((norm_depth[np.isfinite(norm_depth)] <= np.median(n)).mean()),
    }


def build_relationships(instances):
    rels = []
    for i, a in enumerate(instances):
        for b in instances[i+1:]:
            ia, ca, cb = iou(a["bbox"], b["bbox"]), center(a["bbox"]), center(b["bbox"])
            if containment(a["bbox"], b["bbox"]) > .82: rels.append({"subject":a["instance_id"],"predicate":"inside","object":b["instance_id"],"confidence":.85,"source":"bbox_geometry"})
            elif containment(b["bbox"], a["bbox"]) > .82: rels.append({"subject":a["instance_id"],"predicate":"contains","object":b["instance_id"],"confidence":.85,"source":"bbox_geometry"})
            elif ia > .18: rels.append({"subject":a["instance_id"],"predicate":"overlaps","object":b["instance_id"],"confidence":min(1.0,ia+.4),"source":"bbox_geometry"})
            dx, dy = cb[0]-ca[0], cb[1]-ca[1]
            diag = max(1.0, math.hypot(a["bbox"][2]-a["bbox"][0],a["bbox"][3]-a["bbox"][1]))
            if abs(dx) < diag*1.5 and abs(dy) < diag*1.5: rels.append({"subject":a["instance_id"],"predicate":"near","object":b["instance_id"],"confidence":.65,"source":"centroid_geometry"})
            if abs(dx) > abs(dy): pred = "left_of" if dx > 0 else "right_of"
            else: pred = "above" if dy > 0 else "below"
            rels.append({"subject":a["instance_id"],"predicate":pred,"object":b["instance_id"],"confidence":.6,"source":"centroid_geometry"})
            da, db = a.get("normalized_depth_median"), b.get("normalized_depth_median")
            if da is not None and db is not None:
                pred = "similar_depth" if abs(da-db)<.04 else ("front_of" if da>db else "behind")
                rels.append({"subject":a["instance_id"],"predicate":pred,"object":b["instance_id"],"confidence":min(.9,.55+abs(da-db)),"source":"relative_depth"})
    return rels


def overlay(rgb, masks, tags):
    out = rgb.astype(np.float32).copy()
    for tag in tags:
        m = masks.get(tag)
        if m is None or not m.any(): continue
        color = np.array(COLORS[tag], np.float32)
        out[m] = out[m] * .58 + color * .42
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_instances(rgb, instances):
    im = Image.fromarray(rgb.copy()); draw = ImageDraw.Draw(im)
    for x in instances:
        color = COLORS.get(x["tag"], (255,255,255)); b = x["bbox"]
        draw.rectangle(b, outline=color, width=2)
        draw.text((b[0]+2,b[1]+2), f'{x["instance_id"]} {x["tag"]} {x["confidence"]:.2f}', fill=color)
    return np.asarray(im)


def process_scene(scene_dir, args, device):
    scene_dir = Path(scene_dir); out = scene_dir / args.output_name
    previews, debug = out / "previews", out / "debug"
    previews.mkdir(parents=True, exist_ok=True); debug.mkdir(parents=True, exist_ok=True)
    image = Image.open(scene_dir / "rgb.jpg").convert("RGB"); rgb = np.asarray(image)
    h, w = rgb.shape[:2]
    raw_depth = np.load(scene_dir / "depth_raw.npy").astype(np.float32)
    norm_depth = np.load(scene_dir / "depth_norm.npy").astype(np.float32)
    if raw_depth.shape != (h,w) or norm_depth.shape != (h,w): raise ValueError("Core depth/RGB pixel alignment mismatch")

    labels, facade, vegetation, sky, road = run_context(image, device)
    clip_scores = run_scene_clip(image, device)
    raw, searched, skipped = run_dino(image, facade, args, device)
    accepted, rejected = consolidate(raw, facade, w, h, args.max_instances_per_tag)
    flat, refined, sam_failures = refine_sam(image, accepted, args.sam_batch_size, device)

    semantic = {tag: np.zeros((h,w), bool) for tag in TAGS}
    semantic["facade"] = facade; semantic["vegetation"] = vegetation
    instances, instance_masks = [], []
    counters = defaultdict(int); ref_to_id = {}
    for det, mask in zip(flat, refined):
        tag = det["tag"]; output_tag = "window" if tag == "window_group" else tag
        # A group is a hierarchy node.  Its children, not its broad SAM region,
        # define the pixel-level Window union and prevent group spill-over.
        if tag != "window_group":
            semantic[output_tag] |= mask
        counters[tag] += 1; iid = f"{tag}_{counters[tag]:04d}"
        ref_to_id[id(det)] = iid
        ys, xs = np.where(mask)
        centroid = [float(xs.mean()),float(ys.mean())] if len(xs) else center(det["bbox"])
        rec = {
            "instance_id": iid, "tag": tag, "category": CATEGORIES[tag], "bbox": [round(v,2) for v in det["bbox"]],
            "centroid": centroid, "pixel_area": int(mask.sum()), "confidence": float(det["score"]),
            "source_model": DINO_ID, "mask_index": len(instance_masks), "scene_id": scene_dir.name,
            "dino_score": float(det["score"]), "multi_scale_support": 1, "facade_overlap": float(det.get("facade_overlap",0)),
            "local_context": {"facade_overlap":float(det.get("facade_overlap",0)),"area_ratio":float(det.get("area_ratio",0))},
            "parent_instance_id": None, "group_id": None, "reliability": RELIABILITY[output_tag],
        }
        rec.update(depth_stats(mask, raw_depth, norm_depth)); instances.append(rec); instance_masks.append(mask)

    # Resolve hierarchical window membership without changing semantic overlap.
    for det in flat:
        if det.get("tag") != "window_group": continue
        gid = ref_to_id.get(id(det))
        for child in det.get("children_refs",[]):
            cid = ref_to_id.get(id(child))
            if cid:
                for rec in instances:
                    if rec["instance_id"] == cid: rec["parent_instance_id"] = gid; rec["group_id"] = gid

    semantic["wire"] = derive_wire(rgb, facade)
    # No trustworthy column/setback evidence in V1: preserve the fixed channel
    # as an explicit zero mask instead of hallucinating a ground-floor arcade.
    semantic["arcade_candidate"] = np.zeros((h,w), bool)
    relationships = build_relationships(instances)
    for rec in instances:
        if rec["tag"] != "facade" and rec["facade_overlap"] > .55:
            relationships.append({"subject":rec["instance_id"],"predicate":"attached_to","object":"facade_semantic","confidence":rec["facade_overlap"],"source":"mask_context"})

    np.savez_compressed(out / "semantic_masks.npz", **{k:v.astype(np.uint8) for k,v in semantic.items()})
    stack = np.stack(instance_masks).astype(np.uint8) if instance_masks else np.zeros((0,h,w),np.uint8)
    np.savez_compressed(out / "instance_masks.npz", masks=stack, instance_ids=np.array([x["instance_id"] for x in instances]))
    (out / "instances.json").write_text(json.dumps(instances,indent=2,ensure_ascii=False),encoding="utf-8")
    (out / "relationships.json").write_text(json.dumps({"depth_convention":"relative; larger means nearer","relationships":relationships},indent=2,ensure_ascii=False),encoding="utf-8")
    raw_serializable = {k:v for k,v in raw.items()}
    (debug / "raw_candidates.json").write_text(json.dumps(raw_serializable,indent=2,ensure_ascii=False),encoding="utf-8")
    (debug / "rejected_candidates.json").write_text(json.dumps(rejected,indent=2,ensure_ascii=False),encoding="utf-8")
    (debug / "window_search_tiles.json").write_text(json.dumps({"searched":searched,"skipped":skipped},indent=2),encoding="utf-8")
    features = {
        "scene_id":scene_dir.name,"image_size":[w,h],"scene_context_scores":clip_scores,
        "effective_facade_ratio":float(facade.mean()),"sky_ratio":float(sky.mean()),"road_ratio":float(road.mean()),
        "vegetation_ratio":float(vegetation.mean()),"searched_window_tiles":len(searched),"skipped_window_tiles":len(skipped),
        "instance_counts":dict(counters),"semantic_pixel_counts":{k:int(v.sum()) for k,v in semantic.items()},
    }
    (out / "scene_features.json").write_text(json.dumps(features,indent=2,ensure_ascii=False),encoding="utf-8")
    metadata = {
        "parser_version":VERSION,"models":{"segformer":SEGFORMER_ID,"grounding_dino":DINO_ID,"sam2":SAM_ID,"clip":CLIP_ID},
        "pixel_alignment":{"rgb":[h,w],"depth_raw":list(raw_depth.shape),"depth_norm":list(norm_depth.shape),"semantic_masks":[h,w]},
        "semantic_tags":TAGS,"multi_label":True,"depth_type":"relative_non_metric_larger_is_nearer",
        "sam_role":"accepted-instance boundary refinement only; never semantic hard veto","sam_failures":sam_failures,
        "completion_status":"complete",
    }
    (out / "parser_metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf-8")

    Image.fromarray(overlay(rgb,semantic,TAGS)).save(previews/"semantic_overview.jpg",quality=92)
    Image.fromarray(overlay(rgb,semantic,["facade","window","door","balcony","railing","grille","awning","storefront","rolling_shutter","arcade_candidate"])).save(previews/"architecture_overlay.jpg",quality=92)
    Image.fromarray(overlay(rgb,semantic,["signboard","air_conditioner","utility_pole","wire"])).save(previews/"attached_objects_overlay.jpg",quality=92)
    Image.fromarray(overlay(rgb,semantic,["vegetation","person","vehicle","street_object"])).save(previews/"environment_overlay.jpg",quality=92)
    Image.fromarray(draw_instances(rgb,instances)).save(previews/"instance_overlay.jpg",quality=92)
    depth_color = cv2.cvtColor(cv2.applyColorMap((np.clip(norm_depth,0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
    Image.fromarray(overlay(depth_color,semantic,TAGS)).save(previews/"depth_semantic_overlay.jpg",quality=92)
    Image.fromarray(overlay(rgb,semantic,["window"])).save(previews/"window_overlay.jpg",quality=92)
    Image.fromarray(overlay(rgb,semantic,["facade"])).save(previews/"facade_overlay.jpg",quality=92)
    print(json.dumps({"scene":scene_dir.name,"instances":dict(counters),"facade_ratio":features["effective_facade_ratio"],"output":str(out)},ensure_ascii=False))
    return features


def main():
    args = parse_args(); root = Path(args.dataset_root)
    if args.all: scenes = sorted(p for p in root.glob("scene_*") if p.is_dir())
    elif args.scenes: scenes = [Path(x) if Path(x).exists() else root/x for x in args.scenes]
    else: scenes = [Path(args.scene)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{VERSION} | device={device} | scenes={len(scenes)}")
    failures = []
    for scene in scenes:
        try: process_scene(scene,args,device)
        except Exception as exc:
            failures.append({"scene":str(scene),"error":repr(exc)}); print(f"FAILED {scene}: {exc}")
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    if failures:
        print(json.dumps({"failures":failures},indent=2)); raise SystemExit(1)


if __name__ == "__main__":
    main()
