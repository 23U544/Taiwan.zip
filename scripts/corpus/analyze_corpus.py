"""Generate corpus inventory, features, storage, diversity, and completion reports."""
import argparse, csv, json
from collections import Counter
from pathlib import Path
import numpy as np
from corpus_core import EXPERIMENTAL_TAGS, REGRESSION_SCENES, SEMANTIC_TAGS, load_manifest, scan_corpus, write_duplicate_report, write_manifest

p=argparse.ArgumentParser(); p.add_argument("--dataset-root",default="dataset"); p.add_argument("--project-root",default="."); args=p.parse_args()
root,project=Path(args.dataset_root),Path(args.project_root); rows,validations=scan_corpus(root,load_manifest(root/"corpus_manifest.csv")); write_manifest(root/"corpus_manifest.csv",rows); duplicates=write_duplicate_report(rows,root/"duplicate_report.csv")
valid_rgb=sum(bool(v["rgb_shape"]) for v in validations); valid_depth=sum(v["depth_complete"] for v in validations); valid_parsing=sum(v["parsing_complete"] for v in validations); corrupt=[v for v in validations if v["errors"]]
inventory=f"""# Corpus Inventory Report

- Corpus version: `playing-models-corpus-v1`
- Total local source scenes: **{len(rows)}**
- Readable RGB: **{valid_rgb}**
- Valid aligned depth pair: **{valid_depth}**
- Complete Parser V1 outputs: **{valid_parsing}**
- Missing parsing: **{sum(not v['parsing_present'] for v in validations)}**
- Incomplete/corrupt scenes: **{len(corrupt)}**
- Exact/near duplicate candidates requiring review: **{len(duplicates)}**

The eight fixed scenes remain a dedicated regression/QA set and are not treated as the whole corpus. Folder existence is not used as status truth; status is recorded in `dataset/corpus_manifest.csv` after integrity validation.
"""
(project/"CORPUS_INVENTORY_REPORT.md").write_text(inventory,encoding="utf-8")

feature_rows=[]; pixel_total=Counter(); scene_occurrence=Counter(); instance_total=Counter(); co=np.zeros((len(SEMANTIC_TAGS),len(SEMANTIC_TAGS)),dtype=np.int64); resolutions=[]; instance_depth=[]
for row in rows:
    scene=root/"scenes"/row["scene_id"]; resolutions.append((int(row["original_width"]),int(row["original_height"])))
    if row["parsing_status"]!="complete": continue
    features=json.loads((scene/"parsing_v1"/"scene_features.json").read_text(encoding="utf-8")); instances=json.loads((scene/"parsing_v1"/"instances.json").read_text(encoding="utf-8"))
    counts=Counter(item["tag"] for item in instances); instance_total.update(counts)
    for item in instances:
        if item.get("normalized_depth_median") is not None: instance_depth.append(float(item["normalized_depth_median"]))
    pixels=features["semantic_pixel_counts"]; area=int(row["original_width"])*int(row["original_height"]); present=[]
    record={"scene_id":row["scene_id"],"source_group_id":row["source_group_id"],"width":row["original_width"],"height":row["original_height"],"facade_ratio":pixels.get("facade",0)/area,"dense_facade_score":features.get("scene_context_scores",{}).get("dense_facade",0),"street_perspective_score":features.get("scene_context_scores",{}).get("street_perspective",0),"mixed_urban_score":features.get("scene_context_scores",{}).get("mixed_urban",0),"landscape_score":features.get("scene_context_scores",{}).get("landscape",0),"vegetation_ratio":pixels.get("vegetation",0)/area}
    for tag in SEMANTIC_TAGS:
        value=int(pixels.get(tag,0)); pixel_total[tag]+=value; record[f"{tag}_pixel_ratio"]=value/area; record[f"{tag}_instances"]=counts[tag]
        if value: scene_occurrence[tag]+=1; present.append(SEMANTIC_TAGS.index(tag))
    for i in present:
        for j in present: co[i,j]+=1
    feature_rows.append(record)
fields=list(feature_rows[0]) if feature_rows else ["scene_id"]
with (root/"corpus_scene_features.csv").open("w",encoding="utf-8-sig",newline="") as handle:
    writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(feature_rows)
np.savetxt(root/"semantic_cooccurrence.csv",co,delimiter=",",fmt="%d",header=",".join(SEMANTIC_TAGS),comments="")

parsed=[root/"scenes"/row["scene_id"] for row in rows if row["parsing_status"]=="complete"]
categories={"rgb":lambda s:s/"rgb.jpg","depth_raw":lambda s:s/"depth_raw.npy","depth_norm":lambda s:s/"depth_norm.npy","depth_preview":lambda s:s/"depth_preview.png","core_metadata":lambda s:s/"metadata.json","semantic_masks":lambda s:s/"parsing_v1"/"semantic_masks.npz","instance_masks":lambda s:s/"parsing_v1"/"instance_masks.npz","json":None,"previews":None,"debug":None}
sizes={key:[] for key in categories}; totals=[]
for scene in parsed:
    total=0
    for key,getter in categories.items():
        if getter: size=getter(scene).stat().st_size
        else:
            folder=scene/"parsing_v1"/key if key in {"previews","debug"} else scene/"parsing_v1"
            files=list(folder.glob("*.json")) if key=="json" else list(folder.rglob("*")); size=sum(path.stat().st_size for path in files if path.is_file())
        sizes[key].append(size); total+=size
    totals.append(total)
def gb(value): return value/(1024**3)
avg={key:(sum(value)/len(value) if value else 0) for key,value in sizes.items()}; avg_total=sum(totals)/len(totals) if totals else 0
storage="# Storage Scale Report\n\nMeasured on **%d** complete Parser V1 scenes.\n\n| Component | Mean MB/scene |\n|---|---:|\n%s\n| Total measured | %.2f |\n\n| Scale | Projected GB |\n|---|---:|\n| 49 scenes | %.2f |\n| 500 scenes | %.2f |\n| 1000 scenes | %.2f |\n\nInstance masks alone project to **%.2f GB at 1000 scenes**. Canonical NPZ remains lossless and reconstructable; if this becomes operationally heavy, add a versioned lossless bit-packed/RLE archive without changing existing `parsing_v1`. Training exports intentionally omit full instance masks, previews, and debug evidence.\n"%(len(parsed),"\n".join(f"| {key} | {value/1024**2:.2f} |" for key,value in avg.items()),avg_total/1024**2,gb(avg_total*49),gb(avg_total*500),gb(avg_total*1000),gb(avg['instance_masks']*1000))
(project/"STORAGE_SCALE_REPORT.md").write_text(storage,encoding="utf-8")
total_pixels=sum(int(row["original_width"])*int(row["original_height"]) for row in rows if row["parsing_status"]=="complete") or 1
rare=sorted(SEMANTIC_TAGS,key=lambda tag:(scene_occurrence[tag],pixel_total[tag]))
diversity="# Corpus Diversity Report\n\nThis is a diagnostic of the current **%d-scene parsed subset**, not the final 500–1000-scene corpus.\n\n| Class | Scene occurrence | Positive pixel ratio | Instances |\n|---|---:|---:|---:|\n%s\n\n## Acquisition priorities\n\nThe least represented evidence is: **%s**. Future lawful sourcing should prioritize independent source groups containing narrow alleys, arcade/ground-floor commercial structure, old apartments and townhouses, sign-heavy shophouses, metal-sheet additions, vegetation-heavy and vehicle-heavy streets, while retaining street-perspective and dense-façade balance. Current CLIP scores are soft descriptors, not hard scene labels. Suggested inverse-frequency loss statistics can be derived from the positive pixel ratios above; loss design remains out of scope.\n"%(len(feature_rows),"\n".join(f"| {tag} | {scene_occurrence[tag]}/{len(feature_rows)} | {pixel_total[tag]/total_pixels:.6f} | {instance_total[tag]} |" for tag in SEMANTIC_TAGS),", ".join(rare[:6]))
(project/"CORPUS_DIVERSITY_REPORT.md").write_text(diversity,encoding="utf-8")
print(json.dumps({"scenes":len(rows),"valid_rgb":valid_rgb,"valid_depth":valid_depth,"valid_parsing":valid_parsing,"failures":len(corrupt),"duplicates":len(duplicates),"parsed_features":len(feature_rows)},indent=2))
