"""Non-destructive Training V1 channel QA for frozen Parser V1 outputs.

Canonical masks are read-only. This tool writes statistical flags, review previews,
and per-scene channel validity metadata outside parsing_v1.
"""
from __future__ import annotations
import argparse,csv,json,math,os
from collections import Counter,defaultdict
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
from corpus_core import PARSER_VERSION,load_manifest,utc_now,validate_scene

CHANNELS=["facade","window","signboard","vegetation","person","vehicle"]
ORDER=["depth_norm",*CHANNELS]
COLORS={"facade":(40,190,80),"window":(30,180,255),"signboard":(255,210,20),"vegetation":(20,145,40),"person":(255,50,140),"vehicle":(80,220,80)}
FIELDS=["scene_id","channel","quality_status","reason","coverage","largest_instance_ratio","largest_bbox_ratio","instance_count","confidence_mean","confidence_min","reference_set","preview_path","manual_decision","manual_note","review_timestamp"]

def read_csv(path):
    if not path.exists():return []
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def atomic_csv(path,fields,rows):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def percentile(values,q):return float(np.percentile(values,q)) if values else 0.0
def robust_limits(values):
    values=sorted(float(x) for x in values);trim=values[:max(1,math.ceil(len(values)*.95))]
    q1,q3=percentile(trim,25),percentile(trim,75);iqr=max(q3-q1,1e-8)
    return {"p50":percentile(values,50),"p95":percentile(values,95),"p99":percentile(values,99),"trimmed_q1":q1,"trimmed_q3":q3,"trimmed_iqr":iqr,"review_upper":max(percentile(values,95),q3+3*iqr),"catastrophic_upper":max(percentile(trim,99),q3+6*iqr)}
def metrics(scene,channel):
    rgb=np.asarray(Image.open(scene/"rgb.jpg").convert("RGB"));h,w=rgb.shape[:2];area=h*w
    with np.load(scene/"parsing_v1"/"semantic_masks.npz") as z:mask=z[channel].astype(bool)
    instances=json.loads((scene/"parsing_v1"/"instances.json").read_text(encoding="utf-8"));items=[x for x in instances if x.get("tag")==channel]
    pixel=[float(x.get("pixel_area",0))/area for x in items];bbox=[];conf=[]
    for x in items:
        b=x.get("bbox",[0,0,0,0]);bbox.append(max(0,float(b[2])-float(b[0]))*max(0,float(b[3])-float(b[1]))/area);conf.append(float(x.get("confidence",0)))
    return {"coverage":float(mask.mean()),"largest_instance_ratio":max(pixel,default=0),"largest_bbox_ratio":max(bbox,default=0),"instance_count":len(items),"confidence_mean":float(np.mean(conf)) if conf else 0,"confidence_min":min(conf,default=0),"mask":mask,"rgb":rgb,"items":items}
def depth_status(scene):
    try:
        rgb=np.asarray(Image.open(scene/"rgb.jpg"));raw=np.load(scene/"depth_raw.npy");norm=np.load(scene/"depth_norm.npy")
        errors=[]
        if raw.shape!=rgb.shape[:2] or norm.shape!=rgb.shape[:2]:errors.append("alignment")
        if not np.isfinite(raw).all() or not np.isfinite(norm).all():errors.append("non_finite")
        if float(np.ptp(raw))<=1e-8 or float(np.ptp(norm))<.9:errors.append("degenerate_range")
        if float(np.abs(raw).max())==0:errors.append("all_zero")
        if float(norm.min())<-.001 or float(norm.max())>1.001:errors.append("normalized_out_of_range")
        return ("PASS" if not errors else "INVALID_FOR_TRAINING",";".join(errors) or "technical_depth_checks_pass")
    except Exception as exc:return "INVALID_FOR_TRAINING",f"depth_unreadable:{exc!r}"
def preview(scene,channel,m,reason,path):
    rgb=m["rgb"];mask=m["mask"];color=np.array(COLORS[channel]);overlay=rgb.copy();overlay[mask]=(overlay[mask]*.45+color*.55).astype(np.uint8)
    boxes=overlay.copy()
    for x in m["items"]:
        b=[int(v) for v in x.get("bbox",[0,0,0,0])];cv2.rectangle(boxes,(b[0],b[1]),(b[2],b[3]),tuple(int(v) for v in COLORS[channel]),max(2,round(min(rgb.shape[:2])/300)))
    thumb_h=360;scale=thumb_h/rgb.shape[0];width=max(1,round(rgb.shape[1]*scale));panels=[cv2.resize(x,(width,thumb_h),interpolation=cv2.INTER_AREA) for x in (rgb,overlay,boxes)]
    canvas=Image.new("RGB",(width*3,thumb_h+70),(245,245,245));d=ImageDraw.Draw(canvas)
    for i,x in enumerate(panels):canvas.paste(Image.fromarray(x),(i*width,70))
    d.text((8,8),f"{scene.name} | {channel} | {reason}\ncoverage={m['coverage']:.4f} largest_mask={m['largest_instance_ratio']:.4f} largest_bbox={m['largest_bbox_ratio']:.4f} instances={m['instance_count']} conf_mean={m['confidence_mean']:.3f}",fill=(0,0,0));path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path,quality=90)

p=argparse.ArgumentParser();p.add_argument("--dataset-root",default="dataset");p.add_argument("--output-root",default="phase_1_7");p.add_argument("--reference-end",type=int,default=79);p.add_argument("--no-previews",action="store_true");a=p.parse_args()
root=Path(a.dataset_root);out=Path(a.output_root);rows=load_manifest(root/"corpus_manifest.csv");ready=[];cache={}
for row in rows:
    scene=root/"scenes"/row["scene_id"]
    validation=validate_scene(scene,require_parsing=True)
    if validation["valid"] and validation["parser_version"]==PARSER_VERSION:
        ready.append(row)
        for channel in CHANNELS:cache[(row["scene_id"],channel)]=metrics(scene,channel)
references=[r for r in ready if int(r["scene_id"].split("_")[-1])<=a.reference_end]
policy={"policy_version":"playing-models-core-channel-qa-v1","created":utc_now(),"parser_version":PARSER_VERSION,"reference_scene_count":len(references),"reference_definition":f"technically valid scenes 1-{a.reference_end}; robust thresholds use full and upper-5%-trimmed distributions","warning":"Statistical flags are not ground truth. Canonical masks remain unchanged.","channels":{}}
for channel in CHANNELS:
    vals={name:[cache[(r["scene_id"],channel)][name] for r in references] for name in ("coverage","largest_instance_ratio","largest_bbox_ratio","instance_count")}
    policy["channels"][channel]={name:robust_limits(value) for name,value in vals.items()}
(out/"CORE_CHANNEL_QA_POLICY_V1.json").parent.mkdir(parents=True,exist_ok=True);(out/"CORE_CHANNEL_QA_POLICY_V1.json").write_text(json.dumps(policy,indent=2),encoding="utf-8")
manual={(r.get("scene_id"),r.get("channel")):r for r in read_csv(out/"CORE_CHANNEL_MANUAL_REVIEW.csv")}
records=[];flagged=[];validity=[]
for row in ready:
    scene_id=row["scene_id"];dstatus,dreason=depth_status(root/"scenes"/scene_id);bits=[1 if dstatus=="PASS" else 0]
    for channel in CHANNELS:
        m=cache[(scene_id,channel)];limits=policy["channels"][channel];reasons=[];status="PASS"
        for name in ("coverage","largest_instance_ratio","largest_bbox_ratio","instance_count"):
            if m[name]>limits[name]["review_upper"]:reasons.append(f"{name}_robust_outlier")
        if channel=="window" and m["coverage"]==0 and cache[(scene_id,"facade")]["coverage"]>policy["channels"]["facade"]["coverage"]["p50"]:reasons.append("empty_window_with_strong_facade")
        if channel=="facade" and m["coverage"]>.99:reasons.append("near_whole_image_facade")
        catastrophic=(channel in {"signboard","person","vehicle"} and m["coverage"]>max(.25,limits["coverage"]["catastrophic_upper"]) and m["largest_bbox_ratio"]>max(.40,limits["largest_bbox_ratio"]["catastrophic_upper"]))
        if catastrophic:status="INVALID_FOR_TRAINING";reasons.append("conservative_catastrophic_large_region")
        elif reasons:status="REVIEW"
        prior=manual.get((scene_id,channel),{});decision=prior.get("manual_decision","")
        approved=status=="PASS" or decision=="ACCEPT_CHANNEL"
        if decision in {"EXCLUDE_CHANNEL_FROM_TRAINING","UNCERTAIN"}:approved=False
        bits.append(1 if approved else 0)
        rec={"scene_id":scene_id,"channel":channel,"quality_status":status,"reason":";".join(reasons) or "within_reference_limits","coverage":m["coverage"],"largest_instance_ratio":m["largest_instance_ratio"],"largest_bbox_ratio":m["largest_bbox_ratio"],"instance_count":m["instance_count"],"confidence_mean":m["confidence_mean"],"confidence_min":m["confidence_min"],"reference_set":f"scene_000001-scene_{a.reference_end:06d}","preview_path":"","manual_decision":decision,"manual_note":prior.get("manual_note",""),"review_timestamp":prior.get("review_timestamp","")}
        if status!="PASS":
            path=out/"core_channel_review_previews"/f"{scene_id}_{channel}.jpg";rec["preview_path"]=str(path.resolve());flagged.append(rec)
            if not a.no_previews:preview(root/"scenes"/scene_id,channel,m,rec["reason"],path)
        records.append(rec)
    validity.append({"scene_id":scene_id,"depth_quality_status":dstatus,"depth_reason":dreason,"channel_order":"|".join(ORDER),"training_channel_validity":"|".join(str(x) for x in bits),"training_approved":str(any(bits)).lower(),"parser_version":PARSER_VERSION,"qa_policy_version":policy["policy_version"]})
atomic_csv(out/"CORE_CHANNEL_QA_ALL.csv",FIELDS,records);atomic_csv(out/"CORE_CHANNEL_REVIEW_MANIFEST.csv",FIELDS,flagged);atomic_csv(out/"TRAINING_CHANNEL_VALIDITY.csv",list(validity[0]) if validity else [],validity)
print(json.dumps({"canonical_rows":len(rows),"technically_ready":len(ready),"reference_scenes":len(references),"flagged_pairs":len(flagged),"statuses":Counter(r["quality_status"] for r in records),"depth":Counter(r["depth_quality_status"] for r in validity)},indent=2))
