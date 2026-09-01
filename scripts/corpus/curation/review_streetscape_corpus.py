"""Lightweight local manual reviewer for Playing Models streetscape candidates."""
from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import webbrowser
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

DEFAULT_ROOT = Path("taiwan_streetscape_screening_v2_1")
STATUS_VALUES = {"", "approve", "reject", "uncertain"}
REVIEW_FIELDS = [
    "provider", "provider_id", "source_id", "manual_status", "manual_note",
    "review_timestamp", "thumbnail_path", "screening_tier", "original_index",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def candidate_key(row: dict[str, str]) -> str:
    provider = row.get("provider", "").strip()
    source_id = (row.get("provider_id") or row.get("source_id") or "").strip()
    return f"{provider}:{source_id}"


def load_reviews(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    reviews = {}
    for row in read_csv(path):
        key = f"{row.get('provider','').strip()}:{(row.get('provider_id') or row.get('source_id') or '').strip()}"
        if key != ":":
            reviews[key] = row
    return reviews


def write_reviews_atomic(path: Path, reviews: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(reviews.values(), key=lambda row: int(row.get("original_index") or 0)))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def manual_row(item: dict, status: str, note: str) -> dict[str, str]:
    return {
        "provider": item.get("provider", ""),
        "provider_id": item.get("provider_id", ""),
        "source_id": item.get("provider_id") or item.get("source_id", ""),
        "manual_status": status,
        "manual_note": note,
        "review_timestamp": utc_now(),
        "thumbnail_path": item.get("screen_thumbnail_path", ""),
        "screening_tier": item["screening_tier"],
        "original_index": str(item["original_index"]),
    }


def public_item(row: dict, reviews: dict[str, dict[str, str]]) -> dict:
    key = candidate_key(row)
    review = reviews.get(key, {})
    return {
        "key": key,
        "provider": row.get("provider", ""),
        "provider_id": row.get("provider_id", ""),
        "source_id": row.get("provider_id") or row.get("source_id", ""),
        "title": row.get("title", ""),
        "city": row.get("guessed_city", "") or row.get("geo_center_name", ""),
        "source_page": row.get("source_page", ""),
        "screening_tier": row["screening_tier"],
        "original_index": row["original_index"],
        "clip_street_probability": row.get("clip_street_probability", ""),
        "v2_final_score": row.get("v2_final_score", ""),
        "v2_eye_level_score": row.get("v2_eye_level_score", ""),
        "v2_spatial_context_score": row.get("v2_spatial_context_score", ""),
        "v2_playing_models_value_score": row.get("v2_playing_models_value_score", ""),
        "manual_status": review.get("manual_status", ""),
        "manual_note": review.get("manual_note", ""),
        "review_timestamp": review.get("review_timestamp", ""),
        "thumbnail_url": "/thumbnail/" + str(row["original_index"]),
    }


def image_mime(path: Path) -> str:
    with path.open("rb") as handle:
        header = handle.read(16)
    if header.startswith(b"\xff\xd8\xff"): return "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"): return "image/png"
    if header[:6] in (b"GIF87a", b"GIF89a"): return "image/gif"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP": return "image/webp"
    return "application/octet-stream"


def finalize(root: Path, review_path: Path, output_path: Path) -> dict:
    strong = read_csv(root / "v2_1_strong_keep_candidates.csv")
    keep = read_csv(root / "v2_1_keep_candidates.csv")
    reviews = load_reviews(review_path)
    reviewed = [reviews.get(candidate_key(row), {}) for row in keep]
    counts = Counter(row.get("manual_status", "") for row in reviewed)
    approved_rows = []
    output_fields = list(strong[0].keys() if strong else keep[0].keys()) + [
        "manual_status", "manual_note", "review_timestamp", "screening_tier", "original_index"
    ]
    for index, row in enumerate(strong, 1):
        out = dict(row); out.update({"manual_status":"provisional_approved", "manual_note":"", "review_timestamp":"", "screening_tier":"strong_keep", "original_index":index}); approved_rows.append(out)
    for index, row in enumerate(keep, 1):
        review = reviews.get(candidate_key(row), {})
        if review.get("manual_status") == "approve":
            out = dict(row); out.update({"manual_status":"approve", "manual_note":review.get("manual_note",""), "review_timestamp":review.get("review_timestamp",""), "screening_tier":"keep", "original_index":index}); approved_rows.append(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(approved_rows); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    summary = {
        "strong_keep_provisional_approved": len(strong),
        "keep_total": len(keep),
        "keep_reviewed": sum(counts[value] for value in ("approve","reject","uncertain")),
        "keep_approved": counts["approve"], "keep_rejected": counts["reject"], "keep_uncertain": counts["uncertain"],
        "final_approved_count": len(approved_rows),
        "city_distribution": dict(sorted(Counter((row.get("guessed_city") or row.get("geo_center_name") or "unknown") for row in approved_rows).items())),
        "output": str(output_path.resolve()),
    }
    return summary


HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Streetscape Manual Review</title><style>
:root{font-family:Inter,system-ui,sans-serif;color:#e9eef5;background:#101419}*{box-sizing:border-box}body{margin:0;overflow:hidden}
header{height:58px;display:flex;align-items:center;gap:12px;padding:8px 16px;background:#171d24;border-bottom:1px solid #303945}.brand{font-weight:700}.progress{color:#9fb0c2}.filters{margin-left:auto;display:flex;gap:6px}.filters button,.nav button,.actions button{border:1px solid #3b4653;background:#202833;color:#dce5ef;border-radius:7px;padding:8px 12px;cursor:pointer}.filters button.active{background:#385b80}.main{height:calc(100vh - 58px);display:grid;grid-template-columns:minmax(0,1fr) 380px}.viewer{position:relative;display:flex;align-items:center;justify-content:center;background:#090c10;overflow:hidden}.viewer img{max-width:100%;max-height:100%;object-fit:contain;transition:transform .15s}.viewer img.zoom{max-width:none;max-height:none;transform:scale(1.35);cursor:zoom-out}.tier{position:absolute;left:16px;top:14px;background:#101419dd;padding:7px 10px;border-radius:6px}.side{padding:18px;overflow:auto;background:#171d24;border-left:1px solid #303945}.title{font-size:18px;line-height:1.35;margin:8px 0 14px}.city{color:#9fb0c2}.score{display:grid;grid-template-columns:1fr auto;gap:8px;padding:7px 0;border-bottom:1px solid #29323c}.score b{font-variant-numeric:tabular-nums}.source{display:inline-block;margin:14px 0;color:#71b7ff}.actions{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin:12px 0}.actions .approve{background:#1f6b48}.actions .reject{background:#7b3035}.actions .uncertain{background:#77601e}.actions button.selected{outline:3px solid #fff}.note{width:100%;min-height:76px;background:#101419;color:#e9eef5;border:1px solid #3b4653;border-radius:7px;padding:9px}.nav{display:flex;justify-content:space-between;margin-top:12px}.hint{font-size:12px;color:#93a2b3;line-height:1.6;margin-top:14px}.empty{font-size:20px;color:#9fb0c2}.saved{height:20px;color:#74d89d;font-size:12px;margin-top:6px}
</style></head><body><header><div class="brand">STREETSCAPE REVIEW</div><div class="progress" id="progress"></div><div class="filters" id="filters"></div></header><div class="main"><div class="viewer" id="viewer"><div class="tier" id="tier"></div><img id="image" alt="candidate thumbnail"><div class="empty" id="empty" hidden>No items in this filter</div></div><aside class="side"><div class="city" id="city"></div><div class="title" id="title"></div><div id="scores"></div><a class="source" id="source" target="_blank" rel="noopener">Open source page ↗</a><div class="actions"><button class="approve" data-status="approve">A Approve</button><button class="reject" data-status="reject">R Reject</button><button class="uncertain" data-status="uncertain">U Uncertain</button></div><textarea class="note" id="note" maxlength="300" placeholder="Optional note"></textarea><div class="saved" id="saved"></div><div class="nav"><button id="prev">← Previous</button><button id="next">Next →</button></div><div class="hint">A approve · R reject · U uncertain<br>← / → navigate · Space zoom/fit<br>Every decision saves immediately. Notes save on blur.</div></aside></div><script>
let items=[],view=[],pos=0,filter='UNREVIEWED',current=null;const filters=['ALL','UNREVIEWED','APPROVED','REJECTED','UNCERTAIN'];
const $=id=>document.getElementById(id);const fmt=v=>{let n=Number(v);return Number.isFinite(n)?n.toFixed(3):'—'};
function applyFilter(keepKey){view=items.filter(x=>filter==='ALL'||(filter==='UNREVIEWED'?!x.manual_status:x.manual_status===filter.toLowerCase().replace('approved','approve').replace('rejected','reject')));let found=keepKey?view.findIndex(x=>x.key===keepKey):-1;pos=found>=0?found:Math.min(pos,Math.max(0,view.length-1));render()}
function render(){document.querySelectorAll('.filters button').forEach(b=>b.classList.toggle('active',b.textContent===filter));if(!view.length){current=null;$('image').hidden=true;$('tier').hidden=true;$('empty').hidden=false;$('progress').textContent='0 / 0';return}$('image').hidden=false;$('tier').hidden=false;$('empty').hidden=true;current=view[pos];$('progress').textContent=`${pos+1} / ${view.length} · original ${current.original_index} / ${items.length}`;$('tier').textContent=current.screening_tier.toUpperCase();$('image').src=current.thumbnail_url;$('image').className='';$('city').textContent=current.city||'Unknown city';$('title').textContent=current.title||'(untitled)';$('source').href=current.source_page||'#';$('source').style.display=current.source_page?'inline-block':'none';$('scores').innerHTML=[['V1 street',current.clip_street_probability],['V2 final',current.v2_final_score],['Eye-level',current.v2_eye_level_score],['Spatial context',current.v2_spatial_context_score],['Playing Models value',current.v2_playing_models_value_score]].map(x=>`<div class="score"><span>${x[0]}</span><b>${fmt(x[1])}</b></div>`).join('');$('note').value=current.manual_note||'';document.querySelectorAll('[data-status]').forEach(b=>b.classList.toggle('selected',b.dataset.status===current.manual_status));$('saved').textContent=current.review_timestamp?`Saved ${current.review_timestamp}`:''}
async function save(status,note,advance){if(!current)return;let key=current.key;let response=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key,status,note})});if(!response.ok){$('saved').textContent='Save failed';return}let data=await response.json();let item=items.find(x=>x.key===key);Object.assign(item,data.review);$('saved').textContent='Saved immediately';if(advance){let old=pos;applyFilter();if(filter==='ALL'&&view.length)pos=Math.min(old+1,view.length-1);render()}else render()}
function move(delta){if(!view.length)return;pos=(pos+delta+view.length)%view.length;render()}
async function init(){let data=await fetch('/api/items').then(r=>r.json());items=data.items;filters.forEach(name=>{let b=document.createElement('button');b.textContent=name;b.onclick=()=>{filter=name;pos=0;applyFilter()};$('filters').appendChild(b)});let first=items.findIndex(x=>!x.manual_status);filter='UNREVIEWED';applyFilter(first>=0?items[first].key:null)}
document.querySelectorAll('[data-status]').forEach(b=>b.onclick=()=>save(b.dataset.status,$('note').value,true));$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('note').addEventListener('blur',()=>current&&save(current.manual_status,$('note').value,false));$('image').onclick=()=>$('image').classList.toggle('zoom');window.addEventListener('keydown',e=>{if(e.target.tagName==='TEXTAREA')return;if(e.key==='a'||e.key==='A')save('approve',$('note').value,true);else if(e.key==='r'||e.key==='R')save('reject',$('note').value,true);else if(e.key==='u'||e.key==='U')save('uncertain',$('note').value,true);else if(e.key==='ArrowRight')move(1);else if(e.key==='ArrowLeft')move(-1);else if(e.code==='Space'){e.preventDefault();$('image').classList.toggle('zoom')}});init();
</script></body></html>'''


def run_server(args) -> None:
    root = Path(args.screening_root).resolve()
    tier_file = "v2_1_keep_candidates.csv" if args.tier == "keep" else "v2_1_review_candidates.csv"
    review_path = Path(args.review_csv).resolve() if args.review_csv else (root / ("KEEP_MANUAL_REVIEW.csv" if args.tier == "keep" else "REVIEW_MANUAL_REVIEW.csv"))
    rows = read_csv(root / tier_file)
    for index, row in enumerate(rows, 1): row["original_index"] = index; row["screening_tier"] = args.tier
    keys = {candidate_key(row): row for row in rows}; by_index = {str(row["original_index"]): row for row in rows}; reviews = load_reviews(review_path); lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, code, data, content_type):
            self.send_response(code); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/": return self.send_bytes(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            if parsed.path == "/api/items":
                with lock: data = {"items":[public_item(row,reviews) for row in rows],"review_csv":str(review_path),"tier":args.tier}
                return self.send_bytes(200,json.dumps(data,ensure_ascii=False).encode("utf-8"),"application/json; charset=utf-8")
            if parsed.path.startswith("/thumbnail/"):
                row=by_index.get(unquote(parsed.path.rsplit("/",1)[-1])); path=Path(row.get("screen_thumbnail_path","")) if row else None
                if not path or not path.is_file(): return self.send_bytes(404,b"thumbnail not found","text/plain")
                return self.send_bytes(200,path.read_bytes(),image_mime(path))
            return self.send_bytes(404,b"not found","text/plain")
        def do_POST(self):
            if urlparse(self.path).path != "/api/review": return self.send_bytes(404,b"not found","text/plain")
            try:
                length=int(self.headers.get("Content-Length","0")); payload=json.loads(self.rfile.read(length)); key=str(payload.get("key",""));status=str(payload.get("status","")).strip().lower();note=str(payload.get("note","")).strip()[:300]
                if key not in keys or status not in STATUS_VALUES: raise ValueError("invalid review payload")
                with lock:
                    reviews[key]=manual_row(keys[key],status,note);write_reviews_atomic(review_path,reviews);review=public_item(keys[key],reviews)
                return self.send_bytes(200,json.dumps({"ok":True,"review":review},ensure_ascii=False).encode("utf-8"),"application/json; charset=utf-8")
            except Exception as exc:return self.send_bytes(400,json.dumps({"ok":False,"error":str(exc)}).encode(),"application/json")
        def log_message(self, format, *values):
            if args.verbose: super().log_message(format,*values)

    server=ThreadingHTTPServer(("127.0.0.1",args.port),Handler);url=f"http://127.0.0.1:{server.server_port}/";print(json.dumps({"url":url,"tier":args.tier,"candidates":len(rows),"review_csv":str(review_path)},ensure_ascii=False,indent=2))
    if not args.no_browser: threading.Timer(.5,lambda:webbrowser.open(url)).start()
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


def main():
    parser=argparse.ArgumentParser(description="Local manual streetscape corpus reviewer")
    parser.add_argument("--screening-root",default=str(DEFAULT_ROOT));parser.add_argument("--tier",choices=("keep","review"),default="keep");parser.add_argument("--review-csv");parser.add_argument("--port",type=int,default=8765);parser.add_argument("--no-browser",action="store_true");parser.add_argument("--verbose",action="store_true");parser.add_argument("--finalize",action="store_true");parser.add_argument("--output",default="APPROVED_STREETSCAPE_CORPUS.csv");args=parser.parse_args()
    root=Path(args.screening_root).resolve();review_path=Path(args.review_csv).resolve() if args.review_csv else root/"KEEP_MANUAL_REVIEW.csv"
    if args.finalize: print(json.dumps(finalize(root,review_path,Path(args.output)),ensure_ascii=False,indent=2));return
    run_server(args)

if __name__=="__main__":main()
