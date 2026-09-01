"""Resumable Depth Anything V2 worker for already-ingested canonical scenes."""
import argparse,gc,json,time
from pathlib import Path
import cv2,numpy as np,torch
from depth_anything_v2.dpt import DepthAnythingV2
from corpus_core import load_manifest,scan_corpus,utc_now,write_manifest

CONFIGS={"vits":{"encoder":"vits","features":64,"out_channels":[48,96,192,384]},"vitb":{"encoder":"vitb","features":128,"out_channels":[96,192,384,768]},"vitl":{"encoder":"vitl","features":256,"out_channels":[256,512,1024,1024]}}
p=argparse.ArgumentParser();p.add_argument('--dataset-root',default='dataset');p.add_argument('--encoder',choices=CONFIGS,default='vitb');p.add_argument('--checkpoint');p.add_argument('--input-size',type=int,default=518);p.add_argument('--device',default='auto');p.add_argument('--start',type=int);p.add_argument('--end',type=int);p.add_argument('--shard',type=int);p.add_argument('--num-shards',type=int);p.add_argument('--scenes',nargs='*');p.add_argument('--force',action='store_true');p.add_argument('--dry-run',action='store_true');args=p.parse_args()
root=Path(args.dataset_root).resolve();manifest=root/'corpus_manifest.csv';rows,_=scan_corpus(root,load_manifest(manifest));write_manifest(manifest,rows)
selected=rows
if args.scenes:selected=[r for r in selected if r['scene_id'] in set(args.scenes)]
if args.start is not None:selected=[r for r in selected if int(r['scene_id'].split('_')[-1])>=args.start]
if args.end is not None:selected=[r for r in selected if int(r['scene_id'].split('_')[-1])<=args.end]
if args.shard is not None:
    if not args.num_shards or not 0<=args.shard<args.num_shards:p.error('--shard requires valid --num-shards')
    selected=[r for i,r in enumerate(selected) if i%args.num_shards==args.shard]
queue=[r for r in selected if args.force or r['depth_status']!='complete'];print(json.dumps({'selected':len(selected),'depth_queue':len(queue),'skipped_complete':len(selected)-len(queue)}))
if args.dry_run or not queue:raise SystemExit(0)
device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
checkpoint=Path(args.checkpoint) if args.checkpoint else Path(__file__).resolve().parent/'checkpoints'/f'depth_anything_v2_{args.encoder}.pth'
model=DepthAnythingV2(**CONFIGS[args.encoder]);model.load_state_dict(torch.load(checkpoint,map_location='cpu'));model=model.to(device).eval();failures=root/'pipeline_failures.jsonl';benchmark=root/'depth_benchmark.jsonl'
for row in queue:
    scene=root/'scenes'/row['scene_id'];started=time.perf_counter()
    try:
        data=np.fromfile(str(scene/'rgb.jpg'),dtype=np.uint8);image=cv2.imdecode(data,cv2.IMREAD_COLOR)
        if image is None:raise ValueError('rgb unreadable')
        h,w=image.shape[:2];raw=np.asarray(model.infer_image(image,args.input_size),dtype=np.float32)
        if raw.shape!=(h,w):raw=cv2.resize(raw,(w,h),interpolation=cv2.INTER_LINEAR)
        if not np.isfinite(raw).all():raise ValueError('non-finite depth')
        lo,hi=float(raw.min()),float(raw.max())
        if hi-lo<1e-12:raise ValueError('degenerate depth range')
        norm=np.clip((raw-lo)/(hi-lo),0,1).astype(np.float32);np.save(scene/'depth_raw.npy',raw);np.save(scene/'depth_norm.npy',norm)
        ok,encoded=cv2.imencode('.png',(norm*255).astype(np.uint8));
        if not ok:raise IOError('depth preview encoding failed')
        encoded.tofile(str(scene/'depth_preview.png'))
        metadata=json.loads((scene/'metadata.json').read_text(encoding='utf-8'));metadata.update({'width':w,'height':h,'encoder':args.encoder,'input_size':args.input_size,'device':device,'depth_type':'relative','depth_direction':'larger_is_nearer','metric_depth':False,'raw_depth':{'min':lo,'max':hi,'mean':float(raw.mean()),'std':float(raw.std()),'finite_ratio':1.0},'normalized_depth':{'min':0.0,'max':1.0,'black':'farther','white':'nearer'}});(scene/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
        event={'scene_id':scene.name,'stage':'depth','timestamp':utc_now(),'seconds':time.perf_counter()-started,'encoder':args.encoder,'input_size':args.input_size,'device':device}
        with benchmark.open('a',encoding='utf-8') as handle:handle.write(json.dumps(event)+'\n')
    except Exception as exc:
        event={'scene_id':scene.name,'stage':'depth','timestamp':utc_now(),'exception':repr(exc),'retry_count':1}
        with failures.open('a',encoding='utf-8') as handle:handle.write(json.dumps(event,ensure_ascii=False)+'\n')
del model;gc.collect();
if torch.cuda.is_available():torch.cuda.empty_cache()
rows,_=scan_corpus(root,load_manifest(manifest));write_manifest(manifest,rows)
