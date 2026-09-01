#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taiwan.zip — Spatial VAE V1 (Colab training script)

Example in Colab:
!python /content/drive/MyDrive/TaiwanZip/taiwan_zip_colab_train.py \
  --data-zip /content/drive/MyDrive/TaiwanZip/taiwan_zip_prototype_bundle.zip \
  --output /content/drive/MyDrive/TaiwanZip/results_v1 \
  --epochs 80 --batch-size 16
"""
from __future__ import annotations

import argparse, json, math, random, shutil, time, zipfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

CHANNELS=["depth_norm","facade","window","signboard","vegetation","person","vehicle"]
SEMANTIC_CHANNELS=CHANNELS[1:]
SEM_COLORS=np.array([[.8,.8,.8],[.2,.55,1.0],[1.0,.5,.1],[.2,.8,.35],[.95,.25,.65],[.95,.8,.15]],np.float32)


def cli():
    p=argparse.ArgumentParser()
    p.add_argument("--data-zip",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--work-dir",default="/content/taiwan_zip_work")
    p.add_argument("--epochs",type=int,default=80)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--latent-dim",type=int,default=128)
    p.add_argument("--kl-beta",type=float,default=0.002)
    p.add_argument("--kl-warmup",type=int,default=20)
    p.add_argument("--depth-weight",type=float,default=1.0)
    p.add_argument("--bce-weight",type=float,default=1.0)
    p.add_argument("--dice-weight",type=float,default=0.5)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--workers",type=int,default=2)
    return p.parse_args()


class NPZDataset(Dataset):
    def __init__(self,path,augment=False):
        d=np.load(path,allow_pickle=False)
        self.depth=d["depth"]; self.sem=d["semantics"]
        self.scene_ids=d["scene_ids"]; self.sample_ids=d["sample_ids"]; self.groups=d["source_group_ids"]
        self.augment=augment
    def __len__(self): return len(self.depth)
    def __getitem__(self,i):
        dep=torch.from_numpy(self.depth[i].astype(np.float32))[None]
        sem=torch.from_numpy(self.sem[i].astype(np.float32))
        x=torch.cat([dep,sem],0)
        if self.augment and random.random()<.5: x=torch.flip(x,[2])
        return {"x":x,"scene_id":str(self.scene_ids[i]),"sample_id":str(self.sample_ids[i]),"group":str(self.groups[i])}


class Down(nn.Module):
    def __init__(self,a,b):
        super().__init__(); self.m=nn.Sequential(nn.Conv2d(a,b,4,2,1),nn.GroupNorm(min(8,b),b),nn.SiLU())
    def forward(self,x): return self.m(x)


class Up(nn.Module):
    def __init__(self,a,b):
        super().__init__(); self.m=nn.Sequential(nn.ConvTranspose2d(a,b,4,2,1),nn.GroupNorm(min(8,b),b),nn.SiLU())
    def forward(self,x): return self.m(x)


class TaiwanZipSpatialVAE(nn.Module):
    def __init__(self,latent_dim=128):
        super().__init__(); self.latent_dim=latent_dim
        self.enc=nn.Sequential(Down(7,32),Down(32,64),Down(64,128),Down(128,256),Down(256,256))
        flat=256*8*8
        self.mu=nn.Linear(flat,latent_dim); self.logvar=nn.Linear(flat,latent_dim)
        self.fc=nn.Linear(latent_dim,flat)
        self.dec=nn.Sequential(Up(256,256),Up(256,128),Up(128,64),Up(64,32),Up(32,32))
        self.depth=nn.Sequential(nn.Conv2d(32,1,3,padding=1),nn.Sigmoid())
        self.sem=nn.Conv2d(32,6,3,padding=1)
    def encode(self,x):
        h=self.enc(x).flatten(1); return self.mu(h),self.logvar(h)
    def reparam(self,mu,lv):
        if not self.training: return mu
        return mu+torch.randn_like(mu)*torch.exp(.5*lv)
    def decode(self,z):
        h=self.fc(z).view(-1,256,8,8); h=self.dec(h); return self.depth(h),self.sem(h)
    def forward(self,x):
        mu,lv=self.encode(x); z=self.reparam(mu,lv); d,s=self.decode(z)
        return d,s,mu,lv,z


def dice_loss(logits,target,eps=1e-6):
    p=torch.sigmoid(logits); dims=(0,2,3)
    inter=(p*target).sum(dims); den=p.sum(dims)+target.sum(dims)
    return 1-((2*inter+eps)/(den+eps)).mean()


def sem_comp(sem):
    h,w=sem.shape[1:]; rgb=np.zeros((h,w,3),np.float32); wt=np.zeros((h,w,1),np.float32)
    for c in range(6):
        p=sem[c][...,None]; rgb+=p*SEM_COLORS[c]; wt+=p
    return np.clip(rgb/np.maximum(wt,1.0),0,1)


def save_generated(model,z,out_dir,prefix,meta,device):
    out_dir.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        d,s=model.decode(z[None].to(device)); d=d[0,0].float().cpu().numpy(); s=torch.sigmoid(s[0]).float().cpu().numpy()
    np.save(out_dir/f"{prefix}_depth.npy",d.astype(np.float32))
    np.savez_compressed(out_dir/f"{prefix}_semantics.npz",**{k:s[i].astype(np.float32) for i,k in enumerate(SEMANTIC_CHANNELS)})
    (out_dir/f"{prefix}_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    return d,s


def main():
    a=cli(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device, torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU")
    work=Path(a.work_dir); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(a.data_zip) as z: z.extractall(work)
    data=work/"training_taiwan_zip_prototype"
    summary=json.loads((data/"prototype_dataset_summary.json").read_text())
    print(json.dumps(summary,indent=2))

    train=NPZDataset(data/"train.npz",True); val=NPZDataset(data/"val.npz"); test=NPZDataset(data/"test.npz"); reg=NPZDataset(data/"regression.npz")
    tl=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=True)
    vl=DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=True)
    print("samples:",len(train),len(val),len(test),len(reg))

    frac=summary["semantic_positive_fraction_train"]
    pw=[]
    for k in SEMANTIC_CHANNELS:
        p=max(float(frac[k]),1e-6); pw.append(float(np.clip((1-p)/p,1,20)))
    posw=torch.tensor(pw,device=device).view(1,6,1,1)
    print("pos_weight:",dict(zip(SEMANTIC_CHANNELS,pw)))

    model=TaiwanZipSpatialVAE(a.latent_dim).to(device)
    print("params:",sum(p.numel() for p in model.parameters()))
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-5)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    history={k:[] for k in ["train","val","depth","bce","dice","kl","beta"]}
    best=float("inf"); best_path=out/"taiwan_zip_best.pt"; last_path=out/"taiwan_zip_last.pt"

    def loss(x,res,beta):
        d,s,mu,lv,_=res; td=x[:,:1]; ts=x[:,1:]
        ld=F.l1_loss(d,td)
        lb=F.binary_cross_entropy_with_logits(s,ts,pos_weight=posw)
        ldi=dice_loss(s,ts)
        kl=-.5*torch.mean(1+lv-mu.pow(2)-lv.exp())
        total=a.depth_weight*ld+a.bce_weight*lb+a.dice_weight*ldi+beta*kl
        return total,ld,lb,ldi,kl

    t0=time.time()
    for ep in range(1,a.epochs+1):
        beta=a.kl_beta*min(1,ep/max(a.kl_warmup,1)); model.train(); sums=np.zeros(5,np.float64); n=0
        for b in tl:
            x=b["x"].to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=device.type=="cuda"):
                res=model(x); ls=loss(x,res,beta)
            scaler.scale(ls[0]).backward(); scaler.step(opt); scaler.update()
            bs=x.size(0); n+=bs; sums += np.array([float(v.detach()) for v in ls])*bs
        tr=sums/max(n,1)
        model.eval(); vs=0.; vn=0
        with torch.no_grad():
            for b in vl:
                x=b["x"].to(device,non_blocking=True)
                with torch.amp.autocast("cuda",enabled=device.type=="cuda"):
                    ls=loss(x,model(x),beta)
                vs+=float(ls[0])*x.size(0); vn+=x.size(0)
        vv=vs/max(vn,1)
        for k,v in zip(["train","depth","bce","dice","kl"],tr): history[k].append(float(v))
        history["val"].append(vv); history["beta"].append(beta)
        ck={"epoch":ep,"model":model.state_dict(),"optimizer":opt.state_dict(),"history":history,"latent_dim":a.latent_dim,"channels":CHANNELS}
        torch.save(ck,last_path)
        if vv<best: best=vv; torch.save(ck,best_path)
        print(f"[{ep:03d}/{a.epochs}] train={tr[0]:.4f} val={vv:.4f} depth={tr[1]:.4f} bce={tr[2]:.4f} dice={tr[3]:.4f} kl={tr[4]:.4f} beta={beta:.6f}")

    ck=torch.load(best_path,map_location=device); model.load_state_dict(ck["model"]); model.eval()
    (out/"training_history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    plt.figure(figsize=(8,4)); plt.plot(history["train"],label="train"); plt.plot(history["val"],label="val"); plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss"); plt.tight_layout(); plt.savefig(out/"training_curve.png",dpi=180); plt.close()

    # Reconstruction
    nrec=min(4,len(reg)); fig,ax=plt.subplots(nrec,4,figsize=(12,3*nrec)); ax=np.atleast_2d(ax)
    with torch.no_grad():
        for r in range(nrec):
            item=reg[r]; x=item["x"][None].to(device); d,s,mu,lv,z=model(x); sem=torch.sigmoid(s[0]).cpu().numpy()
            ax[r,0].imshow(x[0,0].cpu(),cmap="gray",vmin=0,vmax=1); ax[r,0].set_title(item["sample_id"]+"\ninput depth")
            ax[r,1].imshow(sem_comp(x[0,1:].cpu().numpy())); ax[r,1].set_title("input semantics")
            ax[r,2].imshow(d[0,0].cpu(),cmap="gray",vmin=0,vmax=1); ax[r,2].set_title("recon depth")
            ax[r,3].imshow(sem_comp(sem)); ax[r,3].set_title("recon semantics")
            for c in range(4): ax[r,c].axis("off")
    plt.tight_layout(); plt.savefig(out/"reconstruction_grid.png",dpi=180); plt.close()

    # Encode two regression samples
    def enc(ds,i):
        item=ds[i]; x=item["x"][None].to(device)
        with torch.no_grad(): mu,_=model.encode(x)
        return item,mu[0]
    ia,za=enc(reg,0); ib,zb=enc(reg,min(1,len(reg)-1))

    # Interpolation
    idir=out/"generated_interpolation"; vals=[]
    for i,t in enumerate([0,.25,.5,.75,1]):
        z=za*(1-t)+zb*t
        vals.append((t,*save_generated(model,z,idir,f"{i:02d}",{"type":"interpolation","a":ia["sample_id"],"b":ib["sample_id"],"t":t,"depth":"0=farther,1=nearer"},device)))
    fig,ax=plt.subplots(2,len(vals),figsize=(15,6))
    for i,(t,d,s) in enumerate(vals): ax[0,i].imshow(d,cmap="gray",vmin=0,vmax=1); ax[0,i].set_title(f"t={t}"); ax[1,i].imshow(sem_comp(s)); ax[0,i].axis("off"); ax[1,i].axis("off")
    plt.tight_layout(); plt.savefig(out/"latent_interpolation_grid.png",dpi=200); plt.close()

    # Perturbation
    pdir=out/"generated_perturbation"; vals=[]
    for i,sigma in enumerate([0,.10,.25,.50]):
        z=za.clone() if sigma==0 else za+sigma*torch.randn_like(za)
        vals.append((sigma,*save_generated(model,z,pdir,f"{i:02d}",{"type":"perturbation","source":ia["sample_id"],"sigma":sigma,"depth":"0=farther,1=nearer"},device)))
    fig,ax=plt.subplots(2,len(vals),figsize=(12,6))
    for i,(sg,d,s) in enumerate(vals): ax[0,i].imshow(d,cmap="gray",vmin=0,vmax=1); ax[0,i].set_title(f"sigma={sg}"); ax[1,i].imshow(sem_comp(s)); ax[0,i].axis("off"); ax[1,i].axis("off")
    plt.tight_layout(); plt.savefig(out/"latent_perturbation_grid.png",dpi=200); plt.close()

    # Extrapolation
    edir=out/"generated_extrapolation"; vals=[]; direction=zb-za
    for i,t in enumerate([-.5,0,.5,1,1.5]):
        z=za+t*direction
        vals.append((t,*save_generated(model,z,edir,f"{i:02d}",{"type":"extrapolation","a":ia["sample_id"],"b":ib["sample_id"],"t":t,"depth":"0=farther,1=nearer"},device)))
    fig,ax=plt.subplots(2,len(vals),figsize=(15,6))
    for i,(t,d,s) in enumerate(vals): ax[0,i].imshow(d,cmap="gray",vmin=0,vmax=1); ax[0,i].set_title(f"t={t}"); ax[1,i].imshow(sem_comp(s)); ax[0,i].axis("off"); ax[1,i].axis("off")
    plt.tight_layout(); plt.savefig(out/"latent_extrapolation_grid.png",dpi=200); plt.close()

    report={"best_epoch":ck["epoch"],"best_val":best,"runtime_seconds":time.time()-t0,"device":str(device),"latent_dim":a.latent_dim,"channels":CHANNELS,"prototype_summary":summary}
    (out/"run_summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("DONE. Results:",out)

if __name__=="__main__": main()
