#!/usr/bin/env python3
"""
ZetaPhi CIFAR-100 BASELINES — public, reproducible baseline trainers.

This script contains ONLY standard, published baseline architectures used as
reference points in ZetaPhi benchmark reports:
  - Transformer  (multi-head self-attention ViT-style)
  - ConvMixer    (depthwise + pointwise conv mixer; Trockman & Kolter, 2022)

It does NOT contain ZetaPhi's proprietary linear-scaling architecture or any
of its mechanism. It exists so anyone can independently reproduce the baseline
numbers that ZetaPhi's models are compared against.

Recipe (held identical across all lanes for a fair comparison):
  AdamW, weight_decay=1e-2, batch=128, CrossEntropyLoss, no LR schedule.
  Data aug: RandomCrop(32, pad=4) + RandomHorizontalFlip. CIFAR-100 norm.
  Patch 4 -> 8x8=64 tokens, d_model=128, hidden=256.
  Transformer: depth=4, lr=3e-4.
  ConvMixer:   depth=15 (k=3) or 14 (k=5), lr tuned per kernel (see --lr).

Usage:
  python baselines.py --lanes transformer convmixer_k5 --seeds 1 2 3 --epochs 100 --lr 1e-3
"""
from __future__ import annotations
import argparse, json, random, time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BASE = {'image_size':32,'patch':4,'d_model':128,'hidden_dim':256,'depth':4,
        'stem_mid_ratio':0.5,'attn_pool_temperature':1.0,'num_classes':100}
BATCH_SIZE, NUM_WORKERS = 128, 4
WEIGHT_DECAY = 1e-2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = './data_cifar100'

# ---------- shared components (standard ViT/ConvMixer plumbing) ----------
class StrongerStem(nn.Module):
    def __init__(self, in_ch, out_dim, patch, mid_ratio=0.5):
        super().__init__(); mid = max(int(out_dim*mid_ratio), 24)
        self.net = nn.Sequential(nn.Conv2d(in_ch, mid, 3, 1, 1), nn.BatchNorm2d(mid), nn.GELU(),
                                 nn.Conv2d(mid, out_dim, kernel_size=patch, stride=patch))
    def forward(self, x): return self.net(x)

class StemRMSNorm(nn.Module):
    def __init__(self, eps=1e-6): super().__init__(); self.eps=eps
    def forward(self, x): return x / torch.sqrt(torch.mean(x*x, dim=-1, keepdim=True) + self.eps)

class AttentionPool(nn.Module):
    def __init__(self, d_model, temperature=1.0):
        super().__init__(); self.temperature=temperature
        self.attn = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))
    def forward(self, x):
        w = torch.softmax(self.attn(x)/max(self.temperature,1e-6), dim=1)
        return (x*w).sum(dim=1)

# ---------- Transformer baseline ----------
class TransformerBlock(nn.Module):
    def __init__(self, d, h, heads=4):
        super().__init__()
        self.n1=nn.LayerNorm(d); self.attn=nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2=nn.LayerNorm(d); self.mlp=nn.Sequential(nn.Linear(d,h),nn.GELU(),nn.Linear(h,d))
    def forward(self, x):
        y,_=self.attn(self.n1(x),self.n1(x),self.n1(x),need_weights=False); x=x+y
        return x + self.mlp(self.n2(x))

class TransformerVision(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.g=cfg['image_size']//cfg['patch']; d,h=cfg['d_model'],cfg['hidden_dim']
        self.stem=StrongerStem(3,d,cfg['patch'],cfg['stem_mid_ratio']); self.pre=StemRMSNorm()
        self.pos=nn.Parameter(torch.randn(1,self.g*self.g,d)*0.02)
        self.blocks=nn.ModuleList([TransformerBlock(d,h) for _ in range(cfg['depth'])])
        self.pool=AttentionPool(d,cfg['attn_pool_temperature']); self.head=nn.Linear(d,cfg['num_classes'])
    def forward(self,x):
        x=self.pre(self.stem(x).flatten(2).transpose(1,2))+self.pos
        for b in self.blocks: x=b(x)
        return self.head(self.pool(x))

# ---------- ConvMixer baseline ----------
class ConvMixerBlock(nn.Module):
    def __init__(self, d, g, k=3):
        super().__init__(); self.g=g
        self.dw=nn.Conv2d(d,d,k,groups=d,padding=k//2,padding_mode='circular'); self.bn1=nn.BatchNorm2d(d)
        self.pw=nn.Conv2d(d,d,1); self.bn2=nn.BatchNorm2d(d)
    def forward(self,x):
        b,n,d=x.shape; c=x.transpose(1,2).reshape(b,d,self.g,self.g)
        c=c+F.gelu(self.bn1(self.dw(c))); c=F.gelu(self.bn2(self.pw(c)))
        return c.flatten(2).transpose(1,2).contiguous()

class ConvMixerVision(nn.Module):
    def __init__(self, cfg, depth, k=3):
        super().__init__(); self.g=cfg['image_size']//cfg['patch']; d=cfg['d_model']
        self.stem=StrongerStem(3,d,cfg['patch'],cfg['stem_mid_ratio']); self.pre=StemRMSNorm()
        self.pos=nn.Parameter(torch.randn(1,self.g*self.g,d)*0.02)
        self.blocks=nn.ModuleList([ConvMixerBlock(d,self.g,k) for _ in range(depth)])
        self.pool=AttentionPool(d,cfg['attn_pool_temperature']); self.head=nn.Linear(d,cfg['num_classes'])
    def forward(self,x):
        x=self.pre(self.stem(x).flatten(2).transpose(1,2))+self.pos
        for b in self.blocks: x=b(x)
        return self.head(self.pool(x))

def build(lane, cfg):
    if lane=='transformer':   return TransformerVision(cfg)
    if lane=='convmixer_k3':  return ConvMixerVision(cfg, depth=15, k=3)
    if lane=='convmixer_k5':  return ConvMixerVision(cfg, depth=14, k=5)
    raise KeyError(lane)

# ---------- data / train / eval ----------
def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def loaders():
    m,sd=(0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)
    tr_tf=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(m,sd)])
    te_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(m,sd)])
    tr=datasets.CIFAR100(DATA_ROOT,train=True,download=True,transform=tr_tf)
    te=datasets.CIFAR100(DATA_ROOT,train=False,download=True,transform=te_tf)
    return (DataLoader(tr,BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=True),
            DataLoader(te,BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=True))

def train_epoch(m,l,opt,crit):
    m.train(); cor=n=0; ls=0.0
    for xb,yb in l:
        xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad()
        out=m(xb); loss=crit(out,yb); loss.backward(); opt.step()
        ls+=loss.item()*xb.size(0); cor+=(out.argmax(1)==yb).sum().item(); n+=xb.size(0)
    return ls/n, cor/n

@torch.no_grad()
def evaluate(m,l,crit):
    m.eval(); cor=n=0; ls=0.0
    for xb,yb in l:
        xb,yb=xb.to(DEVICE),yb.to(DEVICE); out=m(xb)
        ls+=crit(out,yb).item()*xb.size(0); cor+=(out.argmax(1)==yb).sum().item(); n+=xb.size(0)
    return ls/n, cor/n

def nparams(m): return sum(p.numel() for p in m.parameters())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--lanes',nargs='+',default=['transformer','convmixer_k5'])
    ap.add_argument('--seeds',nargs='+',type=int,default=[1,2,3])
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--out',default='./baseline_results')
    a=ap.parse_args()
    tr,te=loaders(); print(f"device={DEVICE} lanes={a.lanes} seeds={a.seeds} epochs={a.epochs} lr={a.lr}")
    allres=[]
    for lane in a.lanes:
        for seed in a.seeds:
            set_seed(seed); m=build(lane,BASE).to(DEVICE)
            opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=WEIGHT_DECAY); crit=nn.CrossEntropyLoss()
            best=0.0; rows=[]
            for ep in range(1,a.epochs+1):
                trl,tra=train_epoch(m,tr,opt,crit); evl,eva=evaluate(m,te,crit); best=max(best,eva)
                rows.append({'epoch':ep,'train_acc':tra,'eval_acc':eva})
                if ep%10==0 or ep==a.epochs: print(f"{lane} s{seed} ep{ep} train={tra:.4f} eval={eva:.4f} best={best:.4f}",flush=True)
            res={'lane':lane,'seed':seed,'params':nparams(m),'best_eval_acc':best,'final_eval_acc':rows[-1]['eval_acc']}
            allres.append(res)
            d=Path(a.out)/lane/f'seed_{seed}'; d.mkdir(parents=True,exist_ok=True)
            (d/'result.json').write_text(json.dumps({**res,'epochs':rows},indent=2))
    from collections import defaultdict
    agg=defaultdict(list)
    for r in allres: agg[r['lane']].append(r)
    print("\n=== BASELINE SUMMARY (mean over seeds) ===")
    summ=[]
    for lane,rs in agg.items():
        bm=np.mean([r['best_eval_acc'] for r in rs]); bs=np.std([r['best_eval_acc'] for r in rs])
        summ.append({'lane':lane,'params':rs[0]['params'],'best_eval_acc_mean':float(bm),'best_eval_acc_std':float(bs),
                     'seeds':[round(r['best_eval_acc'],4) for r in rs]})
        print(f"{lane:14s} best_eval={bm*100:.2f}±{bs*100:.2f}  params={rs[0]['params']:,}")
    Path(a.out).mkdir(parents=True,exist_ok=True)
    (Path(a.out)/'SUMMARY.json').write_text(json.dumps(summ,indent=2))

if __name__=='__main__': main()
