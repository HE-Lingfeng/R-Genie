"""Fit a residual RAB localization head on paired edits in the revised training set."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F

from training.paper_utils import read_records, sha256


class SpatialHead(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.visual = nn.Sequential(nn.LayerNorm(2560), nn.Linear(2560,width))
        self.query = nn.Sequential(nn.LayerNorm(2560), nn.Linear(2560,width))
        self.refine = nn.Sequential(nn.Conv2d(width+2,64,3,padding=1), nn.GroupNorm(8,64),
                                    nn.GELU(), nn.Conv2d(64,32,3,padding=1), nn.GroupNorm(4,32),
                                    nn.GELU(), nn.Conv2d(32,1,1))
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, visual, readout, native):
        v = self.visual(visual.float())
        q = self.query(readout.float())[:,None,:]
        aligned = v*torch.tanh(q)
        prior = (native.float().clamp_min(1e-8)*256).log()
        dot = aligned.mean(-1)
        grid = torch.cat([aligned, dot[...,None], prior[...,None]],-1).transpose(1,2).reshape(len(v),-1,16,16)
        return self.refine(grid).flatten(1) + prior - 1.8


def rgb(path, size=512):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert('RGB').resize((size,size), Image.Resampling.BICUBIC)


def cache_data(split, device, manifests, features, data):
    manifest = manifests / (split + '.jsonl')
    split_lock = json.loads((manifests / 'split-lock.json').read_text())
    assert sha256(manifest) == split_lock['splits'][split]['sha256']
    rows = read_records(manifest)
    parts = [[] for _ in range(4)]
    for row in rows:
        path = features / (row['id'] + '.pt')
        feat = torch.load(path,map_location='cpu',weights_only=True)
        assert feat['source_sha256']==row['source_sha256']
        assert feat['instruction_sha256']==hashlib.sha256(row['instruction'].encode()).hexdigest()
        images=[]
        for key in ('source','target'):
            path=data/row[key]
            assert sha256(path)==row[key+'_sha256']
            images.append(np.asarray(rgb(path),dtype=np.int16))
        pixels=(np.max(np.abs(images[0]-images[1]),axis=2)>12.75).astype(np.float32)
        region=torch.from_numpy(pixels.reshape(16,32,16,32).mean((1,3))).flatten()
        for part,x in zip(parts,[feat['visual'],feat['readout'],feat['native_prior'],region]):
            part.append(x)
    return [torch.stack(part).to(device) for part in parts]


def quality(logits,y):
    prob=logits.float().sigmoid()
    target=y>=.25
    pred=prob>=.35
    union=(target|pred).sum(-1)
    return torch.where(union>0,(target&pred).sum(-1)/union.clamp_min(1),1.).mean()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rank',type=int,choices=[0,1],required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--manifests', type=Path, default=Path('artifacts/reditbench-resplit-v2'))
    p.add_argument('--features', type=Path, default=Path('artifacts/resplit-grounding-features'))
    p.add_argument('--output', type=Path, default=Path('runs/spatial'))
    args=p.parse_args()
    torch.set_num_threads(2)
    device=torch.device('cuda',args.rank)
    torch.cuda.set_device(device)
    out=args.output
    out.mkdir(parents=True, exist_ok=True)
    train=cache_data('train', device, args.manifests, args.features, args.data)
    val=cache_data('validation', device, args.manifests, args.features, args.data)
    for seed in ([0,2] if args.rank==0 else [1]):
        torch.manual_seed(seed)
        model=SpatialHead().to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=.00035,weight_decay=.02)
        best=-1
        trace=[]
        for epoch in range(1,31):
            model.train()
            for ids in torch.randperm(len(train[-1]),device=device).split(24):
                visual,readout,native,y=[a[ids] for a in train]
                logits=model(visual,readout,native)
                prob=logits.sigmoid()
                dice=1-(2*(y*prob).sum(1)+1)/((y+prob).sum(1)+1)
                loss=F.binary_cross_entropy_with_logits(logits,y)+dice.mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
            model.eval()
            with torch.no_grad():
                pred=torch.cat([model(*b) for b in zip(*(a.split(20) for a in val[:-1]))])
                iou=quality(pred,val[-1]).item()
            trace.append([epoch,iou])
            if iou>best:
                best=iou
                f=out/f'seed{seed}.pt'
                tmp=f.with_suffix('.tmp')
                torch.save(dict(model=model.state_dict(),seed=seed,epoch=epoch,validation_weak_iou=best,
                    train_sha256=sha256(args.manifests/'train.jsonl'),val_sha256=sha256(args.manifests/'validation.jsonl'),
                    implementation=sha256(__file__)),tmp)
                tmp.replace(f)
            if epoch%5==0: print(seed,epoch,iou,best,flush=True)
        (out/f'seed{seed}.json').write_text(json.dumps({'trace':trace,'best_val_iou':best,'weights_sha256':sha256(out/f'seed{seed}.pt')},indent=2))


if __name__=='__main__':
    main()
