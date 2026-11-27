"""Refine a spatial intention head using two implicit cues for different objects in the same photo."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import default_collate

from training.cache_merged_grounding import atomic_write, partition_rows, validate
from training.merged_grounder import MergedGrounder, objective
from training.paper_utils import sha256


def box_occupancy(box):
    x0,y0,x1,y1=box
    if not (0<=x0<x1<=1 and 0<=y0<y1<=1):
        raise ValueError('Invalid annotated box')
    edges=torch.arange(17,dtype=torch.float32)/16
    def span(lo,hi):
        return ((edges[1:].clamp(max=hi)-edges[:-1].clamp(min=lo))*16).clamp_min(0)
    return (span(y0,y1)[:,None]*span(x0,x1)[None,:]).flatten()


def read_pairs(spec, fit, folder):
    from training.train_merged_grounder import feature_inputs
    items={}
    for row in spec['pairs']:
        uid=row['id']
        if uid in items or uid not in fit or fit[uid]['source_sha256']!=row['source_sha256'] or len(row['sides'])!=2:
            raise ValueError('Expected two cues for one distinct fit photo')
        path=folder/(uid+'.pt')
        if sha256(path)!=row['features_sha256']:
            raise ValueError(f'Changed paired readout for {uid}')
        f=torch.load(path,map_location='cpu',weights_only=True)
        texts=[s['instruction'] for s in row['sides']]
        hashes=[hashlib.sha256(t.encode()).hexdigest() for t in texts]
        if (len(set(hashes))!=2 or len(f['sides'])!=2 or
                f['binding']!=dict(backend=spec['paired_extractor'],source=row['source_sha256'],instructions=hashes)):
            raise ValueError('Paired input text, image or extractor changed')
        inputs=[feature_inputs(dict(s,visual=f['visual'])) for s in f['sides']]
        occupancy=torch.stack([box_occupancy(s['box']) for s in row['sides']])
        background=(1-occupancy.sum(0)).clamp_min(0)
        if background.sum()<=0:raise ValueError('Expected visible background')
        items[uid]=(inputs,occupancy/occupancy.sum(-1,keepdim=True),background/background.sum())
    return items


def read_native(ids, fit, cache, warm, state):
    from training.train_merged_grounder import feature_inputs
    meta=state['identity']['partitions']['fit']
    if sha256(warm/'fit.jsonl')!=meta['manifest'] or sha256(warm/'fit.npz')!=meta['regions']:
        raise ValueError('The warm-start spatial labels changed')
    sources=list(map(json.loads,(warm/'fit.jsonl').open()))
    positions={row['id']:i for i,row in enumerate(sources)}
    hashes={}
    for i,expected in enumerate(meta['features']):
        receipt=warm/f'complete-{i}.json'
        if sha256(receipt)!=expected:raise ValueError('Changed warm-start feature receipt')
        hashes.update(json.loads(receipt.read_bytes())['sha256'])
    keys=('source','destination','edit','has_instance','has_geometry','action','requested_dx_dy_log_scale')
    features={}
    with np.load(warm/'fit.npz',allow_pickle=False) as z:
        if z['ids'].tolist()!=list(positions):
            raise ValueError('Warm-start rows and regions differ')
        arrays={key:z[key] for key in keys}
        for uid in ids:
            i=positions[uid]
            if uid in features or sources[i]!=fit[uid]:
                raise ValueError('Repeated or mismatched native fit record')
            path=cache/(uid+'.pt')
            if sha256(path)!=hashes[uid]:raise ValueError(path)
            raw=torch.load(path,map_location='cpu',weights_only=True)
            validate(raw,fit[uid])
            labels={k:torch.as_tensor(a[i]) for k,a in arrays.items()}
            for k in ('source','edit','destination'):
                labels[k]=labels[k].flatten()
            labels['action']=labels['action'].long()
            features[uid]=feature_inputs(raw),labels
    return features


def pair_margins(model,items,keys):
    inputs=default_collate([one for k in keys for one in items[k][0]])
    maps=model(*inputs)['source'].sigmoid().reshape(len(keys),2,-1)
    boxes=torch.stack([items[k][1] for k in keys])
    background=torch.stack([items[k][2] for k in keys])
    matches=maps@boxes.transpose(1,2)
    positive=matches.diagonal(dim1=1,dim2=2)
    other=matches.flip((2,)).diagonal(dim1=1,dim2=2)
    return positive-other,positive-(maps*background[:,None]).sum(2)


def train(args):
    torch.set_num_threads(2)
    rows,digest=partition_rows(args.merged,args.manifests,'fit')
    fit={r['id']:r for r in rows}
    spec=json.loads(args.spec.read_bytes())
    parent=args.warm_start/'checkpoint.pt'
    if digest!=spec['fit_manifest'] or sha256(parent)!=spec['warm_sha256']:
        raise ValueError('Fit manifest or parent checkpoint changed')
    state=torch.load(parent,map_location='cpu',weights_only=True)
    extractor=state['identity']['extractor']
    if any(extractor[k]!=spec['paired_extractor'][k] for k in ('checkpoint','extractor_sha256','dtype')):
        raise ValueError('Different frozen encoders for paired and ordinary cues')
    paired=read_pairs(spec,fit,args.paired_features)
    rehearsal=spec['rehearsal']
    if not paired or not rehearsal or set(rehearsal)&set(paired):
        raise ValueError('Expected separate nonempty paired and ordinary fit photos')
    native=read_native(rehearsal,fit,args.canonical_features,args.warm_start,state)
    pair_ids=list(paired)
    s=spec['settings']
    if (min(s[k] for k in ('epochs','pair_batch','native_batch'))<1 or
            min(s[k] for k in ('lr','temperature','weight'))<=0):
        raise ValueError('Invalid cue training configuration')
    torch.manual_seed(s['seed'])
    model=MergedGrounder()
    model.load_state_dict(state['best_model'],strict=True)
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=s['lr'],weight_decay=.02)
    rng=torch.Generator().manual_seed(s['seed'])
    for _ in range(s['epochs']):
        replay=torch.randperm(len(rehearsal),generator=rng).tolist()
        for step,ix in enumerate(torch.randperm(len(paired),generator=rng).split(s['pair_batch'])):
            ids=[pair_ids[j] for j in ix.tolist()]
            distractor,background=pair_margins(model,paired,ids)
            selected=[rehearsal[replay[(step*s['native_batch']+i)%len(replay)]] for i in range(s['native_batch'])]
            x,y=default_collate([native[k] for k in selected])
            loss=objective(model(*x),y)
            loss+=s['weight']*F.softplus((s['margin']-background)/s['temperature']).mean()
            loss+=s['weight']*F.softplus((s['margin']-distractor)/s['temperature']).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            opt.step()
    identity=dict(parent=spec['warm_sha256'],fit_manifest=digest,spec_sha256=sha256(args.spec),
                  training_code=sha256(__file__),model_code=state['identity']['model_code'],settings=s,
                  paired_extractor=spec['paired_extractor'],paired_sources={k:fit[k]['source_sha256'] for k in paired},
                  rehearsal_sources={k:fit[k]['source_sha256'] for k in native})
    payload=dict(identity=identity,model=model.state_dict())
    args.output.parent.mkdir(exist_ok=True,parents=True)
    with (args.output.with_suffix('.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if args.output.is_file():
            old=torch.load(args.output,map_location='cpu',weights_only=True)
            if old['identity']!=identity or any(not torch.equal(old['model'][k],v) for k,v in payload['model'].items()):
                raise ValueError('Destination already contains a different cue refinement')
        else:
            atomic_write(args.output,lambda tmp:torch.save(payload,tmp))
    print('Trained cue alignment on',len(paired),'paired photos and',len(native),'ordinary fit photos',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('merged','manifests','spec','warm-start','paired-features','canonical-features','output'):
        p.add_argument('--'+key,type=Path,required=True)
    train(p.parse_args())
