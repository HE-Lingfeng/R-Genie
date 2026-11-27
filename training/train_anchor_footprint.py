"""Learn the support of an anchored addition in coordinates defined by the source instance."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 32
PENALTIES = (.1, 1., 10., 100.)
THRESHOLDS = (.20, .35, .50)


def window(mask):
    arr = np.asarray(mask.convert('L'), dtype=np.float32) / 255.
    if arr.shape != (512,512) or not np.isfinite(arr).all() or arr.sum() == 0:
        raise ValueError('Expected a nonempty 512-square source mask')
    axis = np.arange(512, dtype=np.float64) + .5
    mass = arr.sum(dtype=np.float64)
    cx, cy = axis @ arr.sum(0) / mass, axis @ arr.sum(1) / mass
    extent = max(32., 2. * np.sqrt(mass))
    return float(cx - extent/2), float(cy - extent/2), float(extent)


def crop(mask, area, size):
    x,y,extent = area
    # Supersample the extent to retain partially occupied boundary cells.
    n = 4*size
    patch = mask.transform((n,n),Image.Transform.AFFINE,(extent/n,0,x,0,extent/n,y),
                           Image.Resampling.BILINEAR,fillcolor=0)
    return (np.asarray(patch,dtype=np.float32)/255.).reshape(size,4,size,4).mean((1,3))


def features(mask, area):
    x,y,extent = area
    left,top,right,bottom = mask.getbbox()
    geometry = [(x+extent/2)/512,(y+extent/2)/512,extent/512,
                (right-left)/extent,(bottom-top)/extent, np.log(extent/128.)]
    return np.concatenate((crop(mask,area,8).flatten(),geometry)).astype(np.float64)


def read_mask(root, row, key):
    from training.paper_utils import sha256
    path = root/row[key]
    if sha256(path) != row[key+'_sha256']:
        raise ValueError('Changed training annotation '+str(path))
    with Image.open(path) as im:
        return im.convert('L').point(lambda x: 255 if x else 0).resize((512,512),Image.Resampling.NEAREST)


def fit(x,y,penalty):
    mean = x.mean(axis=0)
    center = x-mean
    bias = y.mean(axis=0)
    matrix = np.linalg.solve(center.T @ center + penalty*np.eye(x.shape[1]),center.T @ (y-bias))
    return mean,matrix,bias


def apply(x,parameters):
    mean,matrix,bias = parameters
    return np.clip((x-mean)@matrix+bias,0.,1.)


def iou(probabilities, actual, threshold):
    pred=probabilities>=threshold
    intersection=(pred*actual).sum(axis=1)
    return intersection / np.maximum(1e-9,pred.sum(axis=1)+actual.sum(axis=1)-intersection)


def train(merged,manifests,out):
    from training.cache_merged_grounding import atomic_write, json_bytes, partition_rows
    from training.paper_utils import sha256
    rows,digest=partition_rows(merged,manifests,'fit')
    rows=[r for r in rows if r.get('operation')=='anchored_add']
    if len(rows)<5 or any(r['supervision']!='native-instance-and-edit-region' for r in rows):
        raise ValueError('Expected paired addition and anchor masks on the fit split')
    xs,ys,folds=[],[],[]
    for row in rows:
        mask=read_mask(merged,row,'target_mask')
        area=window(mask)
        xs.append(features(mask,area))
        ys.append(crop(read_mask(merged,row,'edit_region'),area,SIZE).flatten())
        key=('anchor-footprint/v1/'+row['source_group']).encode()
        folds.append(int.from_bytes(hashlib.sha256(key).digest()[:4],'big')%5)
    x,y,folds=np.stack(xs),np.stack(ys).astype(np.float64),np.asarray(folds)
    if set(folds)!={0,1,2,3,4}:
        raise ValueError('Five training folds must be populated')
    scores=[]
    for penalty in PENALTIES:
        preds=np.empty_like(y)
        for f in range(5):
            hold=folds==f
            preds[hold]=apply(x[hold],fit(x[~hold],y[~hold],penalty))
        for t in THRESHOLDS:
            scores.append(dict(penalty=penalty,threshold=t,iou=float(iou(preds,y,t).mean())))
    selected=max(scores,key=lambda s:s['iou'])
    mean,matrix,bias=fit(x,y,selected['penalty'])
    out.mkdir(parents=True,exist_ok=True)
    path=out/'anchor.npz'
    def write_weights(p):
        with p.open('wb') as stream:
            np.savez_compressed(stream,mean=mean,matrix=matrix,bias=bias,threshold=selected['threshold'],size=SIZE)
    atomic_write(path,write_weights)
    report=dict(fit_manifest=digest,examples=len(rows),fold_counts={str(k):int(v) for k,v in Counter(folds).items()},
                fit_id_sha256=hashlib.sha256('\n'.join(r['id'] for r in rows).encode()).hexdigest(),
                training_sha256=sha256(__file__),weights_sha256=sha256(path),selection=selected,cross_validation=scores,
                supervision='The source-instance and edit masks define normalized training windows. Five source-group folds select ridge regularization and the extent threshold on fit examples only.')
    atomic_write(out/'anchor.json',lambda p:p.write_bytes(json_bytes(report)))
    print(json.dumps(dict(fit=len(rows),**selected)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('merged','manifests','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    train(args.merged,args.manifests,args.output)
