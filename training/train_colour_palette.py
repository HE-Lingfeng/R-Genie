"""Learn regional RGB transforms from requested colours and paired training foregrounds."""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import torch

from training.cache_merged_grounding import atomic_write, validate
from training.paper_utils import sha256
from training.train_color_transform import ColourTransform, colour_rows, intention


HUES = set('black white grey gray brown red orange yellow green cyan blue violet purple pink teal magenta turquoise gold silver'.split())
SHADES = set('light dark bright pale deep pastel muted navy royal cobalt mustard forest brick olive mint lime sky hot'.split())
COMMAND = re.compile(r'\b(?:recolou?r|paint|tint|dye|turn|make|change|set)\b', re.I)


def requested_colour(instruction):
    """Read a colour at the end of an editing clause. Ambiguous requests defer to the learned text head."""
    found = set()
    for clause in re.split(r'[.;!?]', instruction.casefold()):
        command = COMMAND.search(clause)
        if command is None:
            continue
        words = re.findall(r"[a-z]+", clause[command.end():])
        if not words or words[-1] not in HUES:
            continue
        start = len(words)-1
        if start and words[start-1] in SHADES:
            start -= 1
        if start and words[start-1] in ('or','and','not','never','than','from','except'):
            return None
        if re.search(r"\b(?:do not|don't|never)\s+$", clause[:command.start()]):
            return None
        found.add(' '.join(words[start:]).replace('gray','grey'))
    return next(iter(found)) if len(found)==1 else None


def fit_palette(rows, moments, selected):
    if [r['id'] for r in rows] != moments['ids']:
        raise ValueError('Foreground statistics do not match fitting samples')
    groups = defaultdict(list)
    for i in selected:
        cue = requested_colour(rows[i]['instruction'])
        if cue:
            groups[cue].append(i)
    transforms = {}
    for cue, indices in sorted(groups.items()):
        if len(indices) < 16:
            continue
        gram = moments['gram'][indices].double().mean(0)
        cross = moments['cross'][indices].double().mean(0)
        theta = torch.linalg.solve(gram + 1e-4*torch.eye(gram.size(0),dtype=gram.dtype),cross).float()
        if theta.shape!=(7,3) or not torch.isfinite(theta).all():
            raise ValueError('Invalid fitted colour transform for '+cue)
        transforms[cue] = dict(count=len(indices),coefficients=theta.tolist())
    return transforms


def fit_mixture(semantic, named, gram, cross):
    """Fit one convex mixing weight using held-out foreground RGB moments."""
    diff=(named-semantic).double()
    g=gram.double()
    residual=cross.double()-g@semantic.double()
    energy=(diff*(g@diff)).sum()
    if not torch.isfinite(energy):
        raise ValueError('Invalid foreground moments')
    if energy<=1e-12:
        return 0.
    weight=(diff*residual).sum()/energy
    if not torch.isfinite(weight):
        raise ValueError('Nonfinite fitted colour mixture')
    return float(weight.clamp(0,1))


@torch.inference_mode()
def calibrate_mixture(rows, values, split, checkpoint, head_rows, cache, calibration_ids):
    """Use disjoint fitting images to combine a previously fitted semantic head with colour terms."""
    snapshot=torch.load(checkpoint,map_location='cpu',weights_only=True)
    fit=snapshot['report']['fit']
    original={r['id']:r for r in map(json.loads,head_rows.open())}
    indices=[i for i,r in enumerate(rows) if r['id'] in original]
    features=cache/'fit'
    identity=json.loads((features/'identity.json').read_bytes())
    if (identity['manifest_sha256']!=split or fit['manifest']!=split or fit['count']!=len(indices)
            or any(identity[k]!=v for k,v in fit['extractor'].items())
            or any(rows[i]!=original[rows[i]['id']] for i in indices)):
        raise ValueError('The semantic checkpoint was trained with different fitting data')
    signature=hashlib.sha256()
    for i in indices:
        uid=rows[i]['id']
        signature.update((uid+' '+sha256(features/(uid+'.pt'))+'\n').encode())
    if signature.hexdigest()!=fit['feature_set_sha256']:
        raise ValueError('Training readouts do not match the semantic checkpoint')
    ids=json.loads(calibration_ids.read_bytes())
    by_id={r['id']:i for i,r in enumerate(rows)}
    if (not isinstance(ids,list) or not ids or not all(isinstance(x,str) for x in ids)
            or len(set(ids))!=len(ids) or not set(ids)<=by_id.keys() or set(ids)&original.keys()):
        raise ValueError('Calibration IDs must be unique fitting recolours outside the head training data')
    calibration=[by_id[uid] for uid in ids]
    transforms=fit_palette(rows,values,indices)
    model=ColourTransform().eval()
    model.load_state_dict(snapshot['model'],strict=True)
    vectors=[]
    inputs=hashlib.sha256()
    for i in calibration:
        path=features/(rows[i]['id']+'.pt')
        feature=torch.load(path,map_location='cpu',weights_only=True)
        validate(feature,rows[i])
        vectors.append(intention(feature))
        inputs.update((rows[i]['id']+' '+sha256(path)+'\n').encode())
    semantic=model(torch.stack(vectors))
    named=semantic.clone()
    for j,i in enumerate(calibration):
        cue=requested_colour(rows[i]['instruction'])
        if cue in transforms:
            named[j]=torch.tensor(transforms[cue]['coefficients'])
    weight=fit_mixture(semantic,named,values['gram'][calibration],values['cross'][calibration])
    return dict(weight=weight,count=len(ids),cache_sha256=inputs.hexdigest(),head_sha256=sha256(checkpoint),
                calibration_sha256=sha256(calibration_ids),head_fit_sha256=sha256(head_rows))


def train(merged, manifests, moments, output, *, mixture_inputs=None):
    rows, split = colour_rows(merged, manifests, 'fit')
    saved = torch.load(moments, map_location='cpu', weights_only=True)
    values = saved['splits']['fit']
    if (values['manifest']!=split or saved['training_code']!=sha256(Path(__file__).with_name('train_color_transform.py'))):
        raise ValueError('Pair statistics differ from the fitting release')
    transforms = fit_palette(rows, values, range(len(rows)))
    if not transforms:
        raise ValueError('No supported requested colour found in the fitting split')
    artifact = dict(fit=split,count=len(rows),training_code=sha256(__file__),moments=sha256(moments),
                    min_support=16,ridge=1e-4,transforms=transforms)
    if mixture_inputs is not None:
        artifact['mixture']=calibrate_mixture(rows,values,split,*mixture_inputs)
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_write(output,lambda p:p.write_text(json.dumps(artifact,sort_keys=True)+'\n'))
    return artifact


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('merged','manifests','moments','output'):
        parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--head',type=Path,help='Optional trained ColourTransform checkpoint')
    parser.add_argument('--head-fit',type=Path,help='Exact fitting JSONL used for the semantic head')
    parser.add_argument('--cache',type=Path,help='Frozen Qwen input cache containing the fit directory')
    parser.add_argument('--calibration-ids',type=Path,help='JSON list of disjoint fitting IDs for the mixing weight')
    args=parser.parse_args()
    inputs=(args.head,args.head_fit,args.cache,args.calibration_ids)
    if any(inputs) and not all(inputs):
        parser.error('Fitting a mixture requires --head, --head-fit, --cache and --calibration-ids together')
    train(args.merged,args.manifests,args.moments,args.output,mixture_inputs=inputs if all(inputs) else None)
