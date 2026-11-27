"""Train an intention-conditioned foreground colour transform on real paired edits."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F

from training.cache_merged_grounding import atomic_write, partition_rows, validate
from training.paper_utils import sha256


def read_pixels(root, row, key):
    path = root / row[key]
    if sha256(path) != row[key + '_sha256']:
        raise ValueError('Changed training media: ' + str(path))
    mask = key == 'target_mask'
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('L' if mask else 'RGB')
        image = image.resize((512, 512), Image.Resampling.NEAREST if mask else Image.Resampling.BICUBIC)
        values = np.asarray(image)
    return values > 0 if mask else values.astype(np.float32) / 255.


def colour_basis(pixels, mean_rgb):
    """A linear map can change the regional mean independently of within-object texture."""
    pixels = np.asarray(pixels, dtype=np.float32)
    constant = np.ones(pixels.shape[:-1] + (1,), dtype=np.float32)
    return np.concatenate((constant, pixels - mean_rgb, np.broadcast_to(mean_rgb, pixels.shape)), axis=-1)


def pair_statistics(source, target, mask):
    x = source[mask]
    if not len(x):
        raise ValueError('An instance mask contains no pixels')
    y = target[mask] - x
    basis = colour_basis(x, x.mean(0))
    return (basis.T @ basis / len(x), basis.T @ y / len(x), np.mean(y*y))


def colour_rows(merged, manifests, split):
    rows, digest = partition_rows(merged, manifests, split)
    selected = [r for r in rows if r.get('operation') == 'recolor']
    if not selected or any(r['supervision'] != 'native-instance-and-edit-region' for r in selected):
        raise ValueError('Expected native recolour pairs')
    return selected, digest


def prepare(args):
    prepared = {}
    for split in ('fit', 'development'):
        rows, digest = colour_rows(args.merged, args.manifests, split)
        records = []
        for row in rows:
            records.append(pair_statistics(*(read_pixels(args.merged, row, key)
                                             for key in ('source', 'target', 'target_mask'))))
        gram, cross, energy = (torch.from_numpy(np.stack(seq)) for seq in zip(*records))
        prepared[split] = dict(manifest=digest, ids=[r['id'] for r in rows], gram=gram, cross=cross, energy=energy)
        print(split, len(rows), 'source/reference foreground moments', flush=True)
    doc = dict(training_code=sha256(__file__), splits=prepared)
    args.moments.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(args.moments, lambda p: torch.save(doc, p))


def intention(feature):
    readout = F.layer_norm(feature['readout'].float(), (2560,))
    words = F.layer_norm(feature['text'][-64:].float().mean(0), (2560,))
    return torch.cat((readout, words))


class ColourTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(5120, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(.15),
                                     nn.Linear(64, 21))
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, vectors):
        return self.encoder(vectors).reshape(-1, 7, 3)


def image_mse(coefficients, gram, cross, energy):
    energy_change = (coefficients * (gram @ coefficients - 2 * cross)).sum((-1, -2)) / 3
    return energy + energy_change


def examples(args, stats, split):
    rows, digest = colour_rows(args.merged, args.manifests, split)
    if stats['manifest'] != digest or stats['ids'] != [r['id'] for r in rows]:
        raise ValueError('Colour supervision is bound to a different manifest')
    folder = args.cache / split
    identity = json.loads((folder/'identity.json').read_bytes())
    if identity['partition'] != split or identity['manifest_sha256'] != digest or identity['backend'] != 'cuda':
        raise ValueError('Wrong frozen Qwen features')
    require_full = split == 'development' or not args.available_fit
    hashes = {}
    if require_full:
        for part in range(identity['shards']):
            record = json.loads((folder/f'complete-{part}.json').read_bytes())
            if record['identity_sha256'] != sha256(folder/'identity.json'):
                raise ValueError('Mismatched feature receipt')
            hashes.update(record['sha256'])
    take, vectors = [], []
    identity_hash = hashlib.sha256()
    for i,row in enumerate(rows):
        path = folder/(row['id']+'.pt')
        if not require_full and not path.is_file():
            continue
        fingerprint = sha256(path)
        if require_full and fingerprint != hashes[row['id']]:
            raise ValueError('Mismatched feature: '+str(path))
        feature = torch.load(path, map_location='cpu', weights_only=True)
        validate(feature,row)
        identity_hash.update((row['id']+' '+fingerprint+'\n').encode())
        take.append(i)
        vectors.append(intention(feature))
    if not take:
        raise ValueError('Empty training or development split')
    return (torch.stack(vectors), *(stats[k][take] for k in ('gram','cross','energy'))), dict(
        count=len(take), eligible=len(rows), manifest=digest, feature_set_sha256=identity_hash.hexdigest(),
        extractor={k:identity[k] for k in ('checkpoint','extractor_sha256','torch','transformers','backend','dtype')})


@torch.inference_mode()
def compare(model, dataset, null_transform, order):
    x, gram, cross, energy = dataset
    model.eval()
    prediction = model(x)
    mse = image_mse(prediction, gram, cross, energy)
    shuffled = image_mse(prediction[order], gram, cross, energy)
    pooled = image_mse(null_transform.expand_as(prediction), gram, cross, energy)
    return dict(foreground_mse=mse.mean().item(), input_mse=energy.mean().item(),
                unconditional_mse=pooled.mean().item(), shuffled_condition_mse=shuffled.mean().item(),
                better_than_input=int((mse < energy).sum()), better_than_unconditional=int((mse < pooled).sum()))


def train(args):
    stats = torch.load(args.moments,map_location='cpu',weights_only=True)
    if stats['training_code'] != sha256(__file__):
        raise ValueError('Regenerate moments with the current training source')
    torch.manual_seed(args.seed)
    fit, fit_id = examples(args, stats['splits']['fit'], 'fit')
    dev, dev_id = examples(args, stats['splits']['development'], 'development')
    if fit_id['extractor'] != dev_id['extractor']:
        raise ValueError('Train and development features use different extractors')
    x, gram, cross, energy = fit
    null_transform = torch.linalg.solve(gram.mean(0)+1e-5*torch.eye(7), cross.mean(0))
    permutation = torch.randperm(len(dev[0]), generator=torch.Generator().manual_seed(args.seed+1))
    model = ColourTransform()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.02)
    best, history = None, []
    for epoch in range(1,args.epochs+1):
        model.train()
        for b in torch.randperm(len(x)).split(64):
            optimizer.zero_grad(set_to_none=True)
            coeff = model(x[b])
            loss = image_mse(coeff,gram[b],cross[b],energy[b]).mean() + 1e-4*coeff.square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            optimizer.step()
        metrics = compare(model,dev,null_transform,permutation)
        if not math.isfinite(metrics['foreground_mse']):
            raise FloatingPointError('Nonfinite development loss')
        history.append(dict(epoch=epoch,**metrics))
        if best is None or metrics['foreground_mse'] < best['metrics']['foreground_mse']:
            best = dict(epoch=epoch,metrics=metrics,model={k:v.detach().clone() for k,v in model.state_dict().items()})
    args.output.mkdir(parents=True,exist_ok=True)
    report = dict(fit=fit_id,development=dev_id,available_fit=args.available_fit,epochs=args.epochs,seed=args.seed,
                  training_code=sha256(__file__),moments_sha256=sha256(args.moments),
                  best_epoch=best['epoch'],metrics=best['metrics'],history=history,
                  selection='Lowest equal-image foreground RGB MSE on the development recolour pairs with native object masks.')
    atomic_write(args.output/'colour.pt', lambda p:torch.save(dict(model=best['model'],report=report),p))
    atomic_write(args.output/'colour.json', lambda p:p.write_text(json.dumps(
        dict(**report,checkpoint_sha256=sha256(args.output/'colour.pt')),indent=2)+'\n'))
    print(json.dumps(dict(fit=len(x),development=len(dev[0]),epoch=best['epoch'],**best['metrics'])),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','train'))
    for key in ('merged','manifests','moments'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--available-fit',action='store_true',help='Prototype using only currently cached fit features.')
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.action=='train' and (args.output is None or args.cache is None or args.epochs<1):
        parser.error('Training requires --output, --cache and a positive epoch count')
    torch.set_num_threads(2)
    prepare(args) if args.action=='prepare' else train(args)
