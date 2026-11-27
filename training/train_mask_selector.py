"""Fit a SAM candidate selector from reference source-instance masks and Qwen intentions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


FEATURES = ('soft_iou', 'sam_confidence', 'precision', 'recall', 'area', 'absolute_log_area_ratio')
BASE = np.array([1., .1, 0., 0., 0., 0.])
PENALTIES = (1., 16., 64., 256.)


def mask_features(masks, confidence, source_probability):
    """One feature vector per nonempty candidate, in the segmenter's original order."""
    masks = masks.reshape(-1, *masks.shape[-2:]).bool().cpu()
    prob = source_probability.detach().float().reshape(16, 16).clamp(0, 1).cpu()
    confidence = confidence.detach().float().reshape(-1).cpu()
    if len(masks) != len(confidence) or not torch.isfinite(confidence).all() or not torch.isfinite(prob).all():
        raise ValueError('Invalid SAM candidates or source intention')
    present = masks.flatten(1).any(1)
    pooled = F.adaptive_avg_pool2d(masks[:,None].float(), (16, 16))[:,0]
    mass = pooled.sum((1, 2))
    intersection = (pooled * prob).sum((1, 2))
    union = mass + prob.sum() - intersection
    features = torch.stack((intersection / union.clamp_min(1e-6), confidence,
                            intersection / mass.clamp_min(1e-6), intersection / prob.sum().clamp_min(1e-6),
                            mass / 256, ((mass + 1e-6) / (prob.sum() + 1e-6)).log().abs()), dim=1)
    return present.numpy(), features[present].double().numpy()


def fit(examples, penalty):
    # Centering within each image fits the relative quality of competing masks.
    # Ridge shrinks toward the existing soft-overlap / SAM-confidence selector.
    x, y = [], []
    for sample in examples:
        f = np.asarray(sample['features'], dtype=np.float64)
        if len(f) < 2:
            continue
        target = np.asarray(sample['iou']) - f @ BASE
        x.append((f - f.mean(0)) / np.sqrt(len(f)))
        y.append((target - target.mean()) / np.sqrt(len(f)))
    if not x:
        raise ValueError('Training requires competing masks on at least one photograph')
    x, y = np.concatenate(x), np.concatenate(y)
    scale = np.sqrt(np.mean(x*x, axis=0)).clip(1e-4)
    z = x / scale
    return BASE + np.linalg.solve(z.T @ z + penalty * np.eye(len(BASE)), z.T @ y) / scale


def selected_iou(sample, weights):
    if not sample['features']:
        return sample['fallback_iou']
    idx = np.argmax(np.asarray(sample['features']) @ weights)
    return sample['iou'][idx]


def macro(scores):
    actions = {op for op, _ in scores}
    return float(np.mean([np.mean([score for op, score in scores if op == action]) for action in actions]))


def train(merged, manifests, examples, output):
    from training.cache_merged_grounding import atomic_write, json_bytes, partition_rows
    from training.paper_utils import sha256
    rows, identity = partition_rows(merged, manifests, 'fit')
    by_id = {r['id']:r for r in rows}
    payload = json.loads(examples.read_bytes())
    data = payload['samples']
    if (payload['partition'] != 'fit' or payload['manifest'] != identity or not data
            or len({s['id'] for s in data}) != len(data) or payload['features'] != list(FEATURES)):
        raise ValueError('Mask examples must belong to the fitting partition')
    folds = []
    for s in data:
        r = by_id[s['id']]
        if (s['source_sha256'] != r['source_sha256'] or s['target_mask_sha256'] != r['target_mask_sha256']
                or s['operation'] != r['operation'] or s['source_group'] != r['source_group']):
            raise ValueError('Mismatched fitting mask or source '+s['id'])
        f, y = np.asarray(s['features']), np.asarray(s['iou'])
        if (len(y) != len(f) or (len(f) and f.shape != (len(y),len(BASE))) or y.shape != (len(f),)
                or not np.isfinite(f).all() or not np.isfinite(y).all() or (y < 0).any() or (y > 1).any()
                or (not len(f) and not 0 <= s['fallback_iou'] <= 1)):
            raise ValueError('Invalid candidate features or IoU in '+s['id'])
        folds.append(int(hashlib.sha256(r['source_group'].encode()).hexdigest()[:12],16) % 5)
    if len(set(folds)) != 5:
        raise ValueError('Expected five fitting folds')
    scores = {}
    for p in PENALTIES:
        held = []
        for i in range(5):
            weights = fit([s for s,g in zip(data,folds) if g != i],p)
            held.extend((s['operation'],selected_iou(s,weights)) for s,g in zip(data,folds) if g == i)
        scores[p] = macro(held)
    penalty = max(PENALTIES, key=lambda p:(scores[p],p))
    result = dict(fit=identity, examples_sha256=sha256(examples), training_sha256=sha256(__file__),
                  features=list(FEATURES), count=len(data), crossvalidated_macro_iou=scores,
                  penalty=penalty, weights=fit(data,penalty).tolist())
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(output, lambda path:path.write_bytes(json_bytes(result)))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('merged','manifests','examples','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    a = parser.parse_args()
    print(json.dumps(train(a.merged,a.manifests,a.examples,a.output)))
