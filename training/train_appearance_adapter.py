"""Fit a regional visual-prompt adapter and its matched image-only control.

The registration JSON lists disjoint fit, calibration and development image IDs.
Each pairs/<split>/<id>.pt file stores source, context and edited-region ViT-H
embeddings, a source-only Qwen readout, and photograph identity. Each split's
complete.json binds those files to the registration by SHA-256. Calibration
selects the epoch; development compares the selected heads on the same photos.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from training.cache_merged_grounding import atomic_write
from training.paper_utils import sha256
from training.train_visual_intention import VisualIntention, error, objective


def read_split(root, name, ids, registration, seen):
    folder = root / name
    receipt = json.loads((folder / 'complete.json').read_bytes())
    if receipt['registration'] != registration or set(receipt['sha256']) != set(ids):
        raise ValueError(f'Unexpected appearance cache in {name}')
    vectors = {key: [] for key in ('source', 'context', 'intention', 'target')}
    actions = []
    for uid in ids:
        if Path(uid).name != uid or uid in seen['ids']:
            raise ValueError('Invalid or repeated image ID')
        path = folder / (uid + '.pt')
        if sha256(path) != receipt['sha256'][uid]:
            raise ValueError(f'Appearance feature checksum mismatch for {uid}')
        record = torch.load(path, weights_only=True, map_location='cpu')
        identity = record['identity']
        digests = {identity['source_sha256'], identity['target_sha256']}
        if (identity['registration'] != registration or identity['id'] != uid
                or digests & seen['photos'] or identity['source_group'] in seen['groups']):
            raise ValueError(f'Incorrect or overlapping photograph in {name}: {uid}')
        seen['photos'].update(digests)
        seen['groups'].add(identity['source_group'])
        seen['ids'].add(uid)
        for key, values in vectors.items():
            vector = record[key].float()
            size = 2560 if key == 'intention' else 1024
            if vector.shape != (size,) or not torch.isfinite(vector).all():
                raise ValueError(f'Invalid {key} for {uid}')
            values.append(vector)
        actions.append(identity['operation'])
    if not actions:
        raise ValueError(f'Empty appearance split {name}')
    return [torch.stack(values) for values in vectors.values()], actions


def fit_model(fit, calibration, with_intention, epochs, batch, seed):
    torch.manual_seed(seed)
    model = VisualIntention()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.015)
    generator = torch.Generator().manual_seed(seed)
    best, best_epoch, best_state = float('inf'), 0, None
    for epoch in range(epochs + 1):
        if epoch:
            model.train()
            for index in torch.randperm(len(fit[0]), generator=generator).split(batch):
                src, ctx, intent, target = [x[index] for x in fit]
                if not with_intention:
                    intent = torch.zeros_like(intent)
                loss = objective(model, src, ctx, intent, target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
        model.eval()
        with torch.no_grad():
            src, ctx, intent, target = calibration
            pred = model(src, ctx, intent if with_intention else torch.zeros_like(intent))
            value = error(pred, target, src)[0].mean().item()
        if value < best:
            best, best_epoch = value, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise ValueError('No finite calibration score')
    model.load_state_dict(best_state)
    return model, {'epoch': best_epoch, 'calibration_error': best}


def swap_indices(ids, actions):
    """Rotate whole readouts across different photographs within each edit action."""
    result = torch.arange(len(ids))
    for action in sorted(set(actions)):
        indices = [i for i, value in enumerate(actions) if value == action]
        indices.sort(key=lambda i: hashlib.sha256(('appearance-swap/' + ids[i]).encode()).digest())
        if len(indices) < 2:
            raise ValueError('Each development action needs two distinct photographs')
        result[indices] = torch.tensor(indices[1:] + indices[:1])
    return result


@torch.inference_mode()
def compare(full, image_only, features, actions, ids):
    src, ctx, intent, target = features
    swapped = intent[swap_indices(ids, actions)]
    outputs = {'source': src, 'image_only': image_only(src, ctx, torch.zeros_like(intent)),
               'joint': full(src, ctx, intent), 'shuffled': full(src, ctx, swapped)}
    rows = [{'id': uid, 'operation': act} for uid, act in zip(ids, actions)]
    for arm, pred in outputs.items():
        cosine = F.cosine_similarity(pred, target)
        direction = error(pred, target, src)[1]
        for i, row in enumerate(rows):
            row[arm] = {'target': cosine[i].item(), 'direction': direction[i].item()}
    groups = {'all': rows}
    groups.update({a: [r for r in rows if r['operation'] == a] for a in sorted(set(actions))})
    summary = {}
    for group, samples in groups.items():
        summary[group] = {arm: {key: sum(r[arm][key] for r in samples) / len(samples)
                                for key in ('target', 'direction')} for arm in outputs}
    return summary, rows


def run(args):
    torch.set_num_threads(args.threads)
    reg = json.loads(args.registration.read_bytes())
    release = sha256(args.registration)
    seen = {k: set() for k in ('ids', 'photos', 'groups')}
    names = ('fit', 'calibration', 'development')
    if set(reg['samples']) != set(names):
        raise ValueError('Expected fit, calibration and development registrations')
    data = {s: read_split(args.pairs, s, reg['samples'][s], release, seen) for s in names}
    models, selected = {}, {}
    for arm in ('image_only', 'joint'):
        models[arm], selected[arm] = fit_model(data['fit'][0], data['calibration'][0],
                                               arm == 'joint', args.epochs, args.batch_size, args.seed)
    summary, rows = compare(models['joint'], models['image_only'], *data['development'],
                            reg['samples']['development'])
    rule = reg['rule']
    def gain(s, metric, control):
        return summary[s]['joint'][metric] - summary[s][control][metric]
    passed = (gain('all', 'target', 'image_only') >= rule['target_gain']
              and gain('all', 'direction', 'image_only') >= rule['direction_gain']
              and gain('all', 'direction', 'shuffled') >= rule['swapped_direction_gain']
              and all(gain(s, 'target', 'image_only') >= rule['min_action_target_gain']
                      for s in summary if s != 'all'))
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / 'adapter.pt'
    state = {'state': models['joint'].state_dict(), 'selected': selected['joint'], 'registration': release}
    atomic_write(checkpoint, lambda p: torch.save(state, p))
    result = dict(registration=release, implementation=sha256(__file__), checkpoint=sha256(checkpoint),
                  samples={s: len(data[s][1]) for s in names}, seed=args.seed, epochs=args.epochs,
                  batch_size=args.batch_size, selected=selected, metrics=summary, pairs=rows, passes=passed)
    atomic_write(args.out / 'development.json', lambda p: p.write_text(json.dumps(result, indent=2) + '\n'))
    print(json.dumps({k: result[k] for k in ('samples', 'metrics', 'passes')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('registration', 'pairs', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=3)
    options = parser.parse_args()
    if min(options.epochs, options.batch_size, options.threads) < 1:
        parser.error('Epochs, batch size and threads must be positive')
    run(options)
