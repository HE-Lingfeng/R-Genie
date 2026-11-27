"""Cache regional visual-prompt supervision for training.train_appearance_adapter.

Use a frozen spatial head with a 'model' state dictionary, cached Qwen readouts
and the ViT-H encoder from h94/IP-Adapter. Supply the photo manifest used to
initialize the spatial head to exclude those photos from the appearance study.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from training.cache_merged_grounding import atomic_write, partition_rows, validate
from training.cache_visual_intentions import ENCODER_SHA256, encode_images, encoder, open_image, views
from training.merged_grounder import MergedGrounder
from training.paper_utils import sha256
from training.train_merged_grounder import feature_inputs

ACTIONS = ('anchored_add', 'replace')


def write_json(path, content):
    atomic_write(path, lambda tmp: tmp.write_text(json.dumps(content, indent=2) + '\n'))


def register(args, partitions, state):
    excluded = set()
    for file in args.exclude_manifest:
        with file.open() as stream:
            excluded.update(json.loads(line)['id'] for line in stream if line.strip())
    parent = state.get('identity', {})
    for key in ('paired_sources', 'rehearsal_sources'):
        excluded.update(parent.get(key, ()))
    binding = dict(grounder=sha256(args.grounder), vision=ENCODER_SHA256, implementation=sha256(__file__),
                   exclusions=[sha256(p) for p in args.exclude_manifest], fit_per_action=args.fit_per_action,
                   calibration_per_action=args.calibration_per_action, seed=args.seed)
    for split, (_, manifest) in partitions.items():
        path = args.features / split / 'identity.json'
        metadata = json.loads(path.read_bytes())
        if (metadata['partition'], metadata['manifest_sha256'], metadata['backend']) != (split, manifest, 'cuda'):
            raise ValueError(f'Qwen feature cache mismatches {split}')
        binding[split] = {'manifest': manifest, 'readouts': sha256(path)}
    gate = dict(target_gain=.005, direction_gain=.02, swapped_direction_gain=.01, min_action_target_gain=0.)
    path = args.out / 'registration.json'
    if not path.exists():
        samples = {s: [] for s in ('fit', 'calibration', 'development')}
        for action in ACTIONS:
            eligible = [r['id'] for r in partitions['fit'][0]
                        if r.get('operation') == action and r['id'] not in excluded
                        and (args.features / 'fit' / (r['id'] + '.pt')).is_file()]
            eligible.sort(key=lambda uid: hashlib.sha256((args.seed + '/' + uid).encode()).digest())
            end = args.fit_per_action + args.calibration_per_action
            if len(eligible) < end:
                raise ValueError(f'Need {end} independent fit readouts for {action}, have {len(eligible)}')
            samples['fit'].extend(eligible[:args.fit_per_action])
            samples['calibration'].extend(eligible[args.fit_per_action:end])
            samples['development'].extend(r['id'] for r in partitions['development'][0] if r.get('operation') == action)
        write_json(path, dict(binding=binding, samples=samples, rule=gate))
    saved = json.loads(path.read_bytes())
    if saved['binding'] != binding or saved['rule'] != gate:
        raise ValueError('Output directory already contains a different appearance registration')
    return saved['samples'], sha256(path), excluded


@torch.inference_mode()
def prepare(args):
    torch.set_num_threads(args.threads)
    parts = {s: partition_rows(args.merged, args.manifests, s) for s in ('fit', 'development')}
    state = torch.load(args.grounder, weights_only=True, map_location='cpu')
    selections, release, excluded = register(args, parts, state)
    head = MergedGrounder().eval()
    head.load_state_dict(state['model'])
    bundle = encoder(args.encoder, torch.device('cpu'))
    seen_groups, seen_digests = set(), set()
    for split, ids in selections.items():
        partition = 'development' if split == 'development' else 'fit'
        rows = {r['id']: r for r in parts[partition][0]}
        group = [rows[uid] for uid in ids]
        counts = Counter(r['operation'] for r in group)
        if split == 'development':
            expected = Counter(r.get('operation') for r in rows.values() if r.get('operation') in ACTIONS)
        else:
            expected = dict.fromkeys(ACTIONS, args.fit_per_action if split == 'fit' else args.calibration_per_action)
        if counts != expected:
            raise ValueError('Registered action counts differ from the sampling plan')
        folder = args.out / 'pairs' / split
        folder.mkdir(exist_ok=True, parents=True)
        hashes = {}
        for row in group:
            uid = row['id']
            digests = {row['source_sha256'], row['target_sha256']}
            if uid in excluded or row['source_group'] in seen_groups or seen_digests & digests:
                raise ValueError(f'Overlapping training or development photograph {uid}')
            seen_groups.add(row['source_group'])
            seen_digests.update(digests)
            feature_path = args.features / partition / (uid + '.pt')
            identity = {k: row[k] for k in ('id', 'source_group', 'operation', 'source_sha256', 'target_sha256')}
            identity.update(registration=release, feature_sha256=sha256(feature_path))
            file = folder / (uid + '.pt')
            if file.is_file():
                if torch.load(file, weights_only=True, map_location='cpu')['identity'] != identity:
                    raise ValueError(f'Stale appearance features for {uid}')
            else:
                feat = torch.load(feature_path, weights_only=True, map_location='cpu')
                validate(feat, row)
                logits = head(*[v[None] for v in feature_inputs(feat)])['edit']
                prior = F.interpolate(logits.sigmoid().reshape(1, 1, 16, 16), (32, 32),
                                      mode='bilinear', align_corners=False)
                source, target = (args.merged / row[k] for k in ('source', 'target'))
                if (sha256(source), sha256(target)) != (row['source_sha256'], row['target_sha256']):
                    raise ValueError(f'Source or target image changed for {uid}')
                pictures, window = views(open_image(source), prior)
                src, ctx, tgt = encode_images(bundle, [*pictures, open_image(target).crop(window)])
                rec = dict(identity=identity, window=window, source=src.clone(), context=ctx.clone(),
                           target=tgt.clone(), intention=feat['readout'].float().clone())
                atomic_write(file, lambda tmp: torch.save(rec, tmp))
            hashes[uid] = sha256(file)
        write_json(folder / 'complete.json', dict(registration=release, sha256=hashes))
        print(split, len(group), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('merged', 'manifests', 'features', 'encoder', 'grounder', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--exclude-manifest', action='append', type=Path, default=[])
    parser.add_argument('--fit-per-action', type=int, default=128)
    parser.add_argument('--calibration-per-action', type=int, default=32)
    parser.add_argument('--seed', default='merged-appearance-1')
    parser.add_argument('--threads', type=int, default=4)
    options = parser.parse_args()
    if min(options.fit_per_action, options.calibration_per_action, options.threads) < 1:
        parser.error('Sample counts and thread count must be positive')
    options.out.mkdir(parents=True, exist_ok=True)
    with (options.out / 'cache.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(options)
