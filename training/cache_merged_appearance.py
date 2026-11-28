"""Cache regional visual-prompt supervision for training.train_appearance_adapter.

Use a frozen spatial head with a 'model' state dictionary, cached Qwen readouts
and the ViT-H encoder from h94/IP-Adapter. Supply the photo manifest used to
initialize the spatial head to exclude those photos from the appearance study.
Use --extend-from with a previous registration to keep its calibration and
development cases while fitting on all remaining eligible training photos.
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
PILOT_CACHE_SHA256 = 'f9cc723007bb3bcd6fddd152e1b1a31667c5432615a29cbe865748cf9418881a'


def write_json(path, content):
    atomic_write(path, lambda tmp: tmp.write_text(json.dumps(content, indent=2) + '\n'))


def expanded_samples(previous, partitions, binding, excluded, gate):
    parent = previous['binding']
    if (parent['grounder'] != binding['grounder'] or parent['vision'] != binding['vision']
            or previous['rule'] != gate):
        raise ValueError('Use the same frozen crop head, vision encoder and selection rule as the pilot')
    # The initial regional-appearance study also recorded these identities in flat fields.
    exclusions = parent['exclusions'] if 'exclusions' in parent else [parent['warm']]
    if exclusions != binding['exclusions']:
        raise ValueError('The excluded training photographs differ from the pilot')
    for split in partitions:
        expected = (parent[split] if split in parent else
                    dict(manifest=parent['partitions'][split], readouts=parent[split + '_readouts']))
        if expected != binding[split]:
            raise ValueError('Different photograph manifest or Qwen readouts for ' + split)
    samples = previous['samples']
    calibration = samples['calibration']
    held = set(calibration)
    eligible = [r['id'] for r in partitions['fit'][0] if r.get('operation') in ACTIONS and r['id'] not in excluded]
    fit = [uid for uid in eligible if uid not in held]
    dev = {r['id'] for r in partitions['development'][0] if r.get('operation') in ACTIONS}
    if (not held or len(held) != len(calibration) or not held.issubset(eligible)
            or not samples['fit'] or not set(samples['fit']).issubset(fit)
            or len(samples['fit']) != len(set(samples['fit']))
            or len(fit) <= len(samples['fit']) or set(samples['development']) != dev
            or len(samples['development']) != len(dev)):
        raise ValueError('The fixed pilot split cannot be extended on these photographs')
    return dict(fit=fit, calibration=calibration, development=samples['development'])


def complete_readouts(folder, rows):
    identity = folder / 'identity.json'
    release = sha256(identity)
    expected = {r['id'] for r in rows}
    hashes = {}
    for rank in range(json.loads(identity.read_bytes())['shards']):
        file = folder / f'complete-{rank}.json'
        receipt = json.loads(file.read_bytes())
        chunk = receipt['sha256']
        if receipt['identity_sha256'] != release or hashes.keys() & chunk.keys():
            raise ValueError('Overlapping or stale native feature shard: ' + str(file))
        hashes.update(chunk)
    if set(hashes) != expected:
        raise ValueError('The feature receipts do not cover the complete photograph manifest')
    return hashes


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
    extend = getattr(args, 'extend_from', None)
    expected_samples = None
    if extend is not None:
        if extend.resolve() == path.resolve():
            raise ValueError('Use a separate output directory for the extended training run')
        for key in ('seed', 'fit_per_action', 'calibration_per_action'):
            del binding[key]
        binding['extended_from'] = sha256(extend)
        expected_samples = expanded_samples(json.loads(extend.read_bytes()), partitions, binding, excluded, gate)
        if not path.exists():
            write_json(path, dict(binding=binding, samples=expected_samples, rule=gate))
    elif not path.exists():
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
    if extend is None and saved['binding']['implementation'] == PILOT_CACHE_SHA256:
        # Existing pilot caches bind the original byte-identical feature calculation.
        binding['implementation'] = PILOT_CACHE_SHA256
    if (saved['binding'] != binding or saved['rule'] != gate
            or (expected_samples is not None and saved['samples'] != expected_samples)):
        raise ValueError('Output directory already contains a different appearance registration')
    return saved['samples'], sha256(path), excluded


@torch.inference_mode()
def prepare(args):
    torch.set_num_threads(args.threads)
    parts = {s: partition_rows(args.merged, args.manifests, s) for s in ('fit', 'development')}
    state = torch.load(args.grounder, weights_only=True, map_location='cpu')
    readouts = ({s: complete_readouts(args.features / s, records) for s, (records, _) in parts.items()}
                if getattr(args, 'extend_from', None) is not None else {})
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
        expected = None
        if split == 'development':
            expected = Counter(r.get('operation') for r in rows.values() if r.get('operation') in ACTIONS)
        elif not readouts:
            expected = dict.fromkeys(ACTIONS, args.fit_per_action if split == 'fit' else args.calibration_per_action)
        if expected is not None and counts != expected:
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
            feature_sha = sha256(feature_path)
            if readouts and feature_sha != readouts[partition][uid]:
                raise ValueError('Qwen features differ from the completed shard for ' + uid)
            identity.update(registration=release, feature_sha256=feature_sha)
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
    parser.add_argument('--extend-from', type=Path, metavar='REGISTRATION',
                        help='Use every eligible fit picture with a completed feature receipt and preserve the calibration and development splits in REGISTRATION')
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
