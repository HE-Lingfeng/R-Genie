"""Prepare 16x16 spatial supervision for the merged training and development sets."""
import argparse
from collections import Counter
import fcntl
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageOps


ACTIONS = ('anchored_add', 'enlarge', 'move', 'recolor', 'remove', 'replace', 'shrink')
GEOMETRIC = {'enlarge', 'move', 'shrink'}
VERSION = 1


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def verified_image(root, row, key, mode):
    raw = (root / row[key]).read_bytes()
    if digest(raw) != row[key + '_sha256']:
        raise ValueError(f'Changed {key} for {row["id"]}')
    with Image.open(BytesIO(raw)) as im:
        return ImageOps.exif_transpose(im).convert(mode)


def grid(binary):
    return binary.reshape(16, 32, 16, 32).mean((1, 3), dtype=np.float32)


def mask_grid(root, row, key):
    image = verified_image(root, row, key, 'L')
    if image.size != tuple(row['canvas']):
        raise ValueError(f'Canvas mismatch for {row["id"]}/{key}')
    return grid(np.asarray(image.resize((512, 512), Image.Resampling.NEAREST)) > 0)


def paired_change(root, row):
    rgb = [np.asarray(verified_image(root, row, k, 'RGB').resize(
        (512, 512), Image.Resampling.BICUBIC), dtype=np.int16) for k in ('source', 'target')]
    # Same RGB tolerance as the paired-change supervision in the original grounder.
    return grid(np.abs(rgb[0] - rgb[1]).max(-1) >= 13)


def prepare_partition(root, manifest, info, out):
    raw = manifest.read_bytes()
    manifest_hash = digest(raw)
    if manifest_hash != info['sha256']:
        raise ValueError(f'Unexpected partition hash for {manifest.name}')
    records = [json.loads(line) for line in raw.splitlines()]
    ids = np.asarray([r['id'] for r in records])
    n = len(records)
    if not n or n != info['count'] or len(set(ids)) != n:
        raise ValueError(f'Invalid record IDs in {manifest.name}')
    if out.exists():
        with np.load(out, allow_pickle=False) as prior:
            if (int(prior['schema_version']) != VERSION or str(prior['manifest_sha256']) != manifest_hash
                    or not np.array_equal(prior['ids'], ids)):
                raise ValueError(f'Incompatible spatial supervision at {out}')
        return
    source = np.zeros((n, 16, 16), dtype=np.float32)
    destination = np.zeros_like(source)
    edit = np.zeros_like(source)
    action = np.full(n, -1, dtype=np.int8)
    has_instance = np.zeros(n, dtype=bool)
    has_geometry = np.zeros(n, dtype=bool)
    requested_transform = np.zeros((n, 3), dtype=np.float32)
    destination_box = np.zeros((n, 4), dtype=np.float32)
    supervision = Counter()
    for i, row in enumerate(records):
        kind = row['supervision']
        supervision[kind] += 1
        if kind == 'paired-change-only':
            edit[i] = paired_change(root, row)
            continue
        if kind != 'native-instance-and-edit-region':
            raise ValueError(f'Unknown supervision {kind} for {ids[i]}')
        op = row['operation']
        action[i] = ACTIONS.index(op)
        has_instance[i] = True
        source[i] = mask_grid(root, row, 'target_mask')
        edit[i] = mask_grid(root, row, 'edit_region')
        if not source[i].any() or not edit[i].any():
            raise ValueError(f'Empty native annotation for {ids[i]}')
        if op in GEOMETRIC:
            has_geometry[i] = True
            destination[i] = mask_grid(root, row, 'after_mask')
            geo = row['geometry']
            destination_box[i] = geo['destination_box_xyxy']
            if op == 'move':
                requested_transform[i, :2] = (geo['dx'], geo['dy'])
            else:
                requested_transform[i, 2] = np.log(geo['scale'])
            if not destination[i].any() or not np.isfinite(requested_transform[i]).all():
                raise ValueError(f'Invalid geometry for {ids[i]}')
    if dict(supervision) != info['by_supervision']:
        raise ValueError(f'Supervision counts changed for {manifest.name}')
    with tempfile.NamedTemporaryFile(dir=out.parent, suffix='.npz', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.savez_compressed(tmp_path, schema_version=VERSION, manifest_sha256=manifest_hash, ids=ids,
                            actions=np.asarray(ACTIONS), action=action, source=source, edit=edit,
                            destination=destination, has_instance=has_instance, has_geometry=has_geometry,
                            requested_dx_dy_log_scale=requested_transform, destination_box_xyxy=destination_box)
        tmp_path.replace(out)
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--merged', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, required=True,
                        help='Directory created by training.prepare_merged_training')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    split = json.loads((args.manifests / 'split-lock.json').read_bytes())
    outer = json.loads((args.merged / 'split-lock.json').read_bytes())
    if (split['parent_train_sha256'] != outer['manifests']['train.jsonl'] or
            split['heldout_test_sha256'] != outer['manifests']['test.jsonl']):
        raise ValueError('The inner partitions belong to another merged split')
    args.output.mkdir(exist_ok=True, parents=True)
    with (args.output / '.regions.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name in ('fit', 'development'):
            out = args.output / (name + '.npz')
            prepare_partition(args.merged, args.manifests / (name + '.jsonl'), split['splits'][name], out)
            with np.load(out, allow_pickle=False) as archive:
                print(name, len(archive['ids']), 'instances', int(archive['has_instance'].sum()),
                      'geometric edits', int(archive['has_geometry'].sum()), flush=True)


if __name__ == '__main__':
    main()
