"""Cache frozen Qwen visual and instruction features on a merged training partition."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile

import torch
import transformers
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from experiments.qwen35 import extract_grounding_features as extractor
from experiments.qwen35.fetch_assets import verify
from training.paper_utils import sha256


def partition_rows(merged, manifests, partition):
    outer = json.loads((merged / 'split-lock.json').read_bytes())
    inner = json.loads((manifests / 'split-lock.json').read_bytes())
    if (inner['parent_train_sha256'] != outer['manifests']['train.jsonl']
            or inner['heldout_test_sha256'] != outer['manifests']['test.jsonl']):
        raise ValueError('These training partitions belong to a different merged dataset')
    path = manifests / (partition + '.jsonl')
    expected = inner['splits'][partition]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected['sha256']:
        raise ValueError(f'Partition checksum mismatch for {path}')
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != expected['count'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError(f'Unexpected number of unique sources in {path}')
    return rows, expected['sha256']


def atomic_write(path, write):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        tmp = Path(temporary.name)
    try:
        write(tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def json_bytes(doc):
    return (json.dumps(doc, sort_keys=True, indent=2) + '\n').encode()


def validate(item, row):
    text_hash = hashlib.sha256(row['instruction'].encode()).hexdigest()
    if item['source_sha256'] != row['source_sha256'] or item['instruction_sha256'] != text_hash:
        raise ValueError(f'Features do not match source and instruction for {row["id"]}')
    if (item['visual'].shape != (256, 2560) or item['readout'].shape != (2560,)
            or item['text'].ndim != 2 or item['text'].shape[1] != 2560
            or not 1 <= len(item['text']) <= 128 or item['native_prior'].shape != (256,)):
        raise ValueError(f'Unexpected feature shape for {row["id"]}')
    if (not all(torch.isfinite(item[k]).all() for k in ('visual', 'readout', 'text', 'native_prior'))
            or item['native_prior'].min() < 0 or abs(item['native_prior'].sum().item()-1) > 1e-4):
        raise ValueError(f'Invalid frozen features for {row["id"]}')


def load_model(path, device):
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation='sdpa',
        local_files_only=True).to(device).eval().requires_grad_(False)
    if model.lm_head.weight is not model.get_input_embeddings().weight:
        model.lm_head.to('cpu')
    model.model.language_model.config._attn_implementation = 'eager'
    return model, processor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('model', 'merged', 'manifests', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--partition', choices=('fit', 'development'), required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--num-threads', default=2, type=int)
    p.add_argument('--shard', default=0, type=int)
    p.add_argument('--shards', default=1, type=int)
    p.add_argument('--limit', type=int, help='Process the first N rows of this shard')
    a = p.parse_args()
    if not 0 <= a.shard < a.shards or a.num_threads < 1 or (a.limit is not None and a.limit < 1):
        p.error('Invalid shard, thread count, or limit')
    torch.set_num_threads(a.num_threads)
    device = torch.device(a.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    origin = json.loads((a.model / 'origin.json').read_bytes())
    pinned = json.loads(Path(extractor.__file__).with_name('assets.json').read_bytes())
    if origin != pinned or not all(verify(a.model / f['path'], f) for f in pinned['files']):
        raise ValueError('Expected the verified Qwen3.5-4B checkpoint')
    rows, manifest_hash = partition_rows(a.merged, a.manifests, a.partition)
    out = a.output / a.partition
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(partition=a.partition, manifest_sha256=manifest_hash,
                    checkpoint=origin['revision'], extractor_sha256=sha256(extractor.__file__),
                    torch=str(torch.__version__), transformers=transformers.__version__,
                    backend=device.type, dtype='bfloat16', shards=a.shards)
    identity_path = out / 'identity.json'
    with (out / '.identity.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if identity_path.exists():
            if json.loads(identity_path.read_bytes()) != identity:
                raise ValueError(f'Incompatible frozen feature cache at {out}')
        else:
            atomic_write(identity_path, lambda path: path.write_bytes(json_bytes(identity)))
    model = None
    shard_rows = rows[a.shard::a.shards]
    selected = shard_rows[:a.limit]
    with (out / f'.shard-{a.shard}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt_path = out / f'complete-{a.shard}.json'
        previous = json.loads(receipt_path.read_bytes()) if receipt_path.exists() else None
        if previous is not None and (
                previous['identity_sha256'] != sha256(identity_path)
                or previous['count'] != len(shard_rows)
                or set(previous['sha256']) != {r['id'] for r in shard_rows}):
            raise ValueError(f'Invalid completion receipt at {receipt_path}')
        for i, row in enumerate(selected, start=1):
            path = out / (row['id'] + '.pt')
            if previous is not None and (not path.is_file() or sha256(path) != previous['sha256'][row['id']]):
                raise ValueError(f'Previously completed frozen features changed: {path}')
            if path.exists():
                validate(torch.load(path, map_location='cpu', weights_only=True), row)
                continue
            src = a.merged / row['source']
            if sha256(src) != row['source_sha256']:
                raise ValueError(f'Source changed for {row["id"]}')
            if model is None:
                model, processor = load_model(a.model, device)
            item = extractor.encode_pair_source(model, processor, src, row['instruction'], device)
            validate(item, row)
            atomic_write(path, lambda tmp: torch.save(item, tmp))
            if i == 1 or i % 10 == 0:
                print(a.partition, i, '/', len(selected), row['id'], flush=True)
        if previous is None and len(selected) == len(shard_rows):
            hashes = {r['id']: sha256(out / (r['id'] + '.pt')) for r in selected}
            receipt = dict(identity_sha256=sha256(identity_path), count=len(hashes), sha256=hashes)
            atomic_write(receipt_path, lambda p: p.write_bytes(json_bytes(receipt)))
        print('Verified', len(selected), 'source-only feature records', flush=True)


if __name__ == '__main__':
    main()
