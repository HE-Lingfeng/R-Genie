"""Two-GPU, frozen-tokenizer caching of optimization and validation pairs only."""
import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist

from models import MAGVITv2
from training.paper_utils import distributed_device, image_tensor, read_records, require_weights, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--manifests', required=True)
    p.add_argument('--vq', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=4)
    a = p.parse_args()
    device, rank, world = distributed_device()
    output, manifests, root = Path(a.output), Path(a.manifests), Path(a.data)
    output.mkdir(parents=True, exist_ok=True)
    weights = require_weights(a.vq)
    records = read_records(manifests / 'train.jsonl') + read_records(manifests / 'validation.jsonl')
    identity = {'vq_weight_sha256': {p.name: sha256(p) for p in weights},
                'manifest_sha256': {s: sha256(manifests / (s + '.jsonl')) for s in ('train', 'validation')},
                'preprocessing': 'RGB_EXIF_resize_bicubic_256x256_normalize_minus1_plus1',
                'token_grid': [16, 16], 'codebook_size': 8192}
    if (output / 'metadata.json').exists():
        if json.loads((output / 'metadata.json').read_text()) != identity:
            raise ValueError('Existing token cache was produced by a different dataset/tokenizer/preprocessing')
    model, loading = MAGVITv2.from_pretrained(a.vq, low_cpu_mem_usage=False, output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError(f'VQ weights did not load exactly: {loading}')
    model = model.to(device).eval().requires_grad_(False)
    local_records = records[rank::world]
    with torch.inference_mode():
        for start in range(0, len(local_records), a.batch_size):
            batch = local_records[start:start + a.batch_size]
            for r in batch:
                for key in ('source', 'target'):
                    if sha256(root / r[key]) != r[key + '_sha256']:
                        raise ValueError(f'Input changed since audit: {r["id"]} / {key}')
            images = torch.stack([image_tensor(root / r[k]) for r in batch for k in ('source', 'target')]).to(device)
            # Keep VQ quantization in FP32: sign changes affect discrete code identity.
            codes = model.get_code(images).reshape(len(batch), 2, -1).cpu()
            if codes.size(-1) != 256 or codes.min() < 0 or codes.max() >= 8192:
                raise ValueError('Unexpected frozen VQ token grid/code range')
            for r, pair in zip(batch, codes):
                dest = output / (r['id'] + '.pt')
                item = {'source': pair[0].short(), 'target': pair[1].short(),
                        'source_sha256': r['source_sha256'], 'target_sha256': r['target_sha256']}
                torch.save(item, dest.with_suffix('.tmp'))
                dest.with_suffix('.tmp').replace(dest)
            print(f'rank={rank} cached={min(start + len(batch), len(local_records))}/{len(local_records)}', flush=True)
    if world > 1:
        dist.barrier()
    if rank == 0:
        (output / 'metadata.json').write_text(json.dumps(identity, indent=2) + '\n')
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
