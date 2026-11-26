import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_records(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [r['id'] for r in records]
    if not records or len(ids) != len(set(ids)):
        raise ValueError(f'Empty manifest or repeated IDs: {path}')
    return records


def distributed_device():
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if not torch.cuda.is_available():
        raise RuntimeError('Full training/token caching requires CUDA; run tests for CPU verification')
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group('nccl', device_id=torch.device('cuda', local_rank))
    return torch.device('cuda', local_rank), rank, world


def image_tensor(path, resolution=256):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('RGB').resize((resolution, resolution), Image.Resampling.BICUBIC)
        array = np.array(im, dtype=np.float32, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1


def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left', local_files_only=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokens = ('<|soi|>', '<|eoi|>', '<|sov|>', '<|eov|>', '<|t2i|>', '<|mmu|>',
              '<|t2v|>', '<|v2v|>', '<|lvg|>')
    tokenizer.add_tokens(list(tokens))
    if len(tokenizer) != 50305:
        raise ValueError('The Show-o visual vocabulary must start at 50305; do not swap in a text-only tokenizer')
    special = {name: tokenizer.convert_tokens_to_ids(f'<|{name}|>') for name in ('soi', 'eoi', 't2i', 'mmu')}
    special['pad'] = tokenizer.pad_token_id
    return tokenizer, special


def text_batch(tokenizer, instructions, device, max_length=128):
    rows = []
    for instruction in instructions:
        ids = tokenizer.encode(instruction, add_special_tokens=False)[:max_length - 2]
        rows.append([tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id])
    batch = tokenizer.pad({'input_ids': rows}, padding='max_length', max_length=max_length, return_tensors='pt')
    return batch.input_ids.to(device), batch.attention_mask.to(device).bool()


def validation_masks(ids, count, seed, device):
    """A sample's corruption is invariant to batch size, order and GPU sharding."""
    masks = []
    for sample_id in ids:
        key = f'{seed}:{sample_id}'.encode()
        sample_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') & ((1 << 63) - 1)
        generator = torch.Generator().manual_seed(sample_seed)
        fraction = math.cos(torch.rand((), generator=generator).item() * math.pi / 2)
        masked_count = max(1, min(count, round(fraction * count)))
        mask = torch.zeros(count, dtype=torch.bool)
        mask[torch.randperm(count, generator=generator)[:masked_count]] = True
        masks.append(mask)
    return torch.stack(masks).to(device)


def require_weights(path):
    path = Path(path)
    names = ('pytorch_model.bin', 'pytorch_model.safetensors', 'model.safetensors',
             'diffusion_pytorch_model.bin', 'diffusion_pytorch_model.safetensors')
    found = [path / name for name in names if (path / name).is_file()]
    if not found:
        raise FileNotFoundError(f'No pretrained weights in {path}; configs alone are not a pretrained model')
    return found


def load_model(config):
    from models.modeling_showo import Showo
    from models.paper_rgenie import PaperRGenie
    require_weights(config.model.showo_path)
    showo, loading = Showo.from_pretrained(config.model.showo_path, low_cpu_mem_usage=False,
                                         output_loading_info=True, llm_model_path=config.model.tokenizer_path)
    if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError(f'Pretrained Show-o did not load exactly: {loading}')
    return PaperRGenie(showo.showo, dimension=config.model.dimension, heads=config.model.heads,
                       depth=config.model.hrm_layers, adapter_width=config.model.visual_adapter_width,
                       lora_rank=config.model.lora_rank, lora_alpha=config.model.lora_alpha,
                       lora_dropout=config.model.lora_dropout,
                       gradient_checkpointing=config.training.gradient_checkpointing)


class CachedEditingDataset(Dataset):
    def __init__(self, manifest, cache):
        self.records = read_records(manifest)
        self.cache = Path(cache)
        metadata = json.loads((self.cache / 'metadata.json').read_text())
        if metadata['manifest_sha256'].get(Path(manifest).stem) != sha256(manifest):
            raise ValueError('Token cache does not match this split manifest')
        self.metadata = metadata

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        item = torch.load(self.cache / (record['id'] + '.pt'), map_location='cpu', weights_only=True)
        for field in ('source_sha256', 'target_sha256'):
            if item[field] != record[field]:
                raise ValueError(f'Cache identity mismatch: {record["id"]} / {field}')
        return record['id'], record['instruction'], item['source'].long(), item['target'].long()


def collate_cached(batch):
    ids, instructions, sources, targets = zip(*batch)
    return list(ids), list(instructions), torch.stack(sources), torch.stack(targets)
