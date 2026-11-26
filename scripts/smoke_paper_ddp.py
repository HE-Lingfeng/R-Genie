"""Synthetic two-GPU infrastructure test, not a pretrained-model result."""
import contextlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import PhiConfig

from models.paper_rgenie import PaperRGenie
from models.phi import PhiForCausalLM
from training.paper_utils import distributed_device


def main():
    device, rank, world = distributed_device()
    if world != 2:
        raise ValueError('This smoke test expects exactly two GPUs')
    torch.manual_seed(123)
    config = PhiConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                       num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                       partial_rotary_factor=0.5, resid_pdrop=0, embd_pdrop=0, attention_dropout=0)
    config._attn_implementation = 'sdpa'
    model = PaperRGenie(PhiForCausalLM(config), visual_offset=40, codebook_size=16,
                        mask_id=63, dimension=8, heads=2, depth=2, adapter_width=8,
                        lora_rank=2, lora_alpha=4, lora_dropout=0).to(device)
    for p in model.parameters():
        if not p.requires_grad:
            p.data = p.data.to(torch.bfloat16)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    wrapper = DDP(model, device_ids=[device.index], broadcast_buffers=False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    losses = []
    torch.manual_seed(100 + rank)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(2):
            source = torch.randint(0, 16, (2, 9), device=device)
            target = source.roll(1, dims=1)
            text = torch.randint(1, 5, (2, 4), device=device)
            with (wrapper.no_sync() if micro == 0 else contextlib.nullcontext()):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = wrapper(source, target, text, torch.ones_like(text, dtype=torch.bool),
                                     dict(mmu=5, soi=6, eoi=7, t2i=8, pad=0), mask_ratio=0.5)
                (result['loss'] / 2).backward()
            losses.append(result['loss'].item())
        torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
        optimizer.step()
    flat = torch.cat([p.detach().float().flatten() for p in parameters])
    minimum, maximum = flat.clone(), flat.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    mismatch = (maximum - minimum).abs().max().item()
    if mismatch > 1e-6:
        raise AssertionError(f'DDP parameter mismatch: {mismatch}')
    for n, p in model.named_parameters():
        if n in frozen and (p.grad is not None or not torch.equal(p, frozen[n])):
            raise AssertionError(f'Frozen weight changed: {n}')
    if rank == 0:
        report = {'test': 'synthetic_not_pretrained', 'world_size': world,
                  'torch': torch.__version__, 'losses_rank0': losses,
                  'max_parameter_difference_between_ranks': mismatch,
                  'frozen_weights_unchanged': True, 'checkpointing': True,
                  'precision': 'BF16 frozen weights/autocast, FP32 adapters',
                  'distributed_HA_negatives': 4}
        dest = Path('artifacts/setup/DDP_SMOKE_COMPLETE.json')
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
