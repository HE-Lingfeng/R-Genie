"""CUDA DDP training with fixed validation masks, atomic resume and exposure logs."""
import argparse
import contextlib
import fcntl
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from training.paper_utils import (CachedEditingDataset, collate_cached, distributed_device,
                                 load_model, load_tokenizer, require_weights, sha256, text_batch,
                                 validation_masks)


def atomic_save(value, path):
    path = Path(path)
    torch.save(value, path.with_suffix('.tmp'))
    path.with_suffix('.tmp').replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state(state['cuda'].cpu())


@torch.no_grad()
def validate(model, loader, tokenizer, special, config, device, rank, world):
    model.eval()
    total = torch.zeros(2, device=device, dtype=torch.float64)
    for ids, instructions, source, target in loader:
        text, valid = text_batch(tokenizer, instructions, device, config.dataset.max_text_length)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = model(source.to(device), target.to(device), text, valid, special,
                           mask_positions=validation_masks(ids, target.size(1), config.training.validation_seed, device),
                           compute_alignment=False)
        total[0] += result['reconstruction'].double() * len(source)
        total[1] += len(source)
    if world > 1:
        dist.all_reduce(total)
    return (total[0] / total[1]).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/paper_4090.yaml')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-steps', type=int, help='Bounded GPU smoke run; still writes a final checkpoint')
    p.add_argument('overrides', nargs='*')
    a = p.parse_args()
    config = OmegaConf.merge(OmegaConf.load(a.config), OmegaConf.from_dotlist(a.overrides))
    if config.training.precision != 'bf16':
        raise ValueError('This trainer uses BF16 autocast with FP32 trainable parameters')
    device, rank, world = distributed_device()
    if world != config.training.expected_world_size:
        raise ValueError(f'Expected {config.training.expected_world_size} GPUs, launched {world} ranks')
    output = Path(config.experiment.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = None
    if rank == 0:
        lock = (output / '.train.lock').open('w')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'identity.json').exists() and not a.resume:
            raise FileExistsError('Run already exists; use --resume or a fresh output directory')
    if world > 1:
        dist.barrier()
    train = CachedEditingDataset(Path(config.dataset.manifests) / 'train.jsonl', config.dataset.cache)
    validation = CachedEditingDataset(Path(config.dataset.manifests) / 'validation.jsonl', config.dataset.cache)
    sampler = DistributedSampler(train, world, rank, shuffle=True, seed=config.training.seed, drop_last=True)
    loader = DataLoader(train, batch_size=config.training.batch_size, sampler=sampler, drop_last=True,
                        collate_fn=collate_cached, num_workers=config.training.num_workers,
                        pin_memory=True, persistent_workers=config.training.num_workers > 0)
    val_loader = DataLoader(Subset(validation, list(range(rank, len(validation), world))),
                            batch_size=config.training.batch_size, collate_fn=collate_cached,
                            num_workers=config.training.num_workers, pin_memory=True)
    if len(loader) == 0 or world * config.training.batch_size < 2:
        raise ValueError('Insufficient data/pairs for training and HA negatives')
    accumulation = config.training.gradient_accumulation_steps
    updates_per_epoch = math.ceil(len(loader) / accumulation)
    total_steps = updates_per_epoch * config.training.epochs
    seed_all(config.training.seed)
    tokenizer, special = load_tokenizer(config.model.tokenizer_path)
    model = load_model(config).to(device)
    # Frozen base in BF16; trainable weights/Adam moments in FP32.
    for parameter in model.parameters():
        if not parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.training.lr, betas=tuple(config.training.betas),
                                  weight_decay=config.training.weight_decay)
    warmup = max(1, round(total_steps * config.training.warmup_ratio))
    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, total_steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    identity = {'config': OmegaConf.to_container(config, resolve=True), 'world_size': world,
                'effective_batch': world * config.training.batch_size * accumulation,
                'trainable_parameters': sum(p.numel() for p in parameters),
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'updates_per_epoch': updates_per_epoch, 'total_steps': total_steps,
                'samples_per_epoch': len(loader) * config.training.batch_size * world,
                'optimization_samples': len(train), 'validation_samples': len(validation),
                'token_cache': train.metadata,
                'base_weights_sha256': {p.name: sha256(p) for p in require_weights(config.model.showo_path)},
                'source_sha256': {str(p): sha256(p) for folder in ('models', 'training')
                                   for p in sorted(Path(folder).glob('*.py'))},
                'torch': torch.__version__, 'gpu': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                'selection_metric': 'fixed_mask_validation_reconstruction_CE_not_Table1',
                'protocol': json.loads((Path(config.dataset.manifests) / 'audit.json').read_text())['protocol']}
    start_epoch, global_step, best = 0, 0, float('inf')
    resume_state = None
    if a.resume:
        # Only load our own local checkpoint: includes Python/NumPy RNG state.
        resume_state = torch.load(output / 'latest.pt', map_location='cpu', weights_only=False)
        if resume_state['identity'] != identity:
            raise ValueError('Resume identity changed: data/config/code/base weights must match exactly')
        model.load_state_dict(resume_state['model'], strict=False)
        optimizer.load_state_dict(resume_state['optimizer'])
        scheduler.load_state_dict(resume_state['scheduler'])
        start_epoch, global_step, best = resume_state['next_epoch'], resume_state['global_step'], resume_state['best']
    if rank == 0:
        (output / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
        OmegaConf.save(config, output / 'config.yaml')
        print(json.dumps({k: identity[k] for k in ('effective_batch', 'trainable_parameters', 'total_steps', 'samples_per_epoch')}), flush=True)
    if world > 1:
        wrapped = DDP(model, device_ids=[device.index], broadcast_buffers=False)
    else:
        wrapped = model
    # DDP construction broadcasts rank-zero weights. Restore per-rank RNG only after setup.
    if resume_state:
        restore_rng(resume_state['rng_by_rank'][rank])
    else:
        seed_all(config.training.seed + rank)
    initial = validate(model, val_loader, tokenizer, special, config, device, rank, world)
    if rank == 0:
        print(json.dumps({'initial_validation_CE': initial, 'global_step': global_step}), flush=True)
    start_time = time.monotonic()
    stopped = False
    for epoch in range(start_epoch, config.training.epochs):
        sampler.set_epoch(epoch)
        wrapped.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = torch.zeros(4, device=device, dtype=torch.float64)
        for microstep, (_, instructions, source, target) in enumerate(loader):
            group_start = (microstep // accumulation) * accumulation
            group_size = min(accumulation, len(loader) - group_start)
            boundary = (microstep - group_start + 1 == group_size)
            sync = contextlib.nullcontext() if boundary or world == 1 else wrapped.no_sync()
            text, valid = text_batch(tokenizer, instructions, device, config.dataset.max_text_length)
            with sync:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = wrapped(source.to(device), target.to(device), text, valid, special)
                    loss = result['loss'] / group_size
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at epoch={epoch} microstep={microstep}')
                loss.backward()
            epoch_loss += torch.stack([result[k].detach().double() for k in
                                        ('loss', 'reconstruction', 'alignment', 'mask_fraction')])
            if boundary:
                norm = torch.nn.utils.clip_grad_norm_(parameters, config.training.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if rank == 0 and (global_step == 1 or global_step % 10 == 0):
                    print(json.dumps({'epoch': epoch + 1, 'step': global_step,
                                      'loss': result['loss'].item(), 'rec': result['reconstruction'].item(),
                                      'ha': result['alignment'].item(), 'grad_norm': norm.item(),
                                      'lr': scheduler.get_last_lr()[0],
                                      'peak_GiB': torch.cuda.max_memory_allocated() / 2**30}), flush=True)
                if a.max_steps and global_step >= a.max_steps:
                    stopped = True
                    break
        if world > 1:
            dist.all_reduce(epoch_loss)
        # A partial epoch is recorded as partial, never an epoch-resumable checkpoint.
        val = validate(model, val_loader, tokenizer, special, config, device, rank, world)
        is_best = val < best
        best = min(best, val)
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, rng_state())
        else:
            states[0] = rng_state()
        if rank == 0:
            metrics = {'epoch': epoch + 1, 'global_step': global_step, 'validation_CE': val,
                       'best_validation_CE': best, 'partial_epoch': stopped,
                       'elapsed_seconds': time.monotonic() - start_time,
                       'mean_train': (epoch_loss / (world * (microstep + 1))).tolist()}
            print(json.dumps(metrics), flush=True)
            with (output / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(metrics) + '\n')
            state = {'model': model.trainable_state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'next_epoch': epoch + 1,
                     'global_step': global_step, 'best': best, 'identity': identity, 'rng_by_rank': states}
            if stopped:
                atomic_save(state, output / 'smoke.pt')
            else:
                atomic_save(state, output / 'latest.pt')
                if is_best:
                    atomic_save({'model': state['model'], 'identity': identity,
                                 'validation_CE': val, 'epoch': epoch + 1}, output / 'best.pt')
                if (epoch + 1) % config.training.save_every == 0:
                    atomic_save({'model': state['model'], 'identity': identity,
                                 'validation_CE': val, 'epoch': epoch + 1}, output / f'epoch-{epoch + 1:03d}.pt')
        if world > 1:
            dist.barrier()
        if stopped:
            break
    if rank == 0:
        marker = 'SMOKE_COMPLETE.json' if stopped else 'TRAINING_COMPLETE.json'
        (output / marker).write_text(json.dumps({'global_step': global_step, 'best_validation_CE': best,
                                                'table1_evaluated': False}, indent=2) + '\n')
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
