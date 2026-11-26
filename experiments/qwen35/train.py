"""DDP training of a Qwen3.5 latent bridge with weak paired-region supervision."""
import argparse
import contextlib
import fcntl
import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from experiments.qwen35.data import BridgeDataset, collate_bridge
from experiments.qwen35.model import QwenBridgeEditor
from models.modeling_showo import Showo
from training.paper_utils import (distributed_device, load_tokenizer, require_weights, sha256,
                                 text_batch, validation_masks)
from training.train_paper import atomic_save, restore_rng, rng_state, seed_all


def load_bridge(config):
    require_weights(config.model.showo_path)
    base, loading = Showo.from_pretrained(config.model.showo_path, low_cpu_mem_usage=False,
                                         output_loading_info=True, llm_model_path=config.model.tokenizer_path)
    if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'Pretrained denoiser did not load exactly: {loading}')
    return QwenBridgeEditor(base.showo, feature_hidden=config.model.feature_hidden,
                            dimension=config.model.dimension, heads=config.model.heads,
                            depth=config.model.hrm_layers, adapter_width=config.model.adapter_width,
                            gradient_checkpointing=config.training.gradient_checkpointing,
                            region_weight=config.model.region_weight,
                            edited_token_weight=config.model.edited_token_weight,
                            native_prior_strength=config.model.native_prior_strength)


def prepare(batch, tokenizer, config, device):
    ids, instructions, _, targets, readout, visual, features, valid, region, native = batch
    text, text_valid = text_batch(tokenizer, instructions, device, config.dataset.max_text_length)
    return ids, dict(target_codes=targets.to(device), text_ids=text, text_valid=text_valid,
                     readout=readout.to(device), visual=visual.to(device), text_features=features.to(device),
                     text_feature_valid=valid.to(device), weak_region=region.to(device), native_prior=native.to(device))


@torch.no_grad()
def validate(model, loader, tokenizer, special, config, device, world):
    model.eval()
    sums = torch.zeros(6, device=device, dtype=torch.float64)
    for batch in loader:
        ids, inputs = prepare(batch, tokenizer, config, device)
        mask = validation_masks(ids, inputs['target_codes'].size(1), config.training.validation_seed, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = model(**inputs, special=special, mask_positions=mask, compute_alignment=False)
        predicted = model.predicted_mask(result['region_probability'], config.selection.region_threshold,
                                         config.selection.min_region_ratio, config.selection.max_region_ratio)
        truth = inputs['weak_region'] >= config.selection.weak_label_occupancy_threshold
        intersection = (predicted & truth).sum(1)
        union = (predicted | truth).sum(1).clamp_min(1)
        n = len(ids)
        sums[0] += result['reconstruction'].double() * n
        sums[1] += result['edited_reconstruction'].double() * n
        sums[2] += (intersection.double() / union).sum()
        sums[3] += predicted.double().mean(1).sum()
        sums[4] += inputs['weak_region'].double().mean(1).sum()
        sums[5] += n
    if world > 1:
        dist.all_reduce(sums)
    means = (sums[:5] / sums[5]).tolist()
    score = means[1] + config.selection.iou_weight * (1 - means[2])
    return dict(selection_score=score, validation_CE=means[0], edited_validation_CE=means[1],
                weak_region_IoU=means[2], predicted_area=means[3], paired_change_area=means[4])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='experiments/qwen35/config.yaml')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-steps', type=int)
    p.add_argument('overrides', nargs='*')
    a = p.parse_args()
    config = OmegaConf.merge(OmegaConf.load(a.config), OmegaConf.from_dotlist(a.overrides))
    device, rank, world = distributed_device()
    if world != config.training.expected_world_size:
        raise ValueError('Unexpected distributed world size')
    output = Path(config.experiment.output)
    output.mkdir(parents=True, exist_ok=True)
    run_lock = None
    if rank == 0:
        run_lock = (output / '.train.lock').open('w')
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'identity.json').exists() and not a.resume:
            raise FileExistsError('Existing run: inspect state, use --resume, or choose a new output')
    if world > 1:
        dist.barrier()
    train = BridgeDataset(Path(config.dataset.manifests) / 'train.jsonl', config.dataset.cache,
                           config.dataset.features, config.dataset.regions)
    validation = BridgeDataset(Path(config.dataset.manifests) / 'validation.jsonl', config.dataset.cache,
                                config.dataset.features, config.dataset.regions)
    sampler = DistributedSampler(train, world, rank, shuffle=True, seed=config.training.seed, drop_last=True)
    loader = DataLoader(train, batch_size=config.training.batch_size, sampler=sampler, drop_last=True,
                        num_workers=config.training.num_workers, collate_fn=collate_bridge, pin_memory=True,
                        persistent_workers=config.training.num_workers > 0)
    val_loader = DataLoader(Subset(validation, list(range(rank, len(validation), world))),
                            batch_size=config.training.batch_size, num_workers=config.training.num_workers,
                            collate_fn=collate_bridge, pin_memory=True)
    if not len(loader):
        raise ValueError('No full training batches')
    accumulation = config.training.gradient_accumulation_steps
    updates = math.ceil(len(loader) / accumulation)
    total = updates * config.training.epochs
    seed_all(config.training.seed)
    tokenizer, special = load_tokenizer(config.model.tokenizer_path)
    model = load_bridge(config).to(device)
    for parameter in model.parameters():
        if not parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.training.lr, betas=tuple(config.training.betas),
                                  weight_decay=config.training.weight_decay)
    warmup = max(1, round(total * config.training.warmup_ratio))
    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, total - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    identity = {'variant': 'frozen_Qwen3.5_latent_bridge_with_weak_region_auxiliary',
                'config': OmegaConf.to_container(config, resolve=True), 'world_size': world,
                'effective_batch': world * config.training.batch_size * accumulation,
                'trainable_parameters': sum(p.numel() for p in parameters),
                'decoder_and_bridge_parameters': sum(p.numel() for p in model.parameters()),
                'native_Qwen_backbone_updated': False, 'generated_reasoning_text': False,
                'samples_per_epoch': len(loader) * config.training.batch_size * world,
                'total_steps': total, 'token_cache': train.base.metadata,
                'feature_identity': train.feature_identity, 'feature_manifest': train.feature_complete,
                'weak_region_identity': train.region_identity,
                'base_weights_sha256': {q.name: sha256(q) for q in require_weights(config.model.showo_path)},
                'source_sha256': {str(q): sha256(q) for folder in ('models', 'training', 'experiments/qwen35')
                                   for q in sorted(Path(folder).glob('*.py'))},
                'torch': torch.__version__, 'test_used_for_selection': False,
                'protocol': 'provisional_release_split_not_table1'}
    epoch_start, global_step, best = 0, 0, float('inf')
    resumed = None
    if a.resume:
        resumed = torch.load(output / 'latest.pt', map_location='cpu', weights_only=False)
        if resumed['identity'] != identity:
            raise ValueError('Resume identity changed')
        model.load_state_dict(resumed['model'], strict=False)
        optimizer.load_state_dict(resumed['optimizer'])
        scheduler.load_state_dict(resumed['scheduler'])
        epoch_start, global_step, best = resumed['next_epoch'], resumed['global_step'], resumed['best']
    if rank == 0:
        (output / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
        OmegaConf.save(config, output / 'config.yaml')
    wrapper = DDP(model, device_ids=[device.index], broadcast_buffers=False) if world > 1 else model
    if resumed:
        restore_rng(resumed['rng_by_rank'][rank])
    else:
        seed_all(config.training.seed + rank)
    initial = validate(model, val_loader, tokenizer, special, config, device, world)
    if rank == 0:
        print(json.dumps({'initial_validation': initial, 'trainable_parameters': identity['trainable_parameters']}), flush=True)
    started, partial = time.monotonic(), False
    for epoch in range(epoch_start, config.training.epochs):
        sampler.set_epoch(epoch)
        wrapper.train()
        optimizer.zero_grad(set_to_none=True)
        for micro, batch in enumerate(loader):
            group_start = micro // accumulation * accumulation
            group_size = min(accumulation, len(loader) - group_start)
            boundary = micro - group_start + 1 == group_size
            _, inputs = prepare(batch, tokenizer, config, device)
            sync = contextlib.nullcontext() if boundary or world == 1 else wrapper.no_sync()
            with sync:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = wrapper(**inputs, special=special)
                    loss = result['loss'] / group_size
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite bridge loss')
                loss.backward()
            if boundary:
                norm = torch.nn.utils.clip_grad_norm_(parameters, config.training.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if rank == 0 and (global_step == 1 or global_step % 20 == 0):
                    print(json.dumps({'epoch': epoch + 1, 'step': global_step, 'loss': result['loss'].item(),
                                      'reconstruction': result['reconstruction'].item(),
                                      'localization': result['localization'].item(), 'alignment': result['alignment'].item(),
                                      'gradient_norm': norm.item(), 'peak_GiB': torch.cuda.max_memory_allocated() / 2**30}), flush=True)
                if a.max_steps and global_step >= a.max_steps:
                    partial = True
                    break
        metrics = validate(model, val_loader, tokenizer, special, config, device, world)
        improved = metrics['selection_score'] < best
        best = min(best, metrics['selection_score'])
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, rng_state())
        else:
            states[0] = rng_state()
        if rank == 0:
            report = dict(epoch=epoch + 1, step=global_step, partial_epoch=partial,
                          elapsed_seconds=time.monotonic() - started, **metrics)
            print(json.dumps(report), flush=True)
            with (output / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(report) + '\n')
            checkpoint = dict(model=model.trainable_state_dict(), optimizer=optimizer.state_dict(),
                              scheduler=scheduler.state_dict(), next_epoch=epoch + 1, global_step=global_step,
                              best=best, rng_by_rank=states, identity=identity)
            if partial:
                atomic_save(checkpoint, output / 'smoke.pt')
            else:
                atomic_save(checkpoint, output / 'latest.pt')
                selected = dict(model=checkpoint['model'], identity=identity, epoch=epoch + 1, validation=metrics)
                if improved:
                    atomic_save(selected, output / 'best.pt')
                if (epoch + 1) % config.training.save_every == 0:
                    atomic_save(selected, output / f'epoch-{epoch + 1:03d}.pt')
        if world > 1:
            dist.barrier()
        if partial:
            break
    if rank == 0:
        (output / ('SMOKE_COMPLETE.json' if partial else 'TRAINING_COMPLETE.json')).write_text(
            json.dumps({'global_step': global_step, 'best_selection_score': best,
                        'table1_evaluated': False, 'test_unsealed': False}, indent=2) + '\n')
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
