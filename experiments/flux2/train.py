"""DDP training of a frozen continuous-flow R-Genie extension."""
import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers import Flux2Transformer2DModel
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from experiments.flux2.conditioning import predicted_mask
from experiments.flux2.data import FlowDataset, collate_flow
from experiments.flux2.model import FluxIntentionEditor
from experiments.qwen35.fetch_assets import verify
from training.paper_utils import distributed_device, sha256
from training.train_paper import atomic_save, restore_rng, rng_state, seed_all


def load_editor(config):
    root = Path(config.model.path)
    origin = json.loads((root/'origin.json').read_text())
    if origin != json.loads(Path('experiments/flux2/assets.json').read_text()):
        raise ValueError('Unexpected native FLUX model revision')
    for entry in origin['files']:
        if entry['path'].startswith('transformer/') and not verify(root/entry['path'],entry):
            raise ValueError('Native transformer bytes differ from the pinned official object')
    transformer, info = Flux2Transformer2DModel.from_pretrained(Path(config.model.path)/'transformer',
        torch_dtype=torch.bfloat16, local_files_only=True, output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError(f'Native FLUX transformer did not load exactly: {info}')
    return FluxIntentionEditor(transformer, feature_hidden=config.model.feature_hidden,
        dimension=config.model.dimension, heads=config.model.heads, depth=config.model.depth,
        adapter_width=config.model.adapter_width, gradient_checkpointing=config.training.gradient_checkpointing)


def fixed_noise(ids, shape, seed, device):
    noise, times = [], []
    for sample_id in ids:
        key = hashlib.sha256(f'{seed}:{sample_id}'.encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(key[:8],'little') & ((1<<63)-1))
        noise.append(torch.randn(shape, generator=generator))
        times.append(torch.randn((), generator=generator).sigmoid())
    return torch.stack(noise).to(device), torch.stack(times).to(device)


def prepare(batch, device):
    return batch['ids'], {k:v.to(device, non_blocking=True) for k,v in batch.items() if k != 'ids'}


@torch.no_grad()
def validate(model, loader, config, device, world):
    model.eval()
    sums = torch.zeros(6, dtype=torch.float64, device=device)
    for batch in loader:
        ids, inputs = prepare(batch, device)
        noise, timestep = fixed_noise(ids, inputs['target'].shape[1:], config.training.validation_seed, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = model(**inputs, noise=noise, timestep=timestep, compute_alignment=False)
        mask = predicted_mask(result['region_probability'], config.selection.region_threshold,
                               config.selection.min_region_ratio, config.selection.max_region_ratio)
        truth = inputs['weak_region'] >= config.selection.weak_label_occupancy_threshold
        iou = (mask & truth).sum(1).double()/(mask | truth).sum(1).clamp_min(1)
        n = len(ids)
        sums[0] += result['flow_MSE'].double()*n
        sums[1] += result['edited_flow_MSE'].double()*n
        sums[2] += iou.sum()
        sums[3] += mask.double().mean(1).sum()
        sums[4] += result['localization'].double()*n
        sums[5] += n
    if world > 1:
        dist.all_reduce(sums)
    means = (sums[:5]/sums[5]).tolist()
    return dict(selection_score=means[1]+config.selection.iou_weight*(1-means[2]),
        validation_flow_MSE=means[0], edited_flow_MSE=means[1], weak_region_IoU=means[2],
        predicted_area=means[3], localization=means[4])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='experiments/flux2/config.yaml')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('overrides', nargs='*')
    args = parser.parse_args()
    config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    device, rank, world = distributed_device()
    if world != config.training.expected_world_size:
        raise ValueError('Unexpected GPU world size')
    output = Path(config.experiment.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = None
    if rank == 0:
        lock = (output/'.train.lock').open('w')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output/'identity.json').exists() and not args.resume:
            raise FileExistsError('Existing run requires explicit resume or a fresh output')
    if world > 1:
        dist.barrier()
    def dataset(split):
        return FlowDataset(Path(config.dataset.manifests)/(split+'.jsonl'), config.dataset.tokens,
            config.dataset.features, config.dataset.regions, config.dataset.flow_cache)
    train, validation = dataset('train'), dataset('validation')
    if train.identity['assets'] != json.loads((Path(config.model.path)/'origin.json').read_text()):
        raise ValueError('Native VAE/text caches were made with different model assets')
    sampler = DistributedSampler(train, world, rank, shuffle=True, seed=config.training.seed, drop_last=True)
    loader = DataLoader(train, batch_size=config.training.batch_size, sampler=sampler, drop_last=True,
        collate_fn=collate_flow, num_workers=config.training.num_workers, pin_memory=True,
        persistent_workers=config.training.num_workers > 0)
    val_loader = DataLoader(Subset(validation, list(range(rank,len(validation),world))),
        batch_size=config.training.batch_size, collate_fn=collate_flow,
        num_workers=config.training.num_workers, pin_memory=True)
    if not len(loader):
        raise ValueError('No full training batches')
    seed_all(config.training.seed)
    model = load_editor(config)
    warmstart = torch.load(config.model.warmstart, map_location='cpu', weights_only=False)
    if warmstart['identity']['feature_manifest'] != train.bridge.feature_complete:
        raise ValueError('Localization warmstart used another source-only feature cache')
    if warmstart['identity']['test_used_for_selection']:
        raise ValueError('Test-contaminated warmstart is prohibited')
    prefix = ('feature_norm.', 'visual_adapter.', 'conditioner.', 'alignment.')
    warm = {k:v for k,v in warmstart['model'].items() if k.startswith(prefix)}
    expected = {k for k in model.state_dict() if k.startswith(prefix)}
    if set(warm) != expected:
        raise ValueError('Incomplete latent/localization warmstart')
    model.load_state_dict(warm, strict=False)
    del warmstart, warm
    model = model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    accumulation = config.training.gradient_accumulation_steps
    updates = math.ceil(len(loader)/accumulation)
    total = updates*config.training.epochs
    optimizer = torch.optim.AdamW(parameters, lr=config.training.lr, betas=tuple(config.training.betas), weight_decay=0)
    warmup = max(1,round(total*config.training.warmup_ratio))
    def schedule(step):
        if step < warmup:
            return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,total-warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    identity = dict(variant='Qwen3.5_latent_intention_to_frozen_FLUX2_continuous_flow',
        config=OmegaConf.to_container(config,resolve=True), world_size=world,
        effective_batch=world*config.training.batch_size*accumulation,
        samples_per_epoch=len(loader)*config.training.batch_size*world, total_steps=total,
        trainable_parameters=sum(p.numel() for p in parameters),
        native_generation_backbone_updated=False, native_Qwen_backbone_updated=False,
        flow_cache_identity=train.identity, flow_cache_manifest=train.complete,
        feature_identity=train.bridge.feature_identity, feature_manifest=train.bridge.feature_complete,
        weak_region_identity=train.bridge.region_identity,
        warmstart_sha256=sha256(config.model.warmstart),
        model_origin=json.loads((Path(config.model.path)/'origin.json').read_text()),
        source_sha256={str(p):sha256(p) for folder in ('experiments/flux2','experiments/qwen35','training')
            for p in sorted(Path(folder).glob('*.py'))}, torch=torch.__version__,
        runtime={name:importlib.metadata.version(name) for name in
            ('diffusers','transformers','accelerate','huggingface-hub','tokenizers','safetensors','numpy')},
        test_used_for_selection=False, protocol='provisional_release_split_not_Table1')
    epoch_start, step, best = 0,0,float('inf')
    resumed = None
    if args.resume:
        resumed = torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
        if resumed['identity'] != identity:
            raise ValueError('Resume identity changed')
        model.load_state_dict(resumed['model'],strict=False)
        optimizer.load_state_dict(resumed['optimizer'])
        scheduler.load_state_dict(resumed['scheduler'])
        epoch_start,step,best = resumed['next_epoch'],resumed['global_step'],resumed['best']
    if rank == 0:
        (output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        OmegaConf.save(config,output/'config.yaml')
        requirements = Path('requirements-flux2.txt')
        if requirements.exists():
            (output/requirements.name).write_text(requirements.read_text())
    wrapper = DDP(model,device_ids=[device.index],broadcast_buffers=False) if world > 1 else model
    if resumed:
        restore_rng(resumed['rng_by_rank'][rank])
    else:
        seed_all(config.training.seed+rank)
    initial = validate(model,val_loader,config,device,world)
    if rank == 0:
        print(json.dumps(dict(initial_validation=initial,trainable_parameters=identity['trainable_parameters'])),flush=True)
    empty_prompt = train.empty_prompt.to(device)
    started,partial = time.monotonic(),False
    for epoch in range(epoch_start,config.training.epochs):
        sampler.set_epoch(epoch)
        wrapper.train()
        optimizer.zero_grad(set_to_none=True)
        for micro,batch in enumerate(loader):
            start = micro//accumulation*accumulation
            size = min(accumulation,len(loader)-start)
            boundary = micro-start+1 == size
            ids,inputs = prepare(batch,device)
            keep = torch.rand(len(ids),device=device) >= config.training.condition_dropout
            inputs['prompt_embeds'] = torch.where(keep[:,None,None],inputs['prompt_embeds'],empty_prompt[None])
            inputs['condition_keep'] = keep
            sync = contextlib.nullcontext() if boundary or world == 1 else wrapper.no_sync()
            with sync:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    result = wrapper(**inputs)
                    loss = result['loss']/size
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite flow/region/alignment loss')
                loss.backward()
            if boundary:
                norm = torch.nn.utils.clip_grad_norm_(parameters,config.training.gradient_clip,error_if_nonfinite=True)
                optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
                step += 1
                if rank == 0 and (step == 1 or step%10 == 0):
                    print(json.dumps(dict(epoch=epoch+1,step=step,loss=result['loss'].item(),
                        flow_MSE=result['flow_MSE'].item(),localization=result['localization'].item(),
                        alignment=result['alignment'].item(),gradient_norm=norm.item(),
                        peak_GiB=torch.cuda.max_memory_allocated()/2**30)),flush=True)
                if args.max_steps and step >= args.max_steps:
                    partial = True
                    break
        metrics = validate(model,val_loader,config,device,world)
        improved = metrics['selection_score'] < best
        best = min(best,metrics['selection_score'])
        states = [None]*world
        if world > 1:
            dist.all_gather_object(states,rng_state())
        else:
            states[0] = rng_state()
        if rank == 0:
            report = dict(epoch=epoch+1,step=step,partial_epoch=partial,elapsed_seconds=time.monotonic()-started,**metrics)
            print(json.dumps(report),flush=True)
            with (output/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(report)+'\n')
            saved = dict(model=model.trainable_state_dict(),optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),next_epoch=epoch+1,global_step=step,best=best,
                rng_by_rank=states,identity=identity)
            if partial:
                atomic_save(saved,output/'smoke.pt')
            else:
                atomic_save(saved,output/'latest.pt')
                selected = dict(model=saved['model'],identity=identity,epoch=epoch+1,validation=metrics)
                if improved:
                    atomic_save(selected,output/'best.pt')
                if (epoch+1)%config.training.save_every == 0:
                    atomic_save(selected,output/f'epoch-{epoch+1:03d}.pt')
        if world > 1:
            dist.barrier()
        if partial:
            break
    if rank == 0:
        (output/('SMOKE_COMPLETE.json' if partial else 'TRAINING_COMPLETE.json')).write_text(
            json.dumps(dict(global_step=step,best_selection_score=best,test_unsealed=False,
                table1_evaluated=False),indent=2)+'\n')
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
