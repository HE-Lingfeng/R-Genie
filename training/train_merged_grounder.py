"""Fit a spatial intention head on frozen Qwen features and merged edit supervision.

Run with torchrun --standalone --nproc-per-node=2 -m training.train_merged_grounder
after training.cache_merged_grounding and training.prepare_merged_regions.
"""
import argparse
import fcntl
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from training.cache_merged_grounding import atomic_write, partition_rows, validate
from training.merged_grounder import MergedGrounder, centroid_shift, objective
from training.paper_utils import sha256
from training.prepare_merged_regions import ACTIONS, VERSION


def feature_inputs(record):
    text = record['text'][-63:]
    words = text.new_zeros((64, 2560))
    words[-len(text)-1:-1] = text
    words[-1] = record['readout']
    pad = torch.arange(64) < 63-len(text)
    return record['visual'], words, pad, record['native_prior']


class GroundingData(Dataset):
    def __init__(self, merged, manifests, regions, cache, partition):
        self.rows, manifest_hash = partition_rows(merged, manifests, partition)
        self.directory = cache / partition
        id_path = self.directory / 'identity.json'
        identity_sha = sha256(id_path)
        identity = json.loads(id_path.read_bytes())
        if identity['manifest_sha256'] != manifest_hash or identity['partition'] != partition:
            raise ValueError('Source readouts belong to another partition')
        self.extractor = {k: identity[k] for k in
                          ('checkpoint', 'extractor_sha256', 'torch', 'transformers', 'backend', 'dtype')}
        self.hashes = {}
        shards = identity['shards']
        if shards < 1:
            raise ValueError('Invalid number of feature shards')
        receipts = []
        for i in range(shards):
            path = self.directory / f'complete-{i}.json'
            record = json.loads(path.read_bytes())
            expected = {r['id'] for r in self.rows[i::shards]}
            if (record['identity_sha256'] != identity_sha or record['count'] != len(expected)
                    or set(record['sha256']) != expected):
                raise ValueError(f'Missing or mismatched frozen readouts at {path}')
            self.hashes.update(record['sha256'])
            receipts.append(sha256(path))
        region_path = regions / (partition + '.npz')
        with np.load(region_path, allow_pickle=False) as archive:
            if (str(archive['manifest_sha256']) != manifest_hash or int(archive['schema_version']) != VERSION
                    or tuple(archive['actions']) != ACTIONS
                    or archive['ids'].tolist() != [r['id'] for r in self.rows]):
                raise ValueError(f'Spatial labels disagree with {partition} source records')
            self.labels = {k: torch.from_numpy(archive[k]) for k in (
                'source','destination','edit','has_instance','has_geometry','action','requested_dx_dy_log_scale')}
        for key in ('source', 'destination', 'edit'):
            self.labels[key] = self.labels[key].flatten(1)
        self.labels['action'] = self.labels['action'].long()
        self.identity = dict(manifest=manifest_hash, regions=sha256(region_path), features=receipts)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        path = self.directory / (row['id'] + '.pt')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.hashes[row['id']]:
            raise ValueError(f'Changed source features for {row["id"]}')
        features = torch.load(BytesIO(raw), weights_only=True, map_location='cpu')
        validate(features, row)
        return feature_inputs(features), {k: v[i] for k, v in self.labels.items()}


def batch_to(batch, device):
    inputs, labels = batch
    return ([x.to(device, non_blocking=True) for x in inputs],
            {k: v.to(device, non_blocking=True) for k, v in labels.items()})


def occupancy_iou(prediction, target):
    # Fractional target occupancy keeps sub-cell edits in the denominator.
    foreground = (prediction >= 0.5).float()
    overlap = (foreground * target).sum(1)
    area = foreground.sum(1) + target.sum(1) - overlap
    return torch.where(area > 0, overlap / area.clamp_min(1e-6), torch.ones_like(area))


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    by_action = {op: [] for op in ACTIONS}
    source, destination, transforms, placement, correct = [], [], [], [], []
    for batch in loader:
        features, labels = batch_to(batch, device)
        out = model(*features)
        native, geo = labels['has_instance'], labels['has_geometry']
        iou = occupancy_iou(out['edit'].sigmoid(), labels['edit'])
        for i, op in enumerate(ACTIONS):
            by_action[op].extend(iou[labels['action'] == i].tolist())
        source.extend(occupancy_iou(out['source'][native].sigmoid(), labels['source'][native]).tolist())
        destination.extend(occupancy_iou(out['destination'][geo], labels['destination'][geo]).tolist())
        transforms.extend((out['transform'][geo] - labels['requested_dx_dy_log_scale'][geo]).abs().tolist())
        expected_shift = centroid_shift(labels['source'][geo], labels['destination'][geo])
        placement.extend((out['applied_transform'][geo, :2] - expected_shift).abs().tolist())
        correct.extend((out['action'][native].argmax(-1) == labels['action'][native]).tolist())
    if not all(by_action.values()) or not destination:
        raise ValueError('Development partition must cover the seven actions and geometric edits')
    action_ious = {k: sum(v)/len(v) for k, v in by_action.items()}
    return dict(edit_macro_iou=sum(action_ious.values())/len(action_ious), edit_iou_by_action=action_ious,
                source_iou=float(np.mean(source)), destination_iou=float(np.mean(destination)),
                action_accuracy=float(np.mean(correct)), transform_mae=np.mean(transforms, axis=0).tolist(),
                centroid_shift_mae=np.mean(placement, axis=0).tolist())


def train(args):
    rank = int(os.getenv('RANK', 0))
    world = int(os.getenv('WORLD_SIZE', 1))
    local = int(os.getenv('LOCAL_RANK', 0))
    device = torch.device('cuda', local) if args.device == 'cuda' else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        lock = (args.output / '.training.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    data = {name: GroundingData(args.merged, args.manifests, args.regions, args.cache, name)
            for name in ('fit', 'development')}
    if data['fit'].extractor != data['development'].extractor:
        raise ValueError('Use the same readout configuration on both partitions')
    common = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == 'cuda')
    train_sampler = DistributedSampler(data['fit'], num_replicas=world, rank=rank, seed=args.seed)
    fit = DataLoader(data['fit'], sampler=train_sampler, **common)
    dev = DataLoader(data['development'], shuffle=False, **common) if rank == 0 else None
    model = MergedGrounder().to(device)
    wrapped = DistributedDataParallel(model, device_ids=[local] if device.type == 'cuda' else None) if world > 1 else model
    opt = torch.optim.AdamW(wrapped.parameters(), lr=args.lr, weight_decay=0.02)
    identity = dict(partitions={n: d.identity for n, d in data.items()}, extractor=data['fit'].extractor,
                    actions=list(ACTIONS), model_code=sha256(Path(__file__).with_name('merged_grounder.py')),
                    trainer_code=sha256(__file__), seed=args.seed, batch_size=args.batch_size, world=world, lr=args.lr)
    path = args.output / 'checkpoint.pt'
    best, best_state, history = -1., None, []
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['identity'] != identity:
            raise ValueError('Existing checkpoint has a different training configuration')
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        best, best_state, history = state['best_edit_macro_iou'], state['best_model'], state['history']
    for epoch in range(len(history) + 1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        wrapped.train()
        for batch in fit:
            features, labels = batch_to(batch, device)
            opt.zero_grad(set_to_none=True)
            loss = objective(wrapped(*features), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1, error_if_nonfinite=True)
            opt.step()
        if rank == 0:
            metrics = evaluate(model, dev, device)
            history.append(dict(epoch=epoch, **metrics))
            if metrics['edit_macro_iou'] > best:
                best = metrics['edit_macro_iou']
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            state = dict(identity=identity, history=history, model=model.state_dict(), optimizer=opt.state_dict(),
                         best_edit_macro_iou=best, best_model=best_state)
            atomic_write(path, lambda target: torch.save(state, target))
            print(json.dumps(history[-1]), flush=True)
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()
    if rank == 0:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('merged', 'manifests', 'regions', 'cache', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=16, help='Per-device batch size')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=3.5e-4)
    parser.add_argument('--seed', type=int, default=42)
    options = parser.parse_args()
    if min(options.batch_size, options.epochs) < 1 or options.lr <= 0 or options.workers < 0:
        parser.error('Invalid training arguments')
    train(options)
