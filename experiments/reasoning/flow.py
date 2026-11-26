"""Grounded source-reference conditioning and an edit-direction rectified-flow objective."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from experiments.flux2_lora.model import load_editor
from experiments.reasoning.data import AUX_CACHE, SPLITS, STUDY, manifest
from training.paper_utils import sha256

CACHE = STUDY / "flow-inputs"


class GroundedFlow(nn.Module):
    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        self.marker = nn.Parameter(torch.zeros(128))

    def reference(self, src, mask):
        strength = mask.float().clamp(0, 1).unsqueeze(-1)
        return ((1 - 0.65 * strength) * src.float() + strength * self.marker).to(
            src.dtype
        )

    def forward(self, src, tgt, prompt, mask, region, noise, time):
        time = time.view(-1, 1, 1)
        mix = (1 - time) * tgt.float() + time * noise.float()
        velocity = self.editor.vector_field(
            mix.to(src.dtype),
            self.reference(src, mask),
            prompt,
            time.flatten().to(src.dtype),
            None,
        )
        error = (velocity.float() - (noise.float() - tgt.float())).square().mean(-1)
        fg = region.float().clamp(0, 1)
        bg = 1 - fg

        def weighted(value, w):
            return (value * w).sum(1) / w.sum(1).clamp_min(1)

        clean = mix - time * velocity.float()
        goal = tgt.float() - src.float()
        step = clean - src.float()
        direction = 1 - F.cosine_similarity(step, goal, dim=-1, eps=1e-6)
        valid = fg * (goal.square().mean(-1) > 0.01)
        return (
            0.7 * weighted(error, fg)
            + 0.3 * weighted(error, bg)
            + 0.08 * weighted(direction, valid)
        )


class Examples(Dataset):
    def __init__(self, split):
        if split not in ("train", "validation"):
            raise ValueError(split)
        self.files = []
        ids = set()
        for r in manifest(split):
            self.files.append((CACHE / split / (r["id"] + ".pt"), r["id"], False))
            ids.add(r["id"])
        if split == "train":
            opt = SPLITS / "aux-explicit-train.jsonl"
            assert (
                sha256(opt)
                == json.loads((SPLITS / "aux-protocol.json").read_text())[
                    "input_manifest"
                ]
            )
            for line in opt.read_text().splitlines():
                row = json.loads(line)
                key = row["id"]
                if key in ids:
                    continue
                self.files.append((AUX_CACHE / "train" / (key + ".pt"), key, True))
                ids.add(key)
            assert len(self.files) == 749
        else:
            assert len(self.files) == 100

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path, key, extra = self.files[i]
        record = torch.load(path, weights_only=True)
        assert record["id"] == key
        region = record["region"]
        if extra:
            grid = region.reshape(1, 1, 32, 32)
            mask = (
                F.avg_pool2d(grid, 5, 1, 2)
                .roll(tuple(torch.randint(-2, 3, (2,)).tolist()), (-2, -1))
                .flatten()
            )
        else:
            mask = record["mask"]
        return key, record["source"], record["target"], record["prompt"], mask, region


def init_model(device, parent):
    saved = torch.load(parent, map_location="cpu", weights_only=False)
    identity = saved["identity"]
    assert all(sha256(k) == v for k, v in identity["source_sha256"].items())
    editor = load_editor(OmegaConf.create(identity["config"]))
    assert editor.generation_lora_targets == identity["generation_lora_targets"]
    assert {n for n, p in editor.named_parameters() if p.requires_grad} == set(
        saved["model"]
    )
    editor.load_state_dict(saved["model"], strict=False)
    for name, param in editor.named_parameters():
        param.requires_grad_(".lora_A." in name or ".lora_B." in name)
    return GroundedFlow(editor).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent",
        type=Path,
        required=True,
        help="Trained FLUX generation-LoRA checkpoint",
    )
    args = parser.parse_args()
    parent_hash = sha256(args.parent)
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == 2
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    dist.init_process_group("nccl")
    torch.manual_seed(42)
    module = init_model(device, args.parent)
    model = DistributedDataParallel(module, device_ids=[rank], broadcast_buffers=False)
    lora = [p for p in module.editor.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": lora, "lr": 1e-5}, {"params": [module.marker], "lr": 1e-3}],
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    train, val = Examples("train"), Examples("validation")
    sampler = DistributedSampler(train, shuffle=True, seed=42)
    train_loader = DataLoader(
        train, batch_size=3, sampler=sampler, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val,
        batch_size=3,
        sampler=range(rank, len(val), world),
        num_workers=1,
        pin_memory=True,
    )
    rng = torch.Generator(device=device).manual_seed(1806 + rank)

    def losses(batch, dev):
        ids, *parts = batch
        source, target, prompt, mask, region = [
            p.to(device, non_blocking=True) for p in parts
        ]
        if dev:
            gens = [
                torch.Generator(device=device).manual_seed(45371 + int(k)) for k in ids
            ]
            noise = torch.stack(
                [
                    torch.randn(
                        target.shape[1:], device=device, dtype=target.dtype, generator=g
                    )
                    for g in gens
                ]
            )
            t = torch.stack(
                [torch.randn((), device=device, generator=g).sigmoid() for g in gens]
            )
        else:
            noise = torch.randn(
                target.shape, device=device, dtype=target.dtype, generator=rng
            )
            t = torch.randn(len(ids), device=device, generator=rng).sigmoid()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(source, target, prompt, mask, region, noise, t)

    out = STUDY / "flow-training"
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    hist = []
    for epoch in range(4):
        if epoch:
            sampler.set_epoch(epoch)
            model.train()
            for batch in train_loader:
                opt.zero_grad(set_to_none=True)
                losses(batch, False).mean().backward()
                nn.utils.clip_grad_norm_([*lora, module.marker], 1.0)
                opt.step()
        model.eval()
        total = torch.zeros(2, dtype=torch.float64, device=device)
        with torch.no_grad():
            for batch in val_loader:
                values = losses(batch, True)
                total[0] += values.double().sum()
                total[1] += len(values)
        dist.all_reduce(total)
        assert int(total[1]) == 100
        score = float(total[0] / total[1])
        hist.append(score)
        if rank == 0:
            print("epoch", epoch, "validation", score, flush=True)
            if score < best:
                state = {
                    n: p.detach().cpu()
                    for n, p in module.named_parameters()
                    if p.requires_grad
                }
                tmp = out / "selected.tmp"
                torch.save(
                    dict(
                        weights=state,
                        epoch=epoch,
                        validation=score,
                        parent=parent_hash,
                        code=sha256(__file__),
                        grounder_lock=sha256(STUDY / "grounder-lock.json"),
                        manifests={
                            k: sha256(SPLITS / (k + ".jsonl"))
                            for k in ("train", "validation")
                        },
                    ),
                    tmp,
                )
                tmp.replace(out / "selected.pt")
            if epoch == 3:
                (out / "COMPLETE.json").write_text(
                    json.dumps(
                        dict(
                            validation=hist,
                            selected_epoch=hist.index(min(hist)),
                            count_train=len(train),
                            count_validation=len(val),
                            optimizer_steps=3 * len(train_loader),
                            ranks=world,
                            peak_allocated_GiB=torch.cuda.max_memory_allocated()
                            / 2**30,
                            trainable=sum(
                                p.numel()
                                for p in module.parameters()
                                if p.requires_grad
                            ),
                            checkpoint=sha256(out / "selected.pt"),
                        ),
                        indent=2,
                    )
                )
        best = min(best, score)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
