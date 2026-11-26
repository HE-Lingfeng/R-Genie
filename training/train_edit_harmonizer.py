"""Train an intention-conditioned local aligner and matte on precomputed native edits.

Input manifests are JSONL with id, source, target, instruction and source/target
SHA-256 digests. Store one source-only raw generator output as <raw>/<split>/<id>.png,
with id, source_sha256 and sha256 in an adjacent JSON receipt. Use source-only
prior/intention records from training.cache_transport_intentions. Development
uses the unaugmented native output for each image. The optimizer sees train only.

For the four-step experiment, each native training image uses Qwen-Image-Edit-2511
with the Q4_K_M transformer and the four-step BF16 Lightning weights from
lightx2v/Qwen-Image-Edit-2511-Lightning at 107d238d68b8387e32843d0b4a2e24ceb0d3ad86.
Use the original request, the source resized to 512 square, seed 42 + integer ID,
true CFG 1, and the Lightning Euler scheduler with constant log(3) shift.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F

from training.train_intention_transport import digest


def picture(path):
    with Image.open(path) as image:
        return (
            ImageOps.exif_transpose(image)
            .convert("RGB")
            .resize((512, 512), Image.Resampling.BICUBIC)
        )


def tensor(image, size):
    reduced = image.resize((size, size), Image.Resampling.BOX)
    return (
        torch.from_numpy(np.array(reduced, copy=True)).permute(2, 0, 1).float().div(255)
    )


def block(inp, out):
    return nn.Sequential(
        nn.Conv2d(inp, out, 3, padding=1),
        nn.GroupNorm(8, out),
        nn.SiLU(),
        nn.Conv2d(out, out, 3, padding=1),
        nn.GroupNorm(8, out),
        nn.SiLU(),
    )


def scale(x, size):
    return F.interpolate(x, size, mode="bilinear", align_corners=False)


def base_grid(batch, height, width, device):
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * 2 / width - 1
    y = (
        torch.arange(height, device=device, dtype=torch.float32) + 0.5
    ) * 2 / height - 1
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1)[None].expand(batch, -1, -1, -1)


def warp(raw, displacement):
    b, _, h, w = raw.shape
    grid = base_grid(b, h, w, raw.device) + displacement.permute(0, 2, 3, 1)
    return F.grid_sample(
        raw, grid, mode="bilinear", padding_mode="border", align_corners=False
    )


class EditHarmonizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry = block(10, 24)
        self.half_stage = block(24, 40)
        self.quarter = block(40 + 25, 64)
        self.language = nn.Sequential(nn.LayerNorm(2560), nn.Linear(2560, 64))
        self.decode = block(64 + 40, 40)
        self.detail = block(40 + 24, 24)
        self.fields = nn.Conv2d(24, 6, 1)
        nn.init.zeros_(self.fields.weight)
        nn.init.zeros_(self.fields.bias)

    def forward(self, source, raw, prior, intention):
        prior = scale(prior.reshape(-1, 1, 32, 32), source.shape[-2:])
        first = self.entry(torch.cat((source, raw, raw - source, prior), dim=1))
        half = self.half_stage(F.avg_pool2d(first, 2))
        coarse_source = F.avg_pool2d(source, 4)
        coarse_raw = F.avg_pool2d(raw, 4)
        h, w = coarse_source.shape[-2:]
        padded = F.pad(coarse_raw, (2, 2, 2, 2), mode="replicate")
        costs = torch.cat(
            [
                (coarse_source - padded[:, :, y : y + h, x : x + w])
                .abs()
                .mean(1, keepdim=True)
                for y in range(5)
                for x in range(5)
            ],
            dim=1,
        )
        coarse = self.quarter(torch.cat((F.avg_pool2d(half, 2), costs), dim=1))
        coarse = coarse + self.language(intention.float())[:, :, None, None]
        mid = self.decode(torch.cat((scale(coarse, half.shape[-2:]), half), dim=1))
        prediction = self.fields(
            self.detail(torch.cat((scale(mid, source.shape[-2:]), first), dim=1))
        )
        prior = ((prior - 0.02) / 0.40).clamp(0.015, 0.985)
        alpha = torch.sigmoid(torch.logit(prior) + prediction[:, :1])
        # Bounded normalized grid coordinates give a maximum displacement of
        # 32 pixels at 512px and scale consistently during 128px training.
        return alpha, 0.125 * prediction[:, 1:3].tanh(), 0.07 * prediction[:, 3:].tanh()


def composite(source, raw, fields, alpha_only=False):
    size = source.shape[-2:]
    alpha, flow, color = (scale(x, size) for x in fields)
    aligned = raw if alpha_only else (warp(raw, flow) + color).clamp(0, 1)
    return source + alpha * (aligned - source), aligned


def weighted(x, mask):
    return (x * mask).sum((1, 2, 3)) / mask.sum((1, 2, 3)).clamp_min(1)


def robust(x):
    return (x.square() + 1e-6).sqrt().mean(1, keepdim=True)


def objective(output, aligned, fields, target, source, region):
    alpha, flow, _ = fields
    error = robust(output - target)
    foreground = weighted(error, region)
    background = weighted(error, 1 - region)
    logalpha = alpha.clamp(1e-6, 1 - 1e-6)
    mask_loss = -0.5 * (
        weighted(logalpha.log(), region) + weighted(torch.log1p(-logalpha), 1 - region)
    )
    edge = 0.0
    smooth = 0.0
    for axis in (2, 3):
        roi = torch.narrow(region, axis, 0, region.shape[axis] - 1)
        loss = robust(torch.diff(output, dim=axis) - torch.diff(target, dim=axis))
        edge = edge + 0.5 * (0.7 * weighted(loss, roi) + 0.3 * weighted(loss, 1 - roi))
        smooth = smooth + torch.diff(flow, dim=axis).abs().mean()
    registration = weighted(robust(aligned - source), 1 - region)
    return (
        0.60 * foreground
        + 0.20 * background
        + 0.10 * edge
        + 0.08 * mask_loss
        + 0.10 * registration
    ).mean() + 0.015 * smooth


class Samples:
    def __init__(self, split, args):
        manifest = args.manifests / (split + ".jsonl")
        self.digest = digest(manifest)
        self.rows = []
        self.sources = set()
        columns = []
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ids = r["id"]
            src = args.data / r["source"]
            tgt = args.data / r["target"]
            raw = args.raw / split / (ids + ".png")
            info = json.loads(raw.with_suffix(".json").read_text())
            if (info["id"], info["source_sha256"], info["sha256"]) != (
                ids,
                digest(src),
                digest(raw),
            ):
                raise ValueError(f"Native edit does not match the manifest: {ids}")
            if (info["source_sha256"], digest(tgt)) != (
                r["source_sha256"],
                r["target_sha256"],
            ):
                raise ValueError(ids)
            intent = torch.load(
                args.intents / split / (ids + ".pt"),
                map_location="cpu",
                weights_only=True,
            )
            if (intent["source_sha256"], intent["instruction"]) != (
                r["source_sha256"],
                r["instruction"],
            ):
                raise ValueError(f"Intention record does not match: {ids}")
            source, target = picture(src), picture(tgt)
            changed = (
                np.abs(
                    np.asarray(source, dtype=np.int16)
                    - np.asarray(target, dtype=np.int16)
                ).max(-1)
                > 12.75
            )
            mask = torch.from_numpy(
                changed.reshape(128, 4, 128, 4).mean((1, 3)).astype(np.float32)
            )[None]
            columns.append(
                (
                    tensor(source, 128),
                    tensor(picture(raw), 128),
                    tensor(target, 128),
                    mask,
                    intent["prior"],
                    intent["intention"],
                )
            )
            self.sources.add(r["source_sha256"])
            self.rows.append(ids)
        self.tensors = [torch.stack(column) for column in zip(*columns)]

    def batch(self, idx, device):
        return [x[idx].to(device) for x in self.tensors]


def perturb(target):
    count, _, height, width = target.shape
    opts = {"device": target.device}
    displacement = 0.09 * (2 * torch.rand(count, 2, 1, 1, **opts) - 1)
    scale_error = 0.04 * (2 * torch.rand(count, 1, 1, 1, **opts) - 1)
    grid = base_grid(count, height, width, target.device).permute(0, 3, 1, 2)
    raw = warp(target, displacement + scale_error * grid)
    gain = 0.90 + 0.20 * torch.rand(count, 3, 1, 1, **opts)
    bias = 0.06 * (2 * torch.rand(count, 3, 1, 1, **opts) - 1)
    return (raw * gain + bias).clamp(0, 1)


def forward_batch(model, data, indices, dev, synthetic=False):
    source, raw, target, region, prior, intent = data.batch(indices, dev)
    if synthetic:
        raw = perturb(target)
    fields = model(source, raw, prior, intent)
    out, aligned = composite(source, raw, fields)
    return objective(out, aligned, fields, target, source, region)


@torch.no_grad()
def validate(model, data, dev, bs):
    model.eval()
    total = 0.0
    for ids in torch.arange(len(data.rows)).split(bs):
        total += float(forward_batch(model, data, ids, dev)) * len(ids)
    return total / len(data.rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("data", "raw", "manifests", "intents", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=22)
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()
    if min(a.epochs, a.batch_size) < 1:
        p.error("epochs and batch size must be positive")
    torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    tr = Samples("train", a)
    va = Samples("validation", a)
    if tr.sources & va.sources:
        raise ValueError("Training and development sources overlap")
    m = EditHarmonizer().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    initial = best = validate(m, va, dev, a.batch_size)
    rng = torch.Generator().manual_seed(a.seed)
    chosen = 0
    history = [{"epoch": 0, "validation": best}]
    a.out.mkdir(parents=True, exist_ok=True)

    def save():
        temp = a.out / "selected.tmp"
        torch.save(
            {
                "state": {k: v.detach().cpu() for k, v in m.state_dict().items()},
                "epoch": chosen,
                "train": tr.digest,
                "validation": va.digest,
                "seed": a.seed,
            },
            temp,
        )
        temp.replace(a.out / "selected.pt")

    save()
    step = 0
    for epoch in range(1, a.epochs + 1):
        m.train()
        total = 0.0
        for ids in torch.randperm(len(tr.rows), generator=rng).split(a.batch_size):
            step += 1
            loss = forward_batch(m, tr, ids, dev, synthetic=step % 4 == 0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            total += float(loss.detach()) * len(ids)
        score = validate(m, va, dev, a.batch_size)
        if score < best:
            best = score
            chosen = epoch
            save()
        history.append(
            {"epoch": epoch, "train": total / len(tr.rows), "validation": score}
        )
        receipt = {
            "history": history,
            "initial": initial,
            "best": best,
            "selected_epoch": chosen,
            "checkpoint": digest(a.out / "selected.pt"),
            "trainable": sum(v.numel() for v in m.parameters()),
            "updates": step,
        }
        t = a.out / "training.tmp"
        t.write_text(json.dumps(receipt, indent=2))
        t.replace(a.out / "training.json")
        print(history[-1], flush=True)


if __name__ == "__main__":
    main()
