"""Train a source-conditioned correction to rectified-flow editing velocities on cached image pairs."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class IntentionTransport(nn.Module):
    def __init__(self, channels=64, intention_width=2560, width=96):
        super().__init__()
        self.input = nn.Conv2d(3 * channels + 2, width, 3, padding=1)
        self.intention = nn.Sequential(
            nn.LayerNorm(intention_width), nn.Linear(intention_width, width)
        )
        self.time = nn.Sequential(
            nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, width),
                    nn.SiLU(),
                    nn.Conv2d(width, width, 3, padding=1),
                    nn.GroupNorm(8, width),
                    nn.SiLU(),
                    nn.Conv2d(width, width, 3, padding=1),
                )
                for _ in range(2)
            ]
        )
        self.output = nn.Conv2d(width, channels + 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.transport = nn.Parameter(torch.tensor(0.0))

    def forward(self, z, velocity, source, prior, intention, sigma):
        batch, tokens, _channels = z.shape
        stride = int(tokens**0.5)
        if stride * stride != tokens:
            raise ValueError("Expected square packed image-token grids")

        def grid(t):
            return t.transpose(1, 2).reshape(batch, -1, stride, stride)

        s = sigma.float().view(batch, 1, 1)
        displacement = z.float() - s * velocity.float() - source.float()
        p = prior.float().reshape(batch, 1, stride, stride).clamp(0.005, 0.995)
        inp = torch.cat(
            (
                grid(displacement),
                grid(velocity.float()),
                grid(source.float()),
                p,
                s.view(batch, 1, 1, 1).expand_as(p),
            ),
            dim=1,
        )
        basis = torch.cat(
            (
                s.flatten(1),
                torch.sin(torch.pi * s.flatten(1)),
                torch.cos(torch.pi * s.flatten(1)),
            ),
            dim=1,
        )
        h = (
            self.input(inp)
            + (self.intention(intention.float()) + self.time(basis))[:, :, None, None]
        )
        for block in self.blocks:
            h = h + block(h)
        out = self.output(h)
        gate = torch.sigmoid(torch.logit(p) + out[:, :1])
        residual = 2 * out[:, 1:].tanh()
        bridge = grid(displacement / s.clamp_min(0.10)).clamp(-6, 6)
        correction = (
            gate * residual + (1 - gate) * self.transport.tanh().clamp(0, 1) * bridge
        )
        return velocity.float() + correction.flatten(2).transpose(1, 2), gate.flatten(1)


def balanced(pred, truth, region):
    error = (pred.float() - truth.float()).square().mean(-1)
    r = region.float().reshape_as(error)
    fg = (error * r).sum(1) / r.sum(1).clamp_min(1)
    bg = (error * (1 - r)).sum(1) / (1 - r).sum(1).clamp_min(1)
    return 0.65 * fg + 0.35 * bg


class FlowRecords:
    def __init__(self, folder):
        seal = json.loads((folder / "COMPLETE.json").read_text())
        self.rows = []
        self.index = []
        for item in seal["records"]:
            path = folder / (item["id"] + ".pt")
            if digest(path) != item["sha256"]:
                raise ValueError(f"Changed feature cache {path}")
            rec = torch.load(path, map_location="cpu", weights_only=True)
            if rec["id"] != item["id"] or rec["source_sha256"] != item["source_sha256"]:
                raise ValueError(str(path))
            self.index.extend((len(self.rows), i) for i in range(len(rec["sigmas"])))
            self.rows.append(rec)
        self.seal = seal

    def __len__(self):
        return len(self.index)

    def batch(self, indices, device):
        items = []
        for idx in indices:
            i, j = self.index[idx]
            r = self.rows[i]
            s = r["sigmas"][j].float()
            noise = r["noise"][j].float()
            target = r["target"].float()
            z = (1 - s) * target + s * noise
            items.append(
                (
                    z,
                    r["velocity"][j],
                    r["source"],
                    r["prior"],
                    r["intention"],
                    s,
                    noise - target,
                    r["region"],
                )
            )
        return [torch.stack(x).to(device) for x in zip(*items)]


@torch.no_grad()
def validate(model, data, device, bs):
    model.eval()
    loss = []
    base = []
    for ids in torch.arange(len(data)).split(bs):
        *args, truth, region = data.batch(ids.tolist(), device)
        output, _ = model(*args)
        loss.extend(balanced(output, truth, region).cpu().tolist())
        base.extend(balanced(args[1], truth, region).cpu().tolist())
    return sum(loss) / len(loss), sum(base) / len(base)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    a = p.parse_args()
    if a.batch_size < 1 or a.epochs < 1:
        raise ValueError("Expected a positive batch size and epoch count")
    torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    train = FlowRecords(a.cache / "train")
    val = FlowRecords(a.cache / "validation")
    validation_images = {r["source_sha256"] for r in val.rows}
    assert not any(r["source_sha256"] in validation_images for r in train.rows)
    model = IntentionTransport().to(dev)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    rng = torch.Generator().manual_seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    best, base = validate(model, val, dev, a.batch_size)
    if abs(best - base) > 1e-6:
        raise RuntimeError(
            "Zero-initialized correction must reproduce the native velocity"
        )
    history = [{"epoch": 0, "validation": best}]
    chosen = 0

    def save():
        tmp = a.out / "selected.tmp"
        torch.save(
            {
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": chosen,
                "validation": best,
                "train": digest(a.cache / "train/COMPLETE.json"),
                "development": digest(a.cache / "validation/COMPLETE.json"),
                "trainable": sum(v.numel() for v in model.parameters()),
                "seed": a.seed,
            },
            tmp,
        )
        tmp.replace(a.out / "selected.pt")

    save()
    for epoch in range(1, a.epochs + 1):
        model.train()
        running = []
        for ids in torch.randperm(len(train), generator=rng).split(a.batch_size):
            *args, truth, r = train.batch(ids.tolist(), dev)
            output, gate = model(*args)
            objective = balanced(
                output, truth, r
            ).mean() + 0.05 * F.binary_cross_entropy(
                gate.clamp(1e-6, 1 - 1e-6), r.float()
            )
            optim.zero_grad(set_to_none=True)
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optim.step()
            # Keep the analytic source pull feasible so a negative first update
            # cannot permanently saturate the one-sided gate at initialization.
            with torch.no_grad():
                model.transport.clamp_(min=0)
            running.append(float(objective.detach()))
        score, _ = validate(model, val, dev, a.batch_size)
        if score < best:
            best = score
            chosen = epoch
            save()
        history.append(
            {"epoch": epoch, "train": sum(running) / len(running), "validation": score}
        )
        (a.out / "training.json").write_text(
            json.dumps(
                {
                    "history": history,
                    "selected_epoch": chosen,
                    "native": base,
                    "best": best,
                    "checkpoint": digest(a.out / "selected.pt"),
                    "trainable": sum(v.numel() for v in model.parameters()),
                },
                indent=2,
            )
        )
        print(history[-1], flush=True)


if __name__ == "__main__":
    main()
