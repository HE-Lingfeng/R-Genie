"""Train a source-only predictor of the intended edited region's visual prompt.

First prepare train and validation records using training.cache_visual_intentions.
The predictor, not the frozen image encoder or diffusion backbone, is optimized.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from training.train_intention_transport import digest


class VisualIntention(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, 192), nn.SiLU())
        self.language = nn.Sequential(
            nn.LayerNorm(2560), nn.Linear(2560, 192), nn.SiLU()
        )
        self.fuse = nn.Sequential(nn.Linear(576, 192), nn.LayerNorm(192), nn.SiLU())
        self.change = nn.Linear(192, 1024)
        self.gate = nn.Linear(192, 1)
        nn.init.zeros_(self.change.weight)
        nn.init.zeros_(self.change.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.5)

    def forward(self, source, context, intention):
        source, context = source.float(), context.float()
        start, surround = F.normalize(source), F.normalize(context)
        visual = self.visual(torch.cat([start, surround - start], dim=-1))
        text = self.language(intention.float())
        hidden = self.fuse(torch.cat([visual, text, visual * text], dim=-1))
        gate = self.gate(hidden).sigmoid()
        residual = 0.7 * self.change(hidden).tanh() / (source.shape[-1] ** 0.5)
        direction = F.normalize(start + gate * (surround - start) + residual)
        magnitude = torch.lerp(
            source.norm(dim=-1, keepdim=True), context.norm(dim=-1, keepdim=True), gate
        )
        return direction * magnitude


def error(prediction, target, source):
    pred, tgt, src = [F.normalize(x.float()) for x in (prediction, target, source)]
    shift = tgt - src
    weight = (shift.norm(dim=-1) / 0.25).clamp(max=1)
    cosine = (pred * tgt).sum(-1)
    delta = F.cosine_similarity(pred - src, shift)
    return 1 - cosine + 0.30 * weight * (1 - delta), delta


def objective(model, source, context, intention, target):
    prediction = model(source, context, intention)
    base, alignment = error(prediction, target, source)
    if len(source) < 2:
        return base.mean()
    src, tgt, out = [F.normalize(x.float()) for x in (source, target, prediction)]
    movement = F.normalize(out - src)
    desired = F.normalize(tgt - src)
    strength = ((tgt - src).norm(dim=-1) / 0.25).clamp(max=1)
    contrast = F.cross_entropy(
        movement @ desired.T / 0.12,
        torch.arange(len(source), device=source.device),
        reduction="none",
    )
    swapped = model(source, context, intention.roll(1, 0))
    _, wrong = error(swapped, target, source)
    ranking = (0.03 - alignment + wrong).relu()
    return (base + 0.06 * strength * contrast + 0.10 * strength * ranking).mean()


def load_split(cache, manifests, split, device):
    file = manifests / (split + ".jsonl")
    manifest_hash = digest(file)
    rows = [json.loads(s) for s in file.read_text().splitlines() if s.strip()]
    groups = [[], [], [], []]
    photos = set()
    for row in rows:
        d = torch.load(
            cache / split / (row["id"] + ".pt"), weights_only=True, map_location="cpu"
        )
        ident = d["identity"]
        if ident["manifest"] != manifest_hash or any(
            ident[k] != row[k]
            for k in ("id", "source_sha256", "target_sha256", "instruction")
        ):
            raise ValueError(f"Stale cached example {row['id']}")
        if photos.intersection((row["source_sha256"], row["target_sha256"])):
            raise ValueError("Duplicate photograph in split")
        photos.update((row["source_sha256"], row["target_sha256"]))
        for out, key, dim in zip(
            groups,
            ("source", "context", "intention", "target"),
            (1024, 1024, 2560, 1024),
        ):
            v = d[key].float()
            if v.shape != (dim,) or not v.isfinite().all():
                raise ValueError(f"Invalid {key} for {row['id']}")
            out.append(v)
    return [torch.stack(items).to(device) for items in groups], photos, manifest_hash


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    s, c, q, t = data
    pred = model(s, c, q)
    scores, correct = error(pred, t, s)
    _, wrong = error(model(s, c, q.roll(1, 0)), t, s)
    return {
        "objective": float(scores.mean()),
        "target_cosine": float(F.cosine_similarity(pred, t).mean()),
        "direction_cosine": float(correct.mean()),
        "swapped_direction_cosine": float(wrong.mean()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "manifests", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    a = p.parse_args()
    if min(a.epochs, a.batch_size) < 1:
        p.error("Epochs and batch size must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    tr, tr_photos, tr_sha = load_split(a.cache, a.manifests, "train", device)
    val, val_photos, val_sha = load_split(a.cache, a.manifests, "validation", device)
    if tr_photos & val_photos:
        raise ValueError("Train and validation photographs overlap")
    m = VisualIntention().to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.015)
    a.out.mkdir(parents=True, exist_ok=True)
    idx_gen = torch.Generator().manual_seed(a.seed)
    initial = evaluate(m, val)
    best = initial["objective"]
    best_epoch = 0
    hist = [{"epoch": 0, "validation": initial}]

    def save():
        f = a.out / "selected.partial"
        torch.save(
            {
                "state": {k: v.detach().cpu() for k, v in m.state_dict().items()},
                "epoch": best_epoch,
                "train": tr_sha,
                "validation": val_sha,
                "seed": a.seed,
            },
            f,
        )
        f.replace(a.out / "selected.pt")

    save()
    updates = 0
    for epoch in range(1, a.epochs + 1):
        m.train()
        total = 0.0
        for ids in torch.randperm(len(tr[0]), generator=idx_gen).split(a.batch_size):
            batch = [v[ids.to(device)] for v in tr]
            loss = objective(m, *batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            updates += 1
            total += float(loss.detach()) * len(ids)
        ev = evaluate(m, val)
        if ev["objective"] < best:
            best = ev["objective"]
            best_epoch = epoch
            save()
        hist.append({"epoch": epoch, "train": total / len(tr[0]), "validation": ev})
        info = {
            "seed": a.seed,
            "trainable": sum(p.numel() for p in m.parameters()),
            "updates": updates,
            "initial": initial["objective"],
            "best": best,
            "selected_epoch": best_epoch,
            "history": hist,
            "train_manifest": tr_sha,
            "validation_manifest": val_sha,
            "checkpoint": digest(a.out / "selected.pt"),
        }
        tmp = a.out / "training.partial"
        tmp.write_text(json.dumps(info, indent=2))
        tmp.replace(a.out / "training.json")
        print(epoch, ev, flush=True)


if __name__ == "__main__":
    main()
