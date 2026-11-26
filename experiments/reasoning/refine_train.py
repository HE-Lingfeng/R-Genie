"""Supervise fine object boundaries while retaining the source-only reasoning heatmap."""

import argparse
import json

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from experiments.reasoning.data import DATA, SPLITS, STUDY, manifest
from experiments.reasoning.grounding import source_inputs
from experiments.reasoning.select_grounder import load_ensemble, predict
from experiments.resplit.spatial import rgb
from training.paper_utils import sha256

RESOLUTION = (128, 128)
OUT = STUDY / "refinement"


def block(inputs, width):
    return nn.Sequential(
        nn.Conv2d(inputs, width, 3, padding=1),
        nn.GroupNorm(4, width),
        nn.SiLU(),
        nn.Conv2d(width, width, 3, padding=1),
        nn.GroupNorm(4, width),
        nn.SiLU(),
    )


class BoundaryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fine = block(5, 12)
        self.middle = block(12, 20)
        self.context = block(20, 32)
        self.merge = block(32 + 20, 20)
        self.decoder = block(20 + 12, 12)
        self.offset = nn.Conv2d(12, 1, 1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x):
        fine = self.fine(x)
        mid = self.middle(F.avg_pool2d(fine, 2))
        small = self.context(F.avg_pool2d(mid, 2))

        def merge(deep, skip):
            up = F.interpolate(
                deep, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            return torch.cat((up, skip), 1)

        middle = self.merge(merge(small, mid))
        encoded = self.decoder(merge(middle, fine))
        return torch.logit(x[:, 3:4].clamp(0.005, 0.995)) + self.offset(encoded)


@torch.no_grad()
def source(row, split, heads, mode, device):
    assert sha256(DATA / row["source"]) == row["source_sha256"]
    v = [t[None].to(device) for t in source_inputs(row, split)]
    p = predict(heads, mode, v).view(1, 1, 16, 16)
    coarse = F.interpolate(p, RESOLUTION, mode="bilinear", align_corners=False)[0].cpu()
    im = rgb(DATA / row["source"]).resize(RESOLUTION, Image.Resampling.BOX)
    image = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255
    with Image.open(STUDY / "segments" / split / (row["id"] + ".png")) as sm:
        seg = (
            torch.from_numpy(
                np.array(sm.convert("L").resize(RESOLUTION, Image.Resampling.BOX))
            ).float()[None]
            / 255
        )
    return torch.cat((image, coarse, seg), 0)


def training_pairs(split, heads, mode):
    if split not in ("train", "validation"):
        raise ValueError(split)
    xs, ys = [], []
    for row in manifest(split):
        xs.append(source(row, split, heads, mode, "cpu"))
        assert sha256(DATA / row["target"]) == row["target_sha256"]
        a, b = [
            np.asarray(rgb(DATA / row[k]), dtype=np.int16) for k in ("source", "target")
        ]
        occ = (np.abs(a - b).max(-1) > 12.75).reshape(128, 4, 128, 4).mean(axis=(1, 3))
        ys.append(torch.from_numpy(occ).float()[None])
    return torch.stack(xs), torch.stack(ys)


def dice(prob, target):
    return (2 * (prob * target).sum((1, 2, 3)) + 1) / (
        (prob + target).sum((1, 2, 3)) + 1
    )


def objective(logits, y):
    p = logits.sigmoid()
    border = (
        F.l1_loss(p.diff(dim=2), y.diff(dim=2))
        + F.l1_loss(p.diff(dim=3), y.diff(dim=3))
    ) / 2
    return (
        F.binary_cross_entropy_with_logits(logits, y)
        + (1 - dice(p, y)).mean()
        + 0.4 * border
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True, choices=(0, 1))
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device("cuda", args.rank)
    torch.cuda.set_device(device)
    torch.manual_seed(912 + args.rank)
    mode = json.loads((STUDY / "grounder-lock.json").read_text())["selected"]
    heads = load_ensemble(mode, "cpu")
    x, y = training_pairs("train", heads, mode)
    vx, vy = training_pairs("validation", heads, mode)
    del heads
    model = BoundaryHead().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.02)
    out = OUT / f"seed{args.rank}.pt"
    OUT.mkdir(parents=True, exist_ok=True)
    best = -1
    history = []
    for epoch in range(19):
        if epoch:
            model.train()
            for ids in torch.randperm(len(x)).split(8):
                bx, by = x[ids].to(device), y[ids].to(device)
                if torch.rand(()) < 0.5:
                    bx, by = bx.flip((-1,)), by.flip((-1,))
                loss = objective(model(bx), by)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
        model.eval()
        with torch.no_grad():
            prob = torch.cat(
                [model(batch.to(device)).sigmoid().cpu() for batch in vx.split(8)]
            )
            score = dice(prob, vy).mean().item()
        history.append(score)
        if score > best:
            best = score
            tmp = out.with_suffix(".tmp")
            torch.save(
                dict(
                    state=model.state_dict(),
                    epoch=epoch,
                    validation_dice=score,
                    train=sha256(SPLITS / "train.jsonl"),
                    validation=sha256(SPLITS / "validation.jsonl"),
                    grounder=sha256(STUDY / "grounder-lock.json"),
                    implementation=sha256(__file__),
                ),
                tmp,
            )
            tmp.replace(out)
        if epoch % 3 == 0:
            print(args.rank, epoch, score, flush=True)
    out.with_suffix(".json").write_text(
        json.dumps(
            dict(
                history=history,
                checkpoint=sha256(out),
                selected_epoch=history.index(max(history)),
                count_train=len(x),
                count_validation=len(vy),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
