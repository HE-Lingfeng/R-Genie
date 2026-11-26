"""Train implicit-instruction grounding on paired edits and source-only language/vision proposals."""

import argparse
import hashlib
import json

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from experiments.reasoning.data import DATA, FEATURES, SPLITS, STUDY, manifest
from experiments.resplit.spatial import quality, rgb
from training.paper_utils import sha256


class Grounder(nn.Module):
    def __init__(self, width=64, slots=4):
        super().__init__()
        self.visual = nn.Sequential(nn.LayerNorm(2560), nn.Linear(2560, width))
        self.text = nn.Sequential(nn.LayerNorm(2560), nn.Linear(2560, width))
        self.slots = nn.Parameter(torch.randn(1, slots, width) * 0.05)
        self.read_words = nn.MultiheadAttention(width, 4, batch_first=True)
        self.read_image = nn.MultiheadAttention(width, 4, batch_first=True)
        self.read_intention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.refine = nn.Sequential(
            nn.Conv2d(width + 4, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Conv2d(width, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )
        self.proposal_scale = nn.Parameter(torch.tensor([1.0, 1.4]))
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, visual, words, pad, prior, box, segment):
        v = self.visual(visual.float())
        t = self.text(words.float())
        queries = self.slots.expand(v.shape[0], -1, -1) + t[:, -1:]
        queries = (
            queries
            + self.read_words(queries, t, t, key_padding_mask=pad, need_weights=False)[
                0
            ]
        )
        queries = queries + self.read_image(queries, v, v, need_weights=False)[0]
        aligned = self.read_intention(v, queries, queries, need_weights=False)[0]
        agreement = F.cosine_similarity(v, aligned, dim=-1)
        native = (prior.float() * 256).clamp_min(1e-6).log()
        cues = torch.stack((agreement, native, box, segment), dim=-1)
        grid = torch.cat((v * F.layer_norm(aligned, aligned.shape[-1:]), cues), dim=-1)
        correction = self.refine(
            grid.transpose(1, 2).reshape(len(v), -1, 16, 16)
        ).flatten(1)
        return (
            native
            - 1.8
            + correction
            + self.proposal_scale[0] * box
            + self.proposal_scale[1] * segment
        )


def box_grid(boxes):
    axis = torch.arange(16) / 16
    result = torch.zeros(16, 16)
    for x0, y0, x1, y1 in boxes:
        x = (
            torch.minimum(axis + 1 / 16, torch.tensor(x1))
            - torch.maximum(axis, torch.tensor(x0))
        ).clamp_min(0) * 16
        y = (
            torch.minimum(axis + 1 / 16, torch.tensor(y1))
            - torch.maximum(axis, torch.tensor(y0))
        ).clamp_min(0) * 16
        result = torch.maximum(result, y[:, None] * x)
    return result.flatten()


def source_inputs(row, split):
    key = row["id"]
    path = FEATURES / (key + ".pt")
    cached = torch.load(path, map_location="cpu", weights_only=True)
    assert (
        cached["instruction_sha256"]
        == hashlib.sha256(row["instruction"].encode()).hexdigest()
    )
    assert cached["source_sha256"] == row["source_sha256"]
    proposal = json.loads((STUDY / "proposals" / split / (key + ".json")).read_text())
    assert (
        proposal["instruction"] == row["instruction"]
        and proposal["source_sha256"] == row["source_sha256"]
    )
    words = cached["text"][-63:]
    padded = torch.zeros(64, 2560, dtype=words.dtype)
    padded[-len(words) - 1 : -1] = words
    padded[-1] = cached["readout"]
    pad = torch.arange(64) < 63 - len(words)
    with Image.open(STUDY / "segments" / split / (key + ".png")) as image:
        mask = torch.tensor(
            np.asarray(image, dtype=np.float32).reshape(16, 32, 16, 32).mean((1, 3))
            / 255
        ).flatten()
    return (
        cached["visual"],
        padded,
        pad,
        cached["native_prior"],
        box_grid(proposal["boxes"]),
        mask,
    )


def training_data(split, device):
    if split not in ("validation", "train"):
        raise ValueError(
            "Grounding training accepts only optimization and development records"
        )
    rows = manifest(split)
    parts = []
    for row in rows:
        source, target = [DATA / row[k] for k in ("source", "target")]
        assert (
            sha256(source) == row["source_sha256"]
            and sha256(target) == row["target_sha256"]
        )
        delta = (
            np.abs(
                np.asarray(rgb(source), dtype=np.int16)
                - np.asarray(rgb(target), dtype=np.int16)
            ).max(-1)
            > 12.75
        )
        y = torch.tensor(
            delta.reshape(16, 32, 16, 32).mean((1, 3)), dtype=torch.float32
        ).flatten()
        parts.append((*source_inputs(row, split), y))
    return [torch.stack(part).to(device) for part in zip(*parts)]


def grounding_loss(logits, y):
    p = logits.sigmoid()
    dice = 1 - (2 * (p * y).sum(1) + 1) / ((p + y).sum(1) + 1)
    # A normalized foreground distribution gives a small implied object the same
    # total mass as a large one. Skip empty targets in this component.
    mass = y.sum(1)
    activation = -(y * logits.log_softmax(1)).sum(1) / mass.clamp_min(1e-6)
    return (
        F.binary_cross_entropy_with_logits(logits, y)
        + dice.mean()
        + 0.15 * (activation * (mass > 0)).mean()
    )


def control(inputs, mode):
    if mode == "latent":
        return (*inputs[:4], torch.zeros_like(inputs[4]), torch.zeros_like(inputs[5]))
    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True, choices=(0, 1))
    parser.add_argument("--mode", choices=("full", "latent"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate seeds")
    device = torch.device("cuda", args.rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    train = training_data("train", device)
    val = training_data("validation", device)
    out = STUDY / "grounder" / args.mode
    out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = Grounder().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3.5e-4, weight_decay=0.02)
        best = -1
        history = []
        for epoch in range(1, 31):
            model.train()
            for ids in torch.randperm(len(train[-1]), device=device).split(24):
                batch = [part[ids] for part in train]
                inputs = control(batch[:-1], args.mode)
                if args.mode == "full":
                    # Provide examples without a geometric proposal to train the fallback.
                    keep = (torch.rand(len(ids), 1, device=device) > 0.12).float()
                    inputs = (*inputs[:4], inputs[4] * keep, inputs[5] * keep)
                logits = model(*inputs)
                loss = grounding_loss(logits, batch[-1])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
            model.eval()
            with torch.no_grad():
                pred = torch.cat(
                    [
                        model(*control(batch, args.mode))
                        for batch in zip(*(x.split(20) for x in val[:-1]))
                    ]
                )
                iou = quality(pred, val[-1]).item()
            history.append(iou)
            if iou > best:
                best = iou
                path = out / f"seed{seed}.pt"
                tmp = path.with_suffix(".tmp")
                torch.save(
                    dict(
                        state=model.state_dict(),
                        mode=args.mode,
                        epoch=epoch,
                        seed=seed,
                        iou=iou,
                        train=sha256(SPLITS / "train.jsonl"),
                        validation=sha256(SPLITS / "validation.jsonl"),
                        code=sha256(__file__),
                    ),
                    tmp,
                )
                tmp.replace(path)
            if epoch % 5 == 0:
                print(args.mode, seed, epoch, round(iou, 4), round(best, 4), flush=True)
        (out / f"seed{seed}.json").write_text(
            json.dumps(
                dict(val_weak_iou=history, checkpoint=sha256(out / f"seed{seed}.pt")),
                indent=2,
            )
        )
    (out / "COMPLETE.json").write_text(
        json.dumps(
            dict(
                seeds=args.seeds,
                train_count=len(train[-1]),
                validation_count=len(val[-1]),
            )
        )
    )


if __name__ == "__main__":
    main()
