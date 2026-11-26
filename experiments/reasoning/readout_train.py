"""Fit low-rank updates to Qwen's final attention Q/K using paired edit regions and instruction negatives."""

import argparse
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

from experiments.reasoning.data import DATA, SPLITS, STUDY, manifest
from experiments.reasoning.grounding import grounding_loss, source_inputs
from experiments.reasoning.readout_cache import CACHE
from experiments.reasoning.select_grounder import load_ensemble, predict
from experiments.resplit.spatial import rgb
from training.paper_utils import sha256

OUT = STUDY / "readout-training"
KEYS = ("xq", "xk", "q", "k", "cq", "sq", "ck", "sk")


class ProjectionUpdate(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.down = nn.Linear(inputs, 8, bias=False)
        self.up = nn.Linear(8, outputs, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, h):
        return 2 * self.up(self.down(h))


class Readout(nn.Module):
    def __init__(self, mode):
        super().__init__()
        native = torch.load(
            CACHE / "native-norm.pt", map_location="cpu", weights_only=True
        )
        self.eps = native["eps"]
        self.scaling = native["scale"]
        self.register_buffer("qnorm", native["q"].float())
        self.register_buffer("knorm", native["k"].float())
        self.mode = mode
        self.query = ProjectionUpdate(2560, 4096)
        self.key = ProjectionUpdate(2560, 1024)
        if mode == "frozen":
            self.query.requires_grad_(False)
            self.key.requires_grad_(False)
        self.heads = nn.Parameter(torch.zeros(16))
        self.gain = nn.Parameter(torch.tensor(0.25))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def normalize(self, tensor, weight):
        return F.rms_norm(tensor.float(), (tensor.size(-1),), eps=self.eps) * (
            1 + weight
        )

    def forward(self, x, prior):
        q, k = x["q"].float(), x["k"].float()
        if self.mode == "lora":
            q = q + self.query(x["xq"].float()).reshape_as(q)
            k = k + self.key(x["xk"].float()).reshape_as(k)
        q = self.normalize(q, self.qnorm).transpose(1, 2)
        k = self.normalize(k, self.knorm).transpose(1, 2)
        q = apply_rotary_pos_emb(q, q, x["cq"].float(), x["sq"].float())[0]
        k = apply_rotary_pos_emb(k, k, x["ck"].float(), x["sk"].float())[
            0
        ].repeat_interleave(4, dim=1)
        att = (q @ k.transpose(-2, -1)).squeeze(2) * self.scaling
        logp = torch.logsumexp(
            att.log_softmax(-1) + self.heads.log_softmax(0)[None, :, None], dim=1
        )
        correction = self.gain * (logp + math.log(256)).clamp(-7, 7) + self.bias
        return torch.logit(prior.clamp(0.005, 0.995)) + correction, correction


def source_prior(row, split, heads, mode):
    with torch.no_grad():
        args = [t[None] for t in source_inputs(row, split)]
        return predict(heads, mode, args)[0]


def load_data(split, heads, mode):
    if split not in ("train", "validation"):
        raise ValueError(split)
    rows = manifest(split)
    all_data = {k: [] for k in (*KEYS, *("wrong_" + k for k in KEYS), "prior", "label")}
    for row in rows:
        rec = torch.load(
            CACHE / split / (row["id"] + ".pt"), map_location="cpu", weights_only=True
        )
        assert (rec["source_sha256"], rec["instruction"]) == (
            row["source_sha256"],
            row["instruction"],
        )
        for key in KEYS:
            all_data[key].append(rec[key])
        for key, t in rec["wrong"].items():
            all_data["wrong_" + key].append(t)
        all_data["prior"].append(source_prior(row, split, heads, mode))
        assert sha256(DATA / row["target"]) == row["target_sha256"]
        a, b = [
            np.asarray(rgb(DATA / row[k]), dtype=np.int16) for k in ("source", "target")
        ]
        mask = (
            (np.abs(a - b).max(-1) > 12.75)
            .reshape(16, 32, 16, 32)
            .mean((1, 3))
            .astype(np.float32)
        )
        all_data["label"].append(torch.from_numpy(mask).flatten())
    return {k: torch.stack(v) for k, v in all_data.items()}


def run_batch(model, data, ids, device):
    batch = {k: v[ids].to(device) for k, v in data.items()}
    x = {k: batch[k] for k in KEYS}
    wrong = {k: batch.get("wrong_" + k, v) for k, v in x.items()}
    y, prior = batch["label"], batch["prior"]
    l, _ = model(x, prior)
    n, _ = model(wrong, prior)
    diff = ((l.softmax(-1) - n.softmax(-1)) * y).sum(-1)
    return (
        grounding_loss(l, y) + 0.3 * F.relu(0.08 - diff).mean(),
        diff,
        l.sigmoid(),
        y,
        prior,
    )


def summarise(prob, labels):
    soft = (2 * (prob * labels).sum(1) + 1) / ((prob + labels).sum(1) + 1)
    pred = prob >= 0.35
    truth = labels >= 0.25
    return dict(
        dice=float(soft.mean()),
        iou=float(((pred & truth).sum(1) / (pred | truth).sum(1).clamp_min(1)).mean()),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, required=True, choices=(0, 1))
    p.add_argument("--mode", choices=("lora", "frozen"), required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(53)
    device = torch.device("cuda", a.rank)
    torch.cuda.set_device(device)
    mode = json.loads((STUDY / "grounder-lock.json").read_text())["selected"]
    heads = load_ensemble(mode, "cpu")
    train, val = [load_data(split, heads, mode) for split in ("train", "validation")]
    model = Readout(a.mode).to(device)
    opt = torch.optim.AdamW(
        [
            {
                "params": [*model.query.parameters(), *model.key.parameters()],
                "lr": 8e-5,
            },
            {"params": [model.heads, model.gain, model.bias], "lr": 2e-3},
        ],
        weight_decay=0.01,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (a.mode + ".pt")
    best = -1
    hist = []
    for epoch in range(19):
        if epoch:
            model.train()
            for idx in torch.randperm(len(train["prior"])).split(12):
                opt.zero_grad(set_to_none=True)
                loss, *_ = run_batch(model, train, idx, device)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
        model.eval()
        with torch.no_grad():
            result = [
                run_batch(model, val, i, device)
                for i in torch.arange(len(val["prior"])).split(12)
            ]
            diff, pred, y, prior = [
                torch.cat([r[j] for r in result]) for j in (1, 2, 3, 4)
            ]
            metrics = summarise(pred, y)
            metrics["counterfactual_gap"] = float(diff.mean())
            if epoch == 0:
                base = summarise(prior, y)
            hist.append(metrics)
            if metrics["dice"] > best:
                best = metrics["dice"]
                tmp = path.with_suffix(".tmp")
                torch.save(
                    dict(
                        state=model.state_dict(),
                        epoch=epoch,
                        metrics=metrics,
                        mode=a.mode,
                        baseline=base,
                        train=sha256(SPLITS / "train.jsonl"),
                        validation=sha256(SPLITS / "validation.jsonl"),
                        code=sha256(__file__),
                    ),
                    tmp,
                )
                tmp.replace(path)
            if epoch % 3 == 0:
                print(a.mode, epoch, metrics, flush=True)
    path.with_suffix(".json").write_text(
        json.dumps(
            dict(
                history=hist,
                base=base,
                selected_epoch=max(range(len(hist)), key=lambda e: hist[e]["dice"]),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                count_train=len(train["prior"]),
                count_validation=len(val["prior"]),
                checkpoint=sha256(path),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
