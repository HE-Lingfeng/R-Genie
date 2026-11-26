"""Select the source-conditioned mask head on the fixed development partition."""

import json

import torch

from experiments.reasoning.data import SPATIAL, SPLITS
from experiments.reasoning.grounding import STUDY, Grounder, control, training_data
from experiments.resplit.spatial import SpatialHead, quality
from training.paper_utils import sha256


def load_ensemble(mode, device):
    if mode == "parent":
        models = []
        for seed in (0, 1, 2):
            path = SPATIAL / f"seed{seed}.pt"
            if (
                sha256(path)
                != json.loads(path.with_suffix(".json").read_text())["weights_sha256"]
            ):
                raise ValueError(f"Invalid spatial checkpoint: {path}")
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved["train_sha256"] != sha256(SPLITS / "train.jsonl"):
                raise ValueError(
                    "The baseline spatial head used a different training set"
                )
            model = SpatialHead().to(device)
            model.load_state_dict(saved["model"])
            models.append(model.eval().requires_grad_(False))
        return models
    models = []
    for seed in (0, 1, 2):
        p = STUDY / "grounder" / mode / f"seed{seed}.pt"
        assert sha256(p) == json.loads(p.with_suffix(".json").read_text())["checkpoint"]
        record = torch.load(p, weights_only=True, map_location="cpu")
        assert (
            record["train"] == sha256(SPLITS / "train.jsonl") and record["mode"] == mode
        )
        head = Grounder().to(device)
        head.load_state_dict(record["state"])
        models.append(head.eval().requires_grad_(False))
    return models


@torch.inference_mode()
def predict(models, mode, inputs):
    if mode == "parent":
        args = (inputs[0], inputs[1][:, -1], inputs[3])
    else:
        args = control(inputs, mode)
    return torch.stack([h(*args).sigmoid() for h in models]).mean(0)


def metrics(p, y):
    logits = torch.logit(p.clamp(1e-6, 1 - 1e-6))
    hit = y.gather(1, p.argmax(-1, keepdim=True)) >= 0.25
    return {"iou": float(quality(logits, y)), "pointing": float(hit.float().mean())}


def main():
    torch.set_num_threads(3)
    device = "cpu"
    *data, y = training_data("validation", device)
    scores = {}
    decisions = {}
    for mode in ("parent", "latent", "full"):
        models = load_ensemble(mode, device)
        p = torch.cat(
            [predict(models, mode, b) for b in zip(*(v.split(12) for v in data))]
        )
        scores[mode] = metrics(p, y)
        decisions[mode] = scores[mode]["iou"]
        if mode == "full":
            for name, altered in (("boxes_only", data[-2]), ("sam_only", data[-1])):
                scores[name] = metrics(altered, y)
            swapped = (*data[:-2], data[-2].roll(1, 0), data[-1].roll(1, 0))
            pp = torch.cat(
                [predict(models, mode, b) for b in zip(*(v.split(12) for v in swapped))]
            )
            scores["shuffled_geometry"] = metrics(pp, y)
    chosen = max(decisions, key=decisions.get)
    result = dict(
        selected=chosen,
        scores=scores,
        validation=sha256(SPLITS / "validation.jsonl"),
        count=len(y),
        protocol=sha256(__file__),
    )
    filename = STUDY / "grounder-lock.json"
    if filename.exists() and json.loads(filename.read_text()) != result:
        raise ValueError("Refusing to replace grounding decision")
    filename.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
