"""Prepare the learned source-only conditions needed for training an edit-velocity adapter.

Run after training and selecting the reasoning grounder. Use the same data paths
and feature caches as experiments.reasoning.select_grounder. Pass the resulting
output directory to training.cache_intention_transport with --intents.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.reasoning.grounding import STUDY, source_inputs
from experiments.reasoning.select_grounder import load_ensemble, predict


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.rank < args.world:
        parser.error("rank must be in [0, world)")
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    torch.set_num_threads(2)
    mode = json.loads((STUDY / "grounder-lock.json").read_text())["selected"]
    ensemble = load_ensemble(mode, "cpu")
    directory = args.out / args.split
    directory.mkdir(parents=True, exist_ok=True)
    for row in rows[args.rank :: args.world]:
        inputs = source_inputs(row, args.split)
        prior = predict(ensemble, mode, [value[None] for value in inputs]).view(
            1, 1, 16, 16
        )
        prior = F.interpolate(
            prior, (32, 32), mode="bilinear", align_corners=False
        ).flatten()
        rec = {
            "id": row["id"],
            "source_sha256": row["source_sha256"],
            "instruction": row["instruction"],
            "prior": prior,
            "intention": inputs[1][-1],
        }
        path = directory / (row["id"] + ".pt")
        tmp = path.with_suffix(".partial")
        torch.save(rec, tmp)
        tmp.replace(path)


if __name__ == "__main__":
    main()
