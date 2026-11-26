"""Train the intent head and regional flow LoRA on two GPUs.

Use after preparing the resplit manifests and the parent FLUX generation LoRA.
Example: RGENIE_DATA=/path/to/REditBench python -m experiments.reasoning.train --parent runs/flux2_lora/best.pt
Model, feature, manifest and output directories are configured in reasoning.data.
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.reasoning.data import (
    AUX_CACHE,
    DATA,
    FEATURES,
    MODELS,
    SPATIAL,
    SPLITS,
)


def run(module, *args):
    subprocess.run([sys.executable, "-m", module, *map(str, args)], check=True)


def two_ranks(module, *common, modes=None):
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = []
        for rank in (0, 1):
            variant = ("--mode", modes[rank]) if modes else ()
            jobs.append(pool.submit(run, module, "--rank", rank, *common, *variant))
        for job in jobs:
            job.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    if not args.parent.is_file():
        parser.error(f"Missing trained generation LoRA: {args.parent}")
    run(
        "experiments.qwen35.extract_grounding_features",
        "--data",
        DATA,
        "--manifests",
        SPLITS,
        "--model",
        MODELS / "Qwen3.5-4B",
        "--output",
        FEATURES,
    )
    two_ranks(
        "experiments.resplit.spatial",
        "--data",
        DATA,
        "--manifests",
        SPLITS,
        "--features",
        FEATURES,
        "--output",
        SPATIAL,
    )
    two_ranks("experiments.reasoning.proposals")
    two_ranks("experiments.reasoning.segments")
    two_ranks("experiments.reasoning.grounding", modes=("full", "latent"))
    run("experiments.reasoning.select_grounder")
    two_ranks("experiments.reasoning.refine_train")
    run("experiments.resplit.explicit_pairs", "--data", DATA, "--manifests", SPLITS)
    two_ranks(
        "experiments.resplit.aux_cache",
        "--train-only",
        "--data",
        DATA,
        "--model",
        MODELS / "FLUX.2-klein-base-4B",
        "--manifests",
        SPLITS,
        "--output",
        AUX_CACHE,
    )
    two_ranks("experiments.reasoning.flow_cache")
    run(
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "--module",
        "experiments.reasoning.flow",
        "--parent",
        args.parent,
    )


if __name__ == "__main__":
    main()
