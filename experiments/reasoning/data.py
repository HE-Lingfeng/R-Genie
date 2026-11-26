"""Locations used by the reasoning training stages; override defaults in the environment."""

import json
import os
from pathlib import Path

from training.paper_utils import read_records, sha256

DATA = Path(os.getenv("RGENIE_DATA", "REditBench"))
MODELS = Path(os.getenv("RGENIE_MODELS", "hf_model"))
FEATURES = Path(os.getenv("RGENIE_FEATURES", "artifacts/resplit-grounding-features"))
SPLITS = Path(os.getenv("RGENIE_MANIFESTS", "artifacts/reditbench-resplit-v2"))
AUX_CACHE = Path(os.getenv("RGENIE_AUX_CACHE", "artifacts/aux-cache"))
SPATIAL = Path(os.getenv("RGENIE_SPATIAL", "runs/spatial"))
STUDY = Path(os.getenv("RGENIE_STUDY", "runs/reasoning"))


def manifest(split):
    if split not in ("train", "validation"):
        raise ValueError("Expected train or validation")
    path = SPLITS / (split + ".jsonl")
    lock = json.loads((SPLITS / "split-lock.json").read_text())
    if sha256(path) != lock["splits"][split]["sha256"]:
        raise ValueError(f"Manifest checksum changed: {path}")
    return read_records(path)
