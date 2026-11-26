"""Turn source-only reasoning boxes into dense SAM 2 candidate masks."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import Sam2Model, Sam2Processor

from experiments.reasoning.data import DATA, MODELS, STUDY, manifest
from training.paper_utils import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["validation", "train"],
        choices=("train", "validation"),
    )
    args = parser.parse_args()
    device = torch.device("cuda", args.rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    root = MODELS / "sam2.1-tiny"
    assert json.loads((root / "origin.json").read_text()) == json.loads(
        Path("experiments/reasoning/sam2-assets.json").read_text()
    )
    model = Sam2Model.from_pretrained(root, local_files_only=True).eval().to(device)
    processor = Sam2Processor.from_pretrained(root, local_files_only=True)
    for split in args.splits:
        rows = manifest(split)[args.rank :: 2]
        folder = STUDY / "segments" / split
        folder.mkdir(parents=True, exist_ok=True)
        for n, row in enumerate(rows):
            stem = row["id"]
            p = STUDY / "proposals" / split / (stem + ".json")
            record = json.loads(p.read_text())
            assert (
                record["instruction"] == row["instruction"]
                and record["source_sha256"] == row["source_sha256"]
            )
            dest = folder / (stem + ".png")
            if dest.exists():
                continue
            source = DATA / row["source"]
            assert sha256(source) == row["source_sha256"]
            with Image.open(source) as image:
                rgb = (
                    ImageOps.exif_transpose(image)
                    .convert("RGB")
                    .resize((512, 512), Image.Resampling.BICUBIC)
                )
            boxes = [[512 * x for x in box] for box in record["boxes"]]
            binary = np.zeros((512, 512), dtype=np.uint8)
            if boxes:
                inputs = processor(
                    images=rgb, input_boxes=[boxes], return_tensors="pt"
                ).to(device)
                with torch.inference_mode():
                    pred = model(**inputs, multimask_output=False)
                masks = processor.post_process_masks(
                    pred.pred_masks.cpu(), inputs["original_sizes"].cpu()
                )[0]
                binary = masks[:, 0].any(0).byte().numpy() * 255
            tmp = dest.with_suffix(".tmp")
            Image.fromarray(binary).save(tmp, format="PNG")
            tmp.replace(dest)
            if n % 50 == 0:
                print(split, args.rank, n + 1, flush=True)
        (folder / f"complete-{args.rank}.json").write_text(
            json.dumps({"count": len(rows), "implementation": sha256(__file__)})
        )


if __name__ == "__main__":
    main()
