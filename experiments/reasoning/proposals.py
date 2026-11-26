"""Obtain native Qwen target boxes from the source and original request only."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from experiments.reasoning.data import DATA, MODELS, SPLITS, STUDY, manifest
from training.paper_utils import sha256

PROMPT = (
    "You are identifying what to edit in the source photograph. Resolve any implicit "
    "reference using the image and world knowledge. Return only a JSON array, one "
    "entry for each visible target of the edit. Format each entry as "
    '{"label":"specific target","bbox_2d":[x_min,y_min,x_max,y_max]}. '
    "Use integer coordinates from 0 to 1000. For adding something, localize the "
    "destination. For removal, localize the entire object to remove. "
    "Include all requested targets and exclude objects that should stay. Request: "
)


def decode(answer):
    start, end = answer.find("["), answer.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        items = json.loads(answer[start : end + 1])
    except ValueError:
        return []
    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        box = item.get("bbox_2d")
        if (
            isinstance(box, list)
            and len(box) == 4
            and all(type(v) in (int, float) and 0 <= v <= 1000 for v in box)
        ):
            if box[0] < box[2] and box[1] < box[3]:
                result.append([v / 1000 for v in box])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation"),
        default=["validation", "train"],
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device("cuda", args.rank)
    torch.cuda.set_device(device)
    root = MODELS / "Qwen3.5-4B"
    revision = json.loads((root / "origin.json").read_text())["revision"]
    if json.loads((root / "origin.json").read_text()) != json.loads(
        Path("experiments/qwen35/assets.json").read_text()
    ):
        raise ValueError("Expected the registered Qwen3.5-4B assets")
    processor = AutoProcessor.from_pretrained(root, local_files_only=True)
    model = (
        Qwen3_5ForConditionalGeneration.from_pretrained(
            root,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    processor.tokenizer.padding_side = "left"
    for split in args.splits:
        folder = STUDY / "proposals" / split
        folder.mkdir(parents=True, exist_ok=True)
        rows = manifest(split)[args.rank :: 2]
        todo = []
        for row in rows:
            dest = folder / (row["id"] + ".json")
            if dest.exists():
                val = json.loads(dest.read_text())
                if (
                    val["revision"],
                    val["prompt"],
                    val["instruction"],
                    val["source_sha256"],
                ) != (revision, PROMPT, row["instruction"], row["source_sha256"]):
                    raise ValueError("Conflicting proposal " + row["id"])
            else:
                todo.append(row)
        for start in range(0, len(todo), 4):
            batch = todo[start : start + 4]
            messages = []
            for row in batch:
                path = DATA / row["source"]
                assert sha256(path) == row["source_sha256"]
                with Image.open(path) as image:
                    rgb = (
                        ImageOps.exif_transpose(image)
                        .convert("RGB")
                        .resize((512, 512), Image.Resampling.BICUBIC)
                    )
                messages.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": rgb},
                                {"type": "text", "text": PROMPT + row["instruction"]},
                            ],
                        }
                    ]
                )
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                padding=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                ids = model.generate(
                    **inputs, max_new_tokens=160, do_sample=False, use_cache=True
                )
            answers = processor.batch_decode(
                ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            for row, answer in zip(batch, answers):
                dest = folder / (row["id"] + ".json")
                payload = dict(
                    id=row["id"],
                    source_sha256=row["source_sha256"],
                    instruction=row["instruction"],
                    prompt=PROMPT,
                    revision=revision,
                    answer=answer,
                    boxes=decode(answer),
                )
                tmp = dest.with_suffix(".tmp")
                tmp.write_text(json.dumps(payload, ensure_ascii=False))
                tmp.replace(dest)
            if start % 40 == 0:
                print(split, args.rank, start + len(batch), len(todo), flush=True)
        (folder / f"complete-{args.rank}.json").write_text(
            json.dumps(
                {"count": len(rows), "manifest": sha256(SPLITS / (split + ".jsonl"))}
            )
        )


if __name__ == "__main__":
    main()
