"""Build native FLUX input caches for the same grounded prompt for paired training and validation."""

import argparse
import json

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from transformers import AutoTokenizer, Qwen3ForCausalLM

from experiments.reasoning.data import DATA, MODELS, SPLITS, STUDY, manifest
from experiments.reasoning.grounding import source_inputs
from experiments.reasoning.select_grounder import load_ensemble, predict
from experiments.resplit.spatial import rgb
from training.paper_utils import sha256

CACHE = STUDY / "flow-inputs"
TEMPLATE = (
    "Edit the provided photograph. Resolve and execute this request: {instruction} "
    "The visual grounding model identifies these intended targets or destinations: {labels}. "
    "Apply the requested operation to them and keep unrelated content faithful to the source image."
)


def prompt_text(row, split):
    rec = json.loads((STUDY / "proposals" / split / (row["id"] + ".json")).read_text())
    assert (
        rec["source_sha256"] == row["source_sha256"]
        and rec["instruction"] == row["instruction"]
    )
    answer = rec["answer"]
    try:
        items = json.loads(answer[answer.index("[") : answer.rindex("]") + 1])
    except (ValueError, TypeError):
        items = []
    labels = [
        item["label"]
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get("label"), str)
        and item["label"].strip()
    ]
    return TEMPLATE.format(
        instruction=row["instruction"],
        labels=", ".join(labels) or "infer the target from the request",
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["validation", "train"],
        choices=("train", "validation"),
    )
    a = parser.parse_args()
    device = torch.device("cuda", a.rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    root = MODELS / "FLUX.2-klein-base-4B"
    model = (
        Qwen3ForCausalLM.from_pretrained(
            root / "text_encoder",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    vae = (
        AutoencoderKLFlux2.from_pretrained(
            root / "vae", dtype=torch.float32, local_files_only=True
        )
        .to(device)
        .eval()
    )
    proc = Flux2ImageProcessor(vae_scale_factor=16)
    mu = vae.bn.running_mean.reshape(1, -1, 1, 1)
    std = (vae.bn.running_var.reshape(1, -1, 1, 1) + vae.config.batch_norm_eps).sqrt()

    def encode(image):
        x = proc.preprocess(image, height=512, width=512).to(device)
        z = Flux2KleinPipeline._patchify_latents(vae.encode(x).latent_dist.mode())
        return Flux2KleinPipeline._pack_latents((z - mu) / std)[0].cpu().bfloat16()

    mode = json.loads((STUDY / "grounder-lock.json").read_text())["selected"]
    heads = load_ensemble(mode, device)
    for split in a.splits:
        rows = manifest(split)[a.rank :: 2]
        folder = CACHE / split
        folder.mkdir(parents=True, exist_ok=True)
        for n, row in enumerate(rows):
            key = row["id"]
            path = folder / (key + ".pt")
            identity = dict(
                source_sha256=row["source_sha256"],
                target_sha256=row["target_sha256"],
                instruction=row["instruction"],
                lock=sha256(STUDY / "grounder-lock.json"),
                proposal=sha256(STUDY / "proposals" / split / (key + ".json")),
                code=sha256(__file__),
            )
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if saved.get("identity") != identity:
                    raise ValueError(
                        f"Training cache differs from current inputs: {path}"
                    )
                continue
            source = DATA / row["source"]
            assert sha256(source) == row["source_sha256"]
            s = rgb(source)
            feat = [v[None].to(device) for v in source_inputs(row, split)]
            p = predict(heads, mode, feat).view(1, 1, 16, 16)
            p = (
                torch.nn.functional.interpolate(
                    p, (32, 32), mode="bilinear", align_corners=False
                )
                .flatten()
                .cpu()
            )
            prompt = prompt_text(row, split)
            embed = Flux2KleinPipeline._get_qwen3_prompt_embeds(
                model, tokenizer, prompt, device=device
            )[0].cpu()
            record = dict(
                id=key,
                identity=identity,
                source=encode(s),
                prompt=embed,
                mask=p,
                source_sha256=row["source_sha256"],
                instruction=row["instruction"],
                text=prompt,
            )
            target = DATA / row["target"]
            assert sha256(target) == row["target_sha256"]
            t = rgb(target)
            delta = (
                np.abs(
                    np.asarray(t, dtype=np.int16) - np.asarray(s, dtype=np.int16)
                ).max(-1)
                > 12.75
            )
            record.update(
                target=encode(t),
                region=torch.from_numpy(
                    delta.reshape(32, 16, 32, 16).mean((1, 3)).astype(np.float32)
                ).flatten(),
            )
            tmp = path.with_suffix(".tmp")
            torch.save(record, tmp)
            tmp.replace(path)
            if n % 40 == 0:
                print(split, a.rank, n + 1, len(rows), flush=True)
        (folder / f"complete-{a.rank}.json").write_text(
            json.dumps(
                dict(
                    ids=[r["id"] for r in rows],
                    code=sha256(__file__),
                    manifest=sha256(SPLITS / (split + ".jsonl")),
                )
            )
        )


if __name__ == "__main__":
    main()
