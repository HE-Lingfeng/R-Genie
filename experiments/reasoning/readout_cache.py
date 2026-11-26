"""Capture the native final Qwen attention inputs for the intended-target readout."""

import argparse
import hashlib
import json

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from experiments.qwen35.extract_grounding_features import CUE, FRAMING
from experiments.reasoning.data import DATA, MODELS, SPLITS, STUDY, manifest
from experiments.resplit.spatial import rgb
from training.paper_utils import sha256

CACHE = STUDY / "readout-cache"


@torch.inference_mode()
def extract(model, processor, image, instruction, device):
    msgs = [
        dict(
            role="user",
            content=[
                dict(type="image", image=image),
                dict(type="text", text=FRAMING + instruction),
            ],
        )
    ]
    inp = processor.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    suffix = processor.tokenizer(
        CUE, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    inp["input_ids"] = torch.cat([inp.input_ids, suffix], 1)
    for key in ("attention_mask", "mm_token_type_ids", "token_type_ids"):
        if key in inp:
            inp[key] = torch.cat(
                [inp[key], torch.full_like(suffix, int(key == "attention_mask"))], 1
            )
    inp = inp.to(device)
    visual = inp.input_ids[0] == model.config.image_token_id
    assert visual.sum().item() == 256
    att = model.model.language_model.layers[-1].self_attn
    result = {}

    def capture(module, args, kwargs):
        x = kwargs["hidden_states"]
        cos, sin = kwargs["position_embeddings"]
        v = x[:, visual]
        query = x[:, -1:]
        q = att.q_proj(query).view(1, 1, -1, 2 * att.head_dim)[..., : att.head_dim]
        k = att.k_proj(v).view(1, 256, -1, att.head_dim)
        for name, tensor in dict(
            xq=query,
            xk=v,
            q=q,
            k=k,
            cq=cos[:, -1:],
            sq=sin[:, -1:],
            ck=cos[:, visual],
            sk=sin[:, visual],
        ).items():
            result[name] = tensor[0].contiguous().cpu()

    h = att.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        model.model(**inp, use_cache=False)
    finally:
        h.remove()
    last_image = visual.nonzero()[-1, 0].item()
    result["prefix_sha256"] = hashlib.sha256(
        inp.input_ids[0, : last_image + 1].cpu().numpy().tobytes()
    ).hexdigest()
    return result


def wrong_instruction(rows, index):
    # Fixed contrast from this partition; test inference needs only the true request.
    for offset in range(1, len(rows)):
        other = rows[(index + offset) % len(rows)]
        if (
            other["instruction"].strip().casefold()
            != rows[index]["instruction"].strip().casefold()
        ):
            return other["instruction"]
    raise ValueError("No distinct negative instruction")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, required=True, choices=(0, 1))
    p.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation"),
        default=["validation", "train"],
    )
    args = p.parse_args()
    device = torch.device("cuda", 0 if torch.cuda.device_count() == 1 else args.rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    path = MODELS / "Qwen3.5-4B"
    origin = json.loads((path / "origin.json").read_text())
    assert (
        origin["revision"]
        == json.loads(open("experiments/qwen35/assets.json").read())["revision"]
    )
    proc = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = (
        Qwen3_5ForConditionalGeneration.from_pretrained(
            path,
            dtype=torch.bfloat16,
            device_map={"": str(device)},
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .eval()
        .requires_grad_(False)
    )
    att = model.model.language_model.layers[-1].self_attn
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.rank == 0:
        t = CACHE / "native-norm.tmp"
        torch.save(
            dict(
                q=att.q_norm.weight.detach().cpu(),
                k=att.k_norm.weight.detach().cpu(),
                eps=att.q_norm.eps,
                revision=origin["revision"],
                scale=att.scaling,
            ),
            t,
        )
        t.replace(CACHE / "native-norm.pt")
    for split in args.splits:
        rows = manifest(split)
        out = CACHE / split
        out.mkdir(exist_ok=True)
        for n in range(args.rank, len(rows), 2):
            row = rows[n]
            dest = out / (row["id"] + ".pt")
            if (
                dest.exists()
                and torch.load(dest, weights_only=True, map_location="cpu").get(
                    "format"
                )
                == 2
            ):
                continue
            path = DATA / row["source"]
            assert sha256(path) == row["source_sha256"]
            im = rgb(path)
            record = extract(model, proc, im, row["instruction"], device)
            neg = wrong_instruction(rows, n)
            other = extract(model, proc, im, neg, device)
            # Flash/SDPA kernels for a different total sequence length can round differently.
            delta = other["xk"].float() - record["xk"].float()
            relative = float(delta.norm() / record["xk"].float().norm().clamp_min(1e-8))
            if other["prefix_sha256"] != record["prefix_sha256"]:
                raise ValueError(f"Source token prefix changed: {row['id']}")
            record.update(
                wrong={
                    k: other[k] for k in ("xq", "xk", "q", "k", "cq", "sq", "ck", "sk")
                },
                prefix_relative_error=relative,
                wrong_instruction=neg,
            )
            record.update(
                format=2,
                id=row["id"],
                source_sha256=row["source_sha256"],
                instruction=row["instruction"],
                revision=origin["revision"],
            )
            tmp = dest.with_suffix(".tmp")
            torch.save(record, tmp)
            tmp.replace(dest)
            if n % 60 < 2:
                print(split, args.rank, n, len(rows), flush=True)
        (out / f"complete-{args.rank}.json").write_text(
            json.dumps(
                dict(
                    manifest=sha256(SPLITS / (split + ".jsonl")),
                    code=sha256(__file__),
                    ids=[r["id"] for r in rows[args.rank :: 2]],
                )
            )
        )


if __name__ == "__main__":
    main()
