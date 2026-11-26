"""Cache native edit velocities for supervised training on source/target pairs.

Prepare source-only conditions with training.cache_transport_intentions. For
each split, run conditions, velocities and seal in that order. Each row of
--manifest supplies id, instruction, source, target and their SHA-256 hashes.
Image paths are relative to --data-root. --base is the Qwen-Image-Edit-2511
Diffusers directory and --quantized-weight is its Q4_K_M GGUF transformer.
The conditions and velocities stages support one shard per GPU. Run seal once
after every shard finishes. Use a separate cache directory for each backbone and
grounder revision. Pass <cache>/velocities to train_intention_transport.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import (
    GGUFQuantizationConfig,
    QwenImageEditPlusPipeline,
    QwenImageTransformer2DModel,
)
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    CONDITION_IMAGE_SIZE,
    VAE_IMAGE_SIZE,
    calculate_dimensions,
)
from PIL import Image, ImageOps

from training.train_intention_transport import digest


def rgb(path):
    with Image.open(path) as image:
        return (
            ImageOps.exif_transpose(image)
            .convert("RGB")
            .resize((512, 512), Image.Resampling.BICUBIC)
        )


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    torch.save(obj, tmp)
    tmp.replace(path)


def encoder(base, device):
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        base, transformer=None, torch_dtype=torch.bfloat16, local_files_only=True
    )
    pipe.text_encoder.to(device).eval()
    pipe.vae.to(device).eval()
    pipe.vae.enable_tiling()
    return pipe


def backbone(base, weight, device):
    model = QwenImageTransformer2DModel.from_single_file(
        str(weight),
        config=str(base),
        subfolder="transformer",
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    return model.to(device).eval().requires_grad_(False)


def condition(pipe, source, request, device, target=None):
    def encoded(im, size):
        pixels = (
            pipe.image_processor.preprocess(im, height=size, width=size)
            .unsqueeze(2)
            .to(device=device, dtype=pipe.vae.dtype)
        )
        z = pipe._encode_vae_image(pixels, None)
        return pipe._pack_latents(z, 1, z.shape[1], z.shape[3], z.shape[4])[0].cpu()

    w, h = calculate_dimensions(CONDITION_IMAGE_SIZE, 1.0)
    text_image = [pipe.image_processor.resize(source, h, w)]
    e, m = pipe.encode_prompt(prompt=request, image=text_image, device=device)
    neg, nm = pipe.encode_prompt(prompt=" ", image=text_image, device=device)
    refsize, _ = calculate_dimensions(VAE_IMAGE_SIZE, 1.0)
    out = {
        "source": encoded(source, 512),
        "reference": encoded(source, refsize),
        "e": e.cpu(),
        "m": m.cpu() if m is not None else None,
        "neg": neg.cpu(),
        "nm": nm.cpu() if nm is not None else None,
    }
    if target is not None:
        out["target"] = encoded(target, 512)
        delta = (
            np.abs(
                np.asarray(source, dtype=np.int16) - np.asarray(target, dtype=np.int16)
            ).max(-1)
            > 12.75
        )
        out["region"] = torch.from_numpy(
            delta.reshape(32, 16, 32, 16).mean((1, 3)).astype(np.float32)
        ).flatten()
    assert out["source"].shape == (1024, 64) and out["reference"].shape == (4096, 64)
    return out


def native_velocity(model, z, rec, sigma, device):
    ref = rec["reference"].to(device)
    kwargs = {
        "hidden_states": torch.cat([z.to(ref), ref])[None],
        "timestep": sigma.view(1).to(device=device, dtype=ref.dtype),
        "img_shapes": [[(1, 32, 32), (1, 64, 64)]],
        "return_dict": False,
    }

    def predict(text, mask):
        return model(
            **kwargs,
            encoder_hidden_states=rec[text].to(device),
            encoder_hidden_states_mask=rec[mask].to(device)
            if rec[mask] is not None
            else None,
        )[0][0, : z.shape[0]]

    cond = predict("e", "m")
    neg = predict("neg", "nm")
    guided = neg + 4 * (cond - neg)
    norm = torch.norm(guided, dim=-1, keepdim=True).clamp_min(1e-6)
    return (guided * (torch.norm(cond, dim=-1, keepdim=True) / norm)).cpu()


def fixed_velocities(model, rec, ident, device):
    seed = int.from_bytes(
        hashlib.sha256(("RGenie/transport/noise/" + ident).encode()).digest()[:8], "big"
    ) % (2**63 - 1)
    rng = torch.Generator().manual_seed(seed)
    target = rec["target"].float()
    sigmas = []
    noises = []
    velocities = []
    for lo, hi in ((0.05, 0.28), (0.34, 0.68), (0.72, 0.97)):
        s = lo + (hi - lo) * torch.rand((), generator=rng)
        eps = torch.randn(target.shape, generator=rng).to(torch.bfloat16)
        z = (1 - s) * target + s * eps.float()
        v = native_velocity(model, z, rec, s, device)
        if not torch.isfinite(v).all():
            raise FloatingPointError(ident)
        sigmas.append(s)
        noises.append(eps)
        velocities.append(v)
    out = {
        k: rec[k]
        for k in (
            "source",
            "target",
            "region",
            "id",
            "prior",
            "intention",
            "source_sha256",
            "target_sha256",
            "instruction",
        )
    }
    out.update(
        sigmas=torch.stack(sigmas),
        noise=torch.stack(noises),
        velocity=torch.stack(velocities),
    )
    return out


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("conditions", "velocities", "seal"))
    for key in (
        "base",
        "quantized-weight",
        "manifest",
        "data-root",
        "intents",
        "cache",
    ):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--split", required=True, choices=("train", "validation"))
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=1)
    a = p.parse_args()
    if not 0 <= a.rank < a.world:
        p.error("rank must be in [0, world)")
    torch.set_num_threads(2)
    rows = [json.loads(line) for line in a.manifest.read_text().splitlines() if line]
    cond = a.cache / "conditions" / a.split
    vel = a.cache / "velocities" / a.split
    if a.stage == "seal":
        records = []
        for r in rows:
            f = vel / (r["id"] + ".pt")
            v = torch.load(f, map_location="cpu", weights_only=True)
            assert (v["source_sha256"], v["target_sha256"], v["instruction"]) == (
                r["source_sha256"],
                r["target_sha256"],
                r["instruction"],
            )
            records.append(
                {
                    "id": r["id"],
                    "source_sha256": r["source_sha256"],
                    "sha256": digest(f),
                }
            )
        tmp = vel / "COMPLETE.tmp"
        tmp.write_text(
            json.dumps({"records": records, "manifest": digest(a.manifest)}, indent=2)
        )
        tmp.replace(vel / "COMPLETE.json")
        return
    device = torch.device("cuda", a.rank)
    torch.cuda.set_device(device)
    model = (
        encoder(a.base, device)
        if a.stage == "conditions"
        else backbone(a.base, a.quantized_weight, device)
    )
    for n, r in enumerate(rows[a.rank :: a.world]):
        key = r["id"]
        file = (cond if a.stage == "conditions" else vel) / (key + ".pt")
        if file.exists():
            cached = torch.load(file, map_location="cpu", weights_only=True)
            assert (
                cached["id"],
                cached["instruction"],
                cached["source_sha256"],
                cached["target_sha256"],
            ) == (key, r["instruction"], r["source_sha256"], r["target_sha256"])
            continue
        if a.stage == "conditions":
            s, t = [a.data_root / r[k] for k in ("source", "target")]
            assert (digest(s), digest(t)) == (r["source_sha256"], r["target_sha256"])
            intent = torch.load(
                a.intents / a.split / (key + ".pt"),
                map_location="cpu",
                weights_only=True,
            )
            assert (intent["id"], intent["source_sha256"], intent["instruction"]) == (
                key,
                r["source_sha256"],
                r["instruction"],
            )
            cache = condition(model, rgb(s), r["instruction"], device, target=rgb(t))
            cache.update(intent)
            cache["target_sha256"] = r["target_sha256"]
        else:
            v = torch.load(cond / (key + ".pt"), map_location="cpu", weights_only=True)
            assert (v["source_sha256"], v["target_sha256"], v["instruction"]) == (
                r["source_sha256"],
                r["target_sha256"],
                r["instruction"],
            )
            cache = fixed_velocities(model, v, key, device)
        put(file, cache)
        if n % 20 == 0:
            print(
                a.stage,
                a.split,
                a.rank,
                n + 1,
                len(rows[a.rank :: a.world]),
                flush=True,
            )


if __name__ == "__main__":
    main()
