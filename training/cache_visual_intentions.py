"""Prepare train/development appearance supervision for the visual intention head.

Use the ViT-H/14 encoder from h94/IP-Adapter, revision
018e402774aeeddd60609b4ecdb7e298259dc729, under models/image_encoder.
The intent directory contains source-only records from cache_transport_intentions.
A record pairs the source views with an edited-crop target for supervision.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.nn import functional as F
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from training.train_intention_transport import digest

ENCODER_SHA256 = "6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030"
CONFIG_BLOB = "9c841dd0314e27141ed01a24df1734e407a95c4a"


def open_image(path):
    with Image.open(path) as im:
        return (
            ImageOps.exif_transpose(im)
            .convert("RGB")
            .resize((512, 512), Image.Resampling.BICUBIC)
        )


def encoder(folder, device):
    if digest(folder / "model.safetensors") != ENCODER_SHA256:
        raise ValueError("Unexpected ViT-H/14 weights")
    raw = (folder / "config.json").read_bytes()
    if hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() != CONFIG_BLOB:
        raise ValueError("Unexpected ViT-H/14 configuration")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = (
        CLIPVisionModelWithProjection.from_pretrained(
            folder,
            local_files_only=True,
            use_safetensors=True,
            dtype=dtype,
        )
        .eval()
        .to(device)
    )
    # The published IP-Adapter uses the standard CLIP crop and normalization.
    return model, CLIPImageProcessor()


def views(image, prior):
    probability = F.interpolate(
        prior.detach().float().cpu().reshape(1, 1, 32, 32),
        (512, 512),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    selected = probability >= 0.35
    ys, xs = np.nonzero(selected)
    if len(xs):
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        side = min(512, max(192, round(1.4 * max(x1 - x0, y1 - y0))))
        x = max(0, min(512 - side, (x0 + x1 - side) // 2))
        y = max(0, min(512 - side, (y0 + y1 - side) // 2))
        rect = (x, y, x + side, y + side)
    else:
        rect = (0, 0, 512, 512)
    source = np.asarray(image, dtype=np.uint8)
    keep = ~selected
    valid = Image.fromarray((keep * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(24)
    )
    premul = Image.fromarray((source * keep[..., None]).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(24)
    )
    denom = np.asarray(valid, dtype=np.float32)[..., None]
    fill = 255 * np.asarray(premul, dtype=np.float32) / np.maximum(denom, 1)
    neutral = (
        np.median(source[keep], axis=0) if keep.any() else np.array([127, 127, 127])
    )
    fill = np.where(denom >= 4, fill, neutral)
    context = (
        np.where(keep[..., None], source, fill).round().clip(0, 255).astype(np.uint8)
    )
    return [image.crop(rect), Image.fromarray(context).crop(rect)], rect


@torch.inference_mode()
def encode_images(bundle, pictures):
    model, processor = bundle
    pixels = processor(images=pictures, return_tensors="pt").pixel_values
    weight = next(model.parameters())
    values = model(pixel_values=pixels.to(weight)).image_embeds.float().cpu()
    if values.shape != (len(pictures), 1024) or not torch.isfinite(values).all():
        raise ValueError("Invalid visual prompt embeddings")
    return values


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("encoder", "manifests", "data", "intents", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--split", choices=("train", "validation"), required=True)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    if not 0 <= a.rank < a.world or a.limit is not None and a.limit < 1:
        p.error("Invalid shard or limit")
    torch.set_num_threads(4)
    device = torch.device(a.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    file = a.manifests / (a.split + ".jsonl")
    manifest_hash = digest(file)
    rows = [json.loads(s) for s in file.read_text().splitlines() if s.strip()]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Nonunique image ID")
    rows = rows[a.rank :: a.world]
    if a.limit:
        rows = rows[: a.limit]
    directory = a.out / a.split
    directory.mkdir(parents=True, exist_ok=True)
    model = encoder(a.encoder, device)
    for i, row in enumerate(rows):
        key = row["id"]
        if not key.isdecimal():
            raise ValueError("Expected a numeric image ID")
        src, tgt = (a.data / row[n] for n in ("source", "target"))
        if (digest(src), digest(tgt)) != (row["source_sha256"], row["target_sha256"]):
            raise ValueError(f"Mismatched photo pair for {key}")
        condition = a.intents / a.split / (key + ".pt")
        meta = {
            "id": key,
            "source_sha256": row["source_sha256"],
            "target_sha256": row["target_sha256"],
            "instruction": row["instruction"],
            "intention_record": digest(condition),
            "encoder": ENCODER_SHA256,
            "manifest": manifest_hash,
        }
        f = directory / (key + ".pt")
        if f.exists():
            old = torch.load(f, map_location="cpu", weights_only=True)
            if old["identity"] != meta:
                raise ValueError(f"Conflicting cache for {key}")
            continue
        info = torch.load(condition, map_location="cpu", weights_only=True)
        if (info["id"], info["source_sha256"], info["instruction"]) != (
            key,
            row["source_sha256"],
            row["instruction"],
        ):
            raise ValueError(f"Incorrect reasoning feature for {key}")
        images, window = views(open_image(src), info["prior"])
        vectors = encode_images(model, [*images, open_image(tgt).crop(window)])
        record = {
            "identity": meta,
            "window": window,
            "source": vectors[0],
            "context": vectors[1],
            "target": vectors[2],
            "intention": info["intention"].float().cpu(),
        }
        temp = f.with_suffix(f".{a.rank}.partial")
        torch.save(record, temp)
        temp.replace(f)
        if i % 50 == 0:
            print(a.split, a.rank, i + 1, len(rows), flush=True)


if __name__ == "__main__":
    main()
