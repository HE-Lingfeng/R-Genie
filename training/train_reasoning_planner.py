"""Supervise a Qwen3.5 image-editing planner with human and optional teacher commands.

Prepare the JSONL with training.prepare_planner_supervision. Launch this module
with torchrun --standalone --nproc_per_node=2 --module training.train_reasoning_planner
and pass --base, --records, --data-root and --out.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image, ImageOps
from peft import LoraConfig, get_peft_model
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

SYSTEM = ("Rewrite the user's implicit image-editing request as one concise command for an image editor. "
          "Look at the photo and resolve who or what the user means. Specify the visible target, its "
          "location when needed, and the exact requested operation. Keep all requested details. "
          "Output only the command in English.")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def batch_prefix(processor, image, instruction):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]},
    ]
    return processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                         enable_thinking=False, return_dict=True, return_tensors="pt")


def loss_on(model, processor, device, row, root):
    with Image.open(root / row["source"]) as raw:
        im = ImageOps.exif_transpose(raw).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
    inputs = batch_prefix(processor, im, row["instruction"])
    start = inputs["input_ids"].shape[1]
    tokens = processor.tokenizer(row["command"], add_special_tokens=False, return_tensors="pt").input_ids
    eos = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    tokens = torch.cat([tokens, torch.tensor([[eos]])], dim=1)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], tokens], dim=1)
    for field in ("attention_mask", "mm_token_type_ids", "token_type_ids"):
        if field in inputs:
            extra = torch.full_like(tokens, int(field == "attention_mask"))
            inputs[field] = torch.cat([inputs[field], extra], dim=1)
    positions = torch.arange(start - 1, start + tokens.numel() - 1, device=device)
    logits = model(**inputs.to(device), use_cache=False, logits_to_keep=positions).logits
    assert logits.shape[:2] == tokens.shape
    return F.cross_entropy(logits[0].float(), tokens[0].to(device))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "records", "data-root", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.00012)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if min(a.epochs, a.accumulation) < 1:
        raise ValueError("epochs and accumulation must be positive")
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    if world > 1:
        dist.init_process_group("nccl")
    torch.manual_seed(a.seed)
    rows = [json.loads(s) for s in a.records.read_text().splitlines() if s]
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["partition"] in ("fit", "select") and r["kind"] in ("human", "teacher")
        assert r["instruction"].strip() and r["command"].strip()
        assert digest(a.data_root / r["source"]) == r["source_sha256"]
    human = [r for r in rows if r["partition"] == "fit" and r["kind"] == "human"]
    synthetic = [r for r in rows if r["partition"] == "fit" and r["kind"] == "teacher"]
    valid = [r for r in rows if r["partition"] == "select"]
    assert human and valid and all(r["kind"] == "human" for r in valid)
    processor = AutoProcessor.from_pretrained(a.base, local_files_only=True)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(a.base, dtype=torch.bfloat16,
                 attn_implementation="sdpa", local_files_only=True).to(device)
    layers = base.model.language_model.layers
    slots = range(max(0, len(layers) - 8), len(layers))
    names = []
    for i in slots:
        head = f"model.language_model.layers.{i}."
        names.append(head + "mlp.down_proj")
        if getattr(layers[i], "self_attn", None) is not None:
            names.extend(head + "self_attn." + k for k in ("q_proj", "k_proj", "v_proj", "o_proj"))
    known = dict(base.named_modules())
    assert all(n in known for n in names)
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0., target_modules=names, bias="none"))
    params = [x for x in model.parameters() if x.requires_grad]
    assert params and all("lora_" in n for n,x in model.named_parameters() if x.requires_grad)
    ddp = DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False) if world > 1 else model
    optim = torch.optim.AdamW(params, lr=a.lr, weight_decay=.01)
    if rank == 0:
        a.out.mkdir(parents=True, exist_ok=True)
    def validate():
        model.eval()
        totals = torch.zeros(2, dtype=torch.float64, device=device)
        with torch.no_grad():
            for r in valid[rank::world]:
                totals[0] += loss_on(model, processor, device, r, a.data_root)
                totals[1] += 1
        if world > 1:
            dist.all_reduce(totals)
        assert int(totals[1]) == len(valid)
        return float(totals[0] / totals[1])
    history = [{"epoch": 0, "validation_nll": validate()}]
    best = history[0]["validation_nll"]
    chosen = 0
    updates = 0
    for epoch in range(1, a.epochs + 1):
        rng = random.Random(a.seed + epoch)
        training = human + rng.sample(synthetic, min(len(human), len(synthetic)))
        rng.shuffle(training)
        needed = (-len(training)) % world
        training += [training[i % len(training)] for i in range(needed)]
        shard = training[rank::world]
        model.train()
        nll = torch.zeros(2, dtype=torch.float64, device=device)
        for step, r in enumerate(shard):
            group = (step // a.accumulation) * a.accumulation
            width = min(a.accumulation, len(shard) - group)
            end = step + 1 == group + width
            ctx = ddp.no_sync() if world > 1 and not end else contextlib.nullcontext()
            with ctx:
                loss = loss_on(ddp, processor, device, r, a.data_root)
                if not torch.isfinite(loss):
                    raise FloatingPointError(r["id"])
                (loss / width).backward()
            nll[0] += loss.detach()
            nll[1] += 1
            if end:
                torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                optim.step()
                optim.zero_grad(set_to_none=True)
                updates += 1
        if world > 1:
            dist.all_reduce(nll)
        val = validate()
        history.append(dict(epoch=epoch, train_nll=float(nll[0]/nll[1]), validation_nll=val))
        if val < best:
            best, chosen = val, epoch
            if rank == 0:
                model.save_pretrained(a.out / "selected", save_embedding_layers=False)
        if rank == 0:
            print(history[-1], flush=True)
            receipt = dict(history=history, selected_epoch=chosen, trainable=sum(v.numel() for v in params),
                           selected_perplexity=math.exp(best), updates=updates, seed=a.seed, world=world,
                           human_fit=len(human), teacher_fit=len(synthetic), human_validation=len(valid),
                           records_sha256=digest(a.records), targets=names)
            tmp = a.out / "training.tmp"
            tmp.write_text(json.dumps(receipt, indent=2))
            tmp.replace(a.out / "training.json")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
