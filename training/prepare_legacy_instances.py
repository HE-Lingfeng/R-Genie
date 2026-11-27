"""Associate COCO instance polygons with localized removals and replacements in legacy fit pairs."""

import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

from training.prepare_merged_regions import grid, verified_image

ARCHIVE_SHA = "031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009"
MEMBER = "annotations/instances_train2014.json"
REMOVAL = re.compile(
    r"\b(remove|erase|delete|eliminate|wipe out|take (?:it |them )?away|get rid of|make .{0,50}?disappear)\b",
    re.IGNORECASE,
)
REPLACEMENT = re.compile(
    r"\b(replace|substitute|swap|(?:turn|change|transform) .{0,70}? (?:into|to))\b", re.IGNORECASE
)
THRESHOLDS = (13, 32)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_rows(root, manifests):
    outer = json.loads((root / "split-lock.json").read_bytes())["manifests"]
    inner = json.loads((manifests / "split-lock.json").read_bytes())
    path = manifests / "fit.jsonl"
    if (
        inner["parent_train_sha256"] != outer["train.jsonl"]
        or inner["heldout_test_sha256"] != outer["test.jsonl"]
        or sha(path) != inner["splits"]["fit"]["sha256"]
    ):
        raise ValueError("Expected the locked fit partition")
    rows = [
        r
        for r in map(json.loads, path.open())
        if r.get("supervision") == "paired-change-only"
        and r["id"].startswith("coco-")
        and r["source_group"] == "coco:" + r["id"][5:]
    ]
    if len({r["source_group"] for r in rows}) != len(rows):
        raise ValueError("Duplicate legacy source")
    return rows, inner["splits"]["fit"]["sha256"]


def operation(row):
    text = row.get("original_referring_instruction") or row["instruction"]
    if re.search(r"\b(and|then)\s+(add|put|place|give|insert)\b", text, re.IGNORECASE):
        return None
    remove, replace = bool(REMOVAL.search(text)), bool(REPLACEMENT.search(text))
    if remove != replace:
        return "remove" if remove else "replace"


def official_mask(annotation, size):
    w, h = size
    rle = coco_mask.merge(coco_mask.frPyObjects(annotation["segmentation"], h, w))
    binary = Image.fromarray(coco_mask.decode(rle)).resize((512, 512), Image.Resampling.NEAREST)
    return np.asarray(binary, dtype=bool)


def check_source(source, original_path):
    with Image.open(original_path) as original:
        if original.size != source.size:
            return None
        vectors = [
            np.asarray(image.convert("RGB").resize((64, 64), Image.Resampling.BOX), np.float64).ravel()
            for image in (source, original)
        ]
    if any(np.std(v) < 1.0 for v in vectors):
        return None
    mae = float(np.abs(vectors[0] - vectors[1]).mean())
    correlation = float(np.corrcoef(*vectors)[0, 1])
    if mae > 3 or correlation < 0.995:
        return None
    return {"official_jpeg_sha256": sha(original_path), "rgb_mae_8bit": mae, "correlation": correlation}


def match(delta, proposals):
    agreement = []
    for threshold in THRESHOLDS:
        change = delta >= threshold
        mass = int(change.sum())
        if mass < 64:
            return None
        ranked = []
        for ann, mask, area in proposals:
            shared = int(np.count_nonzero(mask & change))
            union = mass + area - shared
            ranked.append((shared / union, shared / area, shared / mass, ann["id"]))
        ranked.sort(reverse=True)
        iou, coverage, purity, ann_id = ranked[0]
        gap = iou - (ranked[1][0] if len(ranked) > 1 else 0)
        if iou < 0.45 or coverage < 0.55 or purity < 0.65 or gap < 0.12:
            return None
        agreement.append(
            {
                "annotation_id": ann_id,
                "threshold": threshold,
                "iou": iou,
                "coverage": coverage,
                "purity": purity,
                "gap": gap,
            }
        )
    return agreement if agreement[0]["annotation_id"] == agreement[1]["annotation_id"] else None


def main(args):
    rows, manifest_sha = fit_rows(args.merged, args.manifests)
    if sha(args.coco_zip) != ARCHIVE_SHA:
        raise ValueError("Expected the official 2014 Train/Val annotations archive")
    with zipfile.ZipFile(args.coco_zip) as archive, archive.open(MEMBER) as stream:
        coco = json.load(stream)
    ids = {int(r["id"][5:]) for r in rows}
    info = {r["id"]: r for r in coco["images"] if r["id"] in ids}
    categories = {r["id"]: r["name"] for r in coco["categories"]}
    by_id = defaultdict(list)
    for a in coco["annotations"]:
        if (
            a["image_id"] in ids
            and not a["iscrowd"]
            and isinstance(a["segmentation"], list)
            and a["segmentation"]
        ):
            by_id[a["image_id"]].append(a)
    del coco
    stats = Counter()
    report, targets = [], []
    for r in rows:
        i = int(r["id"][5:])
        op = operation(r)
        if op is None:
            stats["other_or_uncertain_operation"] += 1
            continue
        if i not in info or not by_id[i]:
            stats["missing_instances"] += 1
            continue
        source = verified_image(args.merged, r, "source", "RGB")
        edited = verified_image(args.merged, r, "target", "RGB")
        image = info[i]
        if (
            source.size != (image["width"], image["height"])
            or edited.size != source.size
            or image["file_name"] != f"COCO_train2014_{i:012d}.jpg"
        ):
            stats["different_canvas_or_filename"] += 1
            continue
        pixels = [
            np.asarray(im.resize((512, 512), Image.Resampling.BICUBIC), dtype=np.int16)
            for im in (source, edited)
        ]
        delta = np.abs(pixels[0] - pixels[1]).max(-1)
        proposals = []
        for ann in by_id[i]:
            mask = official_mask(ann, source.size)
            area = int(mask.sum())
            if area >= 64:
                proposals.append((ann, mask, area))
        decision = match(delta, proposals) if proposals else None
        if not decision:
            stats["insufficient_agreement"] += 1
            continue
        alignment = check_source(source, args.coco_images / image["file_name"])
        if alignment is None:
            stats["different_source_colors_or_geometry"] += 1
            continue
        ann, mask, _ = next(p for p in proposals if p[0]["id"] == decision[0]["annotation_id"])
        category = categories[ann["category_id"]]
        if category in {"bowl", "cup", "wine glass", "bottle"}:
            # A change to all the food or liquid can overlap the entire vessel polygon.
            text = (r.get("original_referring_instruction") or r["instruction"]).casefold()
            before_destination = re.split(r"\b(?:with|into)\b", text, maxsplit=1)[0]
            if not re.search(
                r"\b(bowl|dish|plate|cup|mug|glass|bottle|flask|vessel|container)\b", before_destination
            ):
                stats["ambiguous_container_or_contents"] += 1
                continue
        report.append(
            {
                "id": r["id"],
                "source_sha256": r["source_sha256"],
                "edited_sha256": r["target_sha256"],
                "operation": op,
                "annotation_id": ann["id"],
                "category": category,
                "source_check": alignment,
                "evidence": decision,
            }
        )
        targets.append(grid(mask))
        stats["accepted"] += 1
    if not report:
        raise ValueError("No localized edit matched a COCO instance")
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "fit.npz"
    with tempfile.NamedTemporaryFile(dir=args.output, suffix=".npz", delete=False) as handle:
        tmp = Path(handle.name)
    try:
        np.savez_compressed(
            tmp,
            fit_sha256=manifest_sha,
            coco_archive_sha256=ARCHIVE_SHA,
            ids=np.asarray([r["id"] for r in report]),
            annotation_ids=np.asarray([r["annotation_id"] for r in report]),
            source=np.stack(targets),
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    result = {
        "fit_sha256": manifest_sha,
        "coco_archive_sha256": ARCHIVE_SHA,
        "member": MEMBER,
        "preparation_sha256": sha(Path(__file__)),
        "regions_sha256": sha(path),
        "counts": dict(stats),
        "note": "Native COCO object shapes; assignment to a particular edit inferred using paired fit images.",
        "samples": report,
    }
    with tempfile.NamedTemporaryFile("w", dir=args.output, delete=False) as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
        tmp = Path(stream.name)
    os.replace(tmp, args.output / "matches.json")
    print(dict(stats), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("merged", "manifests", "coco-zip", "coco-images", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    main(p.parse_args())
