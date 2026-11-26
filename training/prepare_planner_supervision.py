"""Create source/request/command training records from REditBench's referring annotations."""
import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare(implicit, referring, data, heldout, teacher, validation_count):
    rows = read_jsonl(implicit)
    commands = json.loads(referring.read_text())
    held = [row for path in heldout for row in read_jsonl(path)]
    held_ids = {row["id"] for row in held}
    held_images = {row[key] for row in held for key in ("source_sha256", "target_sha256")}
    ids = {row["id"] for row in rows}
    if len(ids) != len(rows) or ids & held_ids:
        raise ValueError("Training IDs repeat or overlap the supplied held-out manifests")
    annotated = sorted(ids & commands.keys(), key=lambda k: hashlib.sha256(("RGenie/planner-select/"+k).encode()).digest())
    if not 0 < validation_count < len(annotated):
        raise ValueError("Reserve at least one human example for training and one for checkpoint selection")
    reserved = set(annotated[:validation_count])
    output = []
    for row in rows:
        image_id = row["id"]
        for key in ("source", "target"):
            digest = hashlib.sha256((data/row[key]).read_bytes()).hexdigest()
            if digest != row[key+"_sha256"] or digest in held_images:
                raise ValueError(f"Changed or held-out {key} for {image_id}")
        if image_id in commands:
            kind, command = "human", commands[image_id]["instruction"]
        elif teacher is not None:
            rec = json.loads((teacher/(image_id+".json")).read_text())
            for key in ("id","source_sha256","instruction"):
                if rec[key] != row[key]:
                    raise ValueError(f"Mismatched teacher {key} for {image_id}")
            if rec["training_reference_sha256"] != row["target_sha256"]:
                raise ValueError(f"Mismatched teacher reference for {image_id}")
            kind, command = "teacher", rec["command"]
        else:
            continue
        if not command.strip():
            raise ValueError(f"Empty command for {image_id}")
        item = {key:row[key] for key in ("id","source","source_sha256","instruction")}
        item.update(kind=kind, partition="select" if image_id in reserved else "fit", command=command.strip(),
                    training_reference_sha256=row["target_sha256"] if kind=="teacher" else None)
        output.append(item)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("implicit","referring","data-root","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--heldout-manifest",type=Path,action="append",default=[])
    parser.add_argument("--teacher-records",type=Path,help="Optional directory of offline, training-only teacher records")
    parser.add_argument("--validation-count",type=int,default=24)
    a=parser.parse_args()
    rows=prepare(a.implicit,a.referring,a.data_root,a.heldout_manifest,a.teacher_records,a.validation_count)
    payload="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows)
    if a.output.exists() and a.output.read_text()!=payload:
        raise ValueError("Existing training manifest differs from the prepared records")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.output.with_suffix(".tmp");tmp.write_text(payload);tmp.replace(a.output)
    print(len(rows),"examples,",sum(r["partition"]=="select" for r in rows),"for checkpoint selection")


if __name__=="__main__":main()
