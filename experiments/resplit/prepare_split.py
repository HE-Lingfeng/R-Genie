"""Freeze a user-authorized holdout from the previously unseen implicit examples."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('artifacts/reditbench-v1'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/reditbench-resplit-v2'))
    args = parser.parse_args()
    splits = ('train', 'validation', 'test')
    originals = dict(zip(splits, [
        'c87ea9d10d7db96f98d089b6a6e856e237e409097132fbb38a377c57eba7f312',
        'a557325fcaa089a2053d7eefe6eafeb6502eb9b1f7ff6cdfad26502ff6700db1',
        '48db362696322eb9463ca0013d0ee356ed38279a9f0fe27c90b2073009b38c11']))
    rows = {}
    for split in splits:
        payload = (args.input / (split + '.jsonl')).read_bytes()
        if sha(payload) != originals[split]:
            raise ValueError('Input manifest differs from audited release: ' + split)
        rows[split] = [json.loads(line) for line in payload.splitlines()]
    all_rows = sum(rows.values(), [])
    for key in ('id', 'source_sha256', 'target_sha256'):
        if len({row[key] for row in all_rows}) != len(all_rows):
            raise ValueError('Duplicate ' + key)
    salt = 'REditBench/user-resplit/v2/42/'
    ordered = sorted(rows['test'], key=lambda row: sha((salt + row['id']).encode()))
    extra = ordered[220:]
    rows['train'] += extra
    rows['test'] = ordered[:220]
    args.output.mkdir(parents=True, exist_ok=True)
    audit = {'version': 'user-resplit-v2', 'selection': 'lowest SHA256(salt + canonical decimal ID) from the 258 previously unused implicit examples', 'salt': salt, 'old_manifests': originals, 'added_train_ids': [r['id'] for r in extra], 'splits': {}}
    for split, samples in rows.items():
        samples.sort(key=lambda r: int(r['id']))
        payload = ''.join(json.dumps(r, sort_keys=True, ensure_ascii=False) + '\n' for r in samples).encode()
        dest = args.output / (split + '.jsonl')
        if dest.exists() and dest.read_bytes() != payload:
            raise ValueError('Existing split differs: ' + str(dest))
        dest.write_bytes(payload)
        audit['splits'][split] = {'count': len(samples), 'sha256': sha(payload)}
    (args.output / 'split-lock.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit['splits']))


if __name__ == '__main__':
    main()
