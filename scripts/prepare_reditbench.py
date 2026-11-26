"""Audit the supplied release without inventing instructions, masks or splits."""
import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def image_id(path):
    return str(int(Path(path).stem.removesuffix('_edit').split('_')[-1]))


def prepare(root, output, validation_size=100, seed=42, official_splits=None):
    root, output = Path(root).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    annotation_paths = {
        'implicit_train': root / 'training_reasoning_sentences.json',
        'implicit_test': root / 'testing_reasoning_sentences.json',
        'explicit_train': root / 'training_refering_sentences.json',
    }
    annotations = {
        name: {str(int(k)): v for k, v in json.loads(p.read_text()).items()}
        for name, p in annotation_paths.items()
    }
    sources = {image_id(p): p for p in (root / 'image').iterdir()
               if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}}
    targets = {image_id(p): p for p in (root / 'target').iterdir()
               if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}}
    if set(annotations['implicit_train']) & set(annotations['implicit_test']):
        raise ValueError('Overlapping released training and test IDs')
    matched = set(sources) & set(targets)
    train_ids = sorted(matched & set(annotations['implicit_train']), key=int)
    test_ids = sorted(matched & set(annotations['implicit_test']), key=int)
    if official_splits:
        split_ids = json.loads(Path(official_splits).read_text())
        if set(split_ids) != {'train', 'validation', 'test'}:
            raise ValueError('Split file must contain train, validation and test ID lists')
        split_ids = {k: [str(int(i)) for i in v] for k, v in split_ids.items()}
        if [len(split_ids[k]) for k in ('train', 'validation', 'test')] != [750, 100, 220]:
            raise ValueError('Paper split requires exactly 750 / 100 / 220 IDs')
        for split, ids in split_ids.items():
            if len(set(ids)) != len(ids):
                raise ValueError(f'Duplicate IDs in {split}')
            allowed = set(test_ids) if split == 'test' else set(train_ids)
            if not set(ids) <= allowed:
                raise ValueError(f'{split}: IDs lack implicit annotations/images or cross the released split')
        protocol = 'supplied_paper_split'
    else:
        if not 0 < validation_size < len(train_ids):
            raise ValueError('Validation size must leave nonempty optimization and validation sets')
        shuffled = train_ids.copy()
        random.Random(seed).shuffle(shuffled)
        validation = set(shuffled[:validation_size])
        split_ids = {'train': [i for i in train_ids if i not in validation],
                     'validation': [i for i in train_ids if i in validation], 'test': test_ids}
        protocol = 'provisional_release_split_not_table1'
    keys = list(split_ids)
    for n, a in enumerate(keys):
        for b in keys[n + 1:]:
            if set(split_ids[a]) & set(split_ids[b]):
                raise ValueError(f'Leakage between {a} and {b}')
    records, hashes = {}, {}
    for split, ids in split_ids.items():
        records[split] = []
        for i in ids:
            entry = annotations['implicit_test' if split == 'test' else 'implicit_train'][i]
            instruction = entry.get('sentences', '').strip()
            if not instruction:
                raise ValueError(f'Missing implicit instruction: {i}')
            with Image.open(sources[i]) as im:
                im.verify()
            with Image.open(targets[i]) as im:
                im.verify()
            source_sha, target_sha = digest(sources[i]), digest(targets[i])
            if source_sha in hashes and hashes[source_sha][0] != split:
                raise ValueError(f'Byte-identical image across splits: {i}, {hashes[source_sha]}')
            hashes[source_sha] = (split, i)
            records[split].append({'id': i, 'instruction': instruction,
                                   'source': str(sources[i].relative_to(root)),
                                   'target': str(targets[i].relative_to(root)),
                                   'source_sha256': source_sha, 'target_sha256': target_sha})
        text = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in records[split])
        (output / f'{split}.jsonl').write_text(text)
    all_annotations = set.union(*(set(d) for d in annotations.values()))
    report = {
        'protocol': protocol, 'table1_comparable': False,
        'dataset_root': str(root), 'split_seed': seed,
        'counts': {k: len(v) for k, v in records.items()},
        'source_images': len(sources), 'target_images': len(targets),
        'annotation_counts': {k: len(v) for k, v in annotations.items()},
        'annotation_sha256': {k: digest(v) for k, v in annotation_paths.items()},
        'manifest_sha256': {k: digest(output / f'{k}.jsonl') for k in records},
        'annotation_ids_without_pairs': {k: sorted(set(v) - matched, key=int) for k, v in annotations.items()},
        'paired_ids_without_annotations': sorted(matched - all_annotations, key=int),
        'explicit_only_training_pairs': len((matched & set(annotations['explicit_train'])) - set(train_ids)),
        'missing_for_table1': ['verified 750/100/220 split with implicit instructions',
                               'annotated edit masks', 'exact CLIP/AP/RISE-BS evaluation definitions/checkpoints'],
        'notes': ['Explicit-only examples are not silently substituted for implicit instructions.',
                  'No target-derived mask is accepted as ground truth for background metrics.',
                  'The released 258 test samples remain held out; they are not reduced to 220 arbitrarily.'],
    }
    if official_splits:
        report['missing_for_table1'].pop(0)
        report['official_split_sha256'] = digest(official_splits)
    (output / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--validation-size', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--official-splits')
    a = p.parse_args()
    print(json.dumps(prepare(a.data, a.output, a.validation_size, a.seed, a.official_splits), indent=2))
