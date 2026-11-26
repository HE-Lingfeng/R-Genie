import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.prepare_reditbench import prepare


def release(tmp_path):
    for folder in ('image', 'target'):
        (tmp_path / folder).mkdir()
    for i in range(1, 7):
        Image.new('RGB', (16, 16), (i * 20, 0, 0)).save(tmp_path / 'image' / f'COCO_{i:012d}.jpg')
        Image.new('RGB', (16, 16), (0, i * 20, 0)).save(tmp_path / 'target' / f'COCO_{i:012d}_edit.jpg')
    for name, ids in [('training_reasoning_sentences', range(1, 5)),
                      ('testing_reasoning_sentences', [5]), ('training_refering_sentences', [6])]:
        (tmp_path / (name + '.json')).write_text(json.dumps({str(i): {'sentences': f'Implicit instruction {i}'} for i in ids}))
    return tmp_path


def test_release_split_is_reproducible_and_keeps_test_held_out(tmp_path):
    root = release(tmp_path)
    report = prepare(root, tmp_path / 'out1', validation_size=1)
    prepare(root, tmp_path / 'out2', validation_size=1)
    assert report['counts'] == {'train': 3, 'validation': 1, 'test': 1}
    assert report['table1_comparable'] is False
    assert report['explicit_only_training_pairs'] == 1
    for name in ('train', 'validation', 'test'):
        assert (tmp_path / 'out1' / (name + '.jsonl')).read_bytes() == (tmp_path / 'out2' / (name + '.jsonl')).read_bytes()
    ids = {s: {json.loads(line)['id'] for line in (tmp_path / 'out1' / (s + '.jsonl')).read_text().splitlines()}
           for s in ('train', 'validation', 'test')}
    assert not (ids['train'] & ids['test'])
    assert not (ids['validation'] & ids['test'])
    assert '6' not in set.union(*ids.values())


def test_cross_split_duplicate_images_stop_preparation(tmp_path):
    root = release(tmp_path)
    (root / 'image' / 'COCO_000000000005.jpg').write_bytes((root / 'image' / 'COCO_000000000001.jpg').read_bytes())
    with pytest.raises(ValueError, match='Byte-identical'):
        prepare(root, tmp_path / 'out', validation_size=1)
