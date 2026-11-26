import hashlib
import json

import pytest
import torch

from experiments.qwen35.data import BridgeDataset, collate_bridge
from training.paper_utils import sha256


def fixture(tmp_path):
    tokens, features, regions = [tmp_path / x for x in ('tokens', 'features', 'regions')]
    for folder in (tokens, features, regions):
        folder.mkdir()
    records = [dict(id=str(i), instruction=f'Instruction {i}', source_sha256=f'source{i}',
                    target_sha256=f'target{i}') for i in (1, 2)]
    manifest = tmp_path / 'train.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in records))
    hashes = {'train': sha256(manifest)}
    (tokens / 'metadata.json').write_text(json.dumps({'manifest_sha256': hashes}))
    (features / 'identity.json').write_text(json.dumps({'manifest_sha256': hashes, 'source_only': True}))
    (regions / 'metadata.json').write_text(json.dumps({'manifest_sha256': hashes, 'not_for_table1_background_metrics': True}))
    files = {}
    for i, r in enumerate(records):
        torch.save({'source': torch.zeros(9, dtype=torch.int16), 'target': torch.ones(9, dtype=torch.int16),
                    'source_sha256': r['source_sha256'], 'target_sha256': r['target_sha256']}, tokens / (r['id'] + '.pt'))
        path = features / (r['id'] + '.pt')
        torch.save({'source_sha256': r['source_sha256'],
                    'instruction_sha256': hashlib.sha256(r['instruction'].encode()).hexdigest(),
                    'readout': torch.ones(24), 'visual': torch.ones(9, 24), 'text': torch.ones(2 + 2 * i, 24)}, path)
        torch.save({'source_sha256': r['source_sha256'], 'target_sha256': r['target_sha256'],
                    'region': torch.zeros(9)}, regions / (r['id'] + '.pt'))
        files[r['id']] = sha256(path)
    (features / 'FEATURES_COMPLETE.json').write_text(json.dumps({'sha256': files, 'count': 2}))
    return BridgeDataset(manifest, tokens, features, regions)


def test_feature_alignment_and_padding(tmp_path):
    data = fixture(tmp_path)
    batch = collate_bridge([data[0], data[1]])
    assert batch[0] == ['1', '2']
    assert batch[6].shape == (2, 128, 24)
    assert torch.equal(batch[7].sum(1), torch.tensor([2, 4]))
    assert batch[6][0, 2:].sum() == 0
    assert batch[9].shape == (2, 9)
    assert torch.allclose(batch[9].sum(1), torch.ones(2))


def test_changed_feature_bytes_are_rejected(tmp_path):
    data = fixture(tmp_path)
    path = data.features / '1.pt'
    item = torch.load(path, weights_only=True)
    item['readout'].add_(1)
    torch.save(item, path)
    with pytest.raises(ValueError, match='feature bytes changed'):
        data[0]
