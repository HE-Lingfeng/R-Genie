import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from training.paper_utils import CachedEditingDataset, sha256


class BridgeDataset(Dataset):
    def __init__(self, manifest, tokens, features, regions):
        self.base = CachedEditingDataset(manifest, tokens)
        self.features, self.regions = Path(features), Path(regions)
        self.feature_identity = json.loads((self.features / 'identity.json').read_text())
        self.feature_complete = json.loads((self.features / 'FEATURES_COMPLETE.json').read_text())
        self.region_identity = json.loads((self.regions / 'metadata.json').read_text())
        split = Path(manifest).stem
        for metadata in (self.feature_identity, self.region_identity):
            if metadata['manifest_sha256'][split] != sha256(manifest):
                raise ValueError('Qwen features/weak regions use a different split')
        if not self.feature_identity['source_only'] or not self.region_identity['not_for_table1_background_metrics']:
            raise ValueError('Unexpected feature/weak-label provenance')

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample_id, instruction, source, target = self.base[index]
        record = self.base.records[index]
        path = self.features / (sample_id + '.pt')
        if sha256(path) != self.feature_complete['sha256'][sample_id]:
            raise ValueError('Qwen feature bytes changed after sealing')
        feature = torch.load(path, map_location='cpu', weights_only=True)
        region = torch.load(self.regions / (sample_id + '.pt'), map_location='cpu', weights_only=True)
        if feature['source_sha256'] != record['source_sha256'] or region['source_sha256'] != record['source_sha256']:
            raise ValueError('Qwen source identity mismatch')
        if region['target_sha256'] != record['target_sha256']:
            raise ValueError('Weak region was generated from a different paired target')
        if feature['instruction_sha256'] != hashlib.sha256(instruction.encode()).hexdigest():
            raise ValueError('Qwen instruction identity mismatch')
        native = feature.get('native_prior')
        if native is None:
            if self.feature_identity.get('native_prior'):
                raise ValueError('Feature cache promises native attention but the sample has none')
            native = torch.full((feature['visual'].size(0),), 1 / feature['visual'].size(0))
        if (native.shape != (feature['visual'].size(0),) or not torch.isfinite(native).all()
                or (native < 0).any() or native.sum() <= 0):
            raise ValueError('Invalid native Qwen spatial attention')
        return (sample_id, instruction, source, target, feature['readout'], feature['visual'],
                feature['text'], region['region'], native.float() / native.sum())


def collate_bridge(batch):
    ids, instructions, source, target, readout, visual, text, region, native = zip(*batch)
    hidden = readout[0].numel()
    text_features = torch.zeros(len(batch), 128, hidden, dtype=text[0].dtype)
    text_valid = torch.zeros(len(batch), 128, dtype=torch.bool)
    for i, item in enumerate(text):
        n = min(128, len(item))
        text_features[i, :n] = item[-n:]
        text_valid[i, :n] = True
    return (list(ids), list(instructions), torch.stack(source), torch.stack(target), torch.stack(readout),
            torch.stack(visual), text_features, text_valid, torch.stack(region), torch.stack(native))
