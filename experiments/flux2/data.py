import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from experiments.qwen35.data import BridgeDataset, collate_bridge
from training.paper_utils import sha256


class FlowDataset(Dataset):
    def __init__(self, manifest, tokens, features, regions, flow_cache):
        self.bridge = BridgeDataset(manifest, tokens, features, regions)
        self.cache = Path(flow_cache)
        self.identity = json.loads((self.cache/'identity.json').read_text())
        self.complete = json.loads((self.cache/'CACHE_COMPLETE.json').read_text())
        if self.identity['manifest_sha256'][Path(manifest).stem] != sha256(manifest):
            raise ValueError('Native FLUX cache uses another dataset split')
        if sha256(self.cache/'empty-prompt.pt') != self.complete['empty_prompt_sha256']:
            raise ValueError('Empty prompt cache changed')
        self.empty_prompt = torch.load(self.cache/'empty-prompt.pt', weights_only=True)

    def __len__(self):
        return len(self.bridge)

    def __getitem__(self, index):
        base = self.bridge[index]
        file = self.cache/(base[0]+'.pt')
        if sha256(file) != self.complete['sha256'][base[0]]:
            raise ValueError('Native FLUX cached input bytes changed')
        item = torch.load(file, weights_only=True)
        record = self.bridge.base.records[index]
        if (item['source_sha256'] != record['source_sha256'] or item['target_sha256'] != record['target_sha256']
                or item['instruction_sha256'] != hashlib.sha256(base[1].encode()).hexdigest()):
            raise ValueError('Native FLUX cache source/target/instruction identity mismatch')
        return base, item['source'], item['target'], item['prompt']


def collate_flow(batch):
    bases, source, target, prompt = zip(*batch)
    bridge = collate_bridge(bases)
    return dict(ids=bridge[0], source=torch.stack(source), target=torch.stack(target),
        prompt_embeds=torch.stack(prompt), readout=bridge[4], visual=bridge[5],
        text_features=bridge[6], text_valid=bridge[7], weak_region=bridge[8], native_prior=bridge[9])
