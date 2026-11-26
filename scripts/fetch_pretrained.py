"""Download immutable official initialization revisions; never create random fallbacks."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


ASSETS = {
    'show-o': ('showlab/show-o', 'c2c6f548c00444432a97bc8fbbce7b2fd73fc9b5',
               '67ca6506dc5a0d47bed8f4aa957c9b23153df0d33540b188ac65115cad7eaea4'),
    'magvitv2': ('showlab/magvitv2', '5c3fa78f8b3523347c5cd1a4c97f3c4e96f33d5d',
                 '79656bf974d99c215465a61df5e93b97e03282b0941f62f6294719527873b778'),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='hf_model')
    p.add_argument('--only', choices=list(ASSETS), nargs='+', default=list(ASSETS))
    a = p.parse_args()
    for folder in a.only:
        repo, revision, expected = ASSETS[folder]
        dest = Path(a.root) / folder
        snapshot_download(repo_id=repo, revision=revision, local_dir=dest,
                          token=False, allow_patterns=['config.json', 'pytorch_model.safetensors'])
        h = hashlib.sha256()
        with (dest / 'pytorch_model.safetensors').open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        if h.hexdigest() != expected:
            raise ValueError(f'Weight SHA256 differs from official pinned LFS object: {repo}')
        (dest / 'origin.json').write_text(json.dumps({'repo_id': repo, 'revision': revision,
                                                     'weight_sha256': expected}, indent=2) + '\n')
        print(f'{repo}@{revision} -> {dest}', flush=True)


if __name__ == '__main__':
    main()
