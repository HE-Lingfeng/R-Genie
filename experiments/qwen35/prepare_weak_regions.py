"""Paired-change supervision for training; never an annotated benchmark mask."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.paper_utils import image_tensor, read_records, sha256


def change_grid(source, target, threshold=0.05):
    # Input [C,256,256] in [-1,1]; threshold is on [0,1] mean RGB distance.
    delta = (source - target).abs().mean(0) / 2
    changed = (delta > threshold).float()
    return changed.reshape(16, 16, 16, 16).mean(dim=(1, 3)).flatten()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--manifests', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--threshold', type=float, default=0.05)
    a = p.parse_args()
    root, manifests, output = Path(a.data), Path(a.manifests), Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    records = read_records(manifests / 'train.jsonl') + read_records(manifests / 'validation.jsonl')
    identity = {'evidence_type': 'paired_pixel_change_proxy', 'not_annotated_ground_truth': True,
                'not_for_table1_background_metrics': True, 'threshold': a.threshold,
                'preprocessing': 'EXIF_RGB_bicubic_square_256', 'grid': [16, 16],
                'source_sha256': sha256(__file__),
                'manifest_sha256': {s: sha256(manifests / (s + '.jsonl')) for s in ('train', 'validation')}}
    areas = []
    for r in records:
        for key in ('source', 'target'):
            if sha256(root / r[key]) != r[key + '_sha256']:
                raise ValueError('Paired image changed since audit')
        grid = change_grid(image_tensor(root / r['source']), image_tensor(root / r['target']), a.threshold)
        torch.save({'region': grid, 'source_sha256': r['source_sha256'], 'target_sha256': r['target_sha256']},
                    output / (r['id'] + '.pt'))
        areas.append(grid.mean().item())
    identity.update(count=len(records), mean_change_area=float(np.mean(areas)),
                    median_change_area=float(np.median(areas)), near_zero_change_count=sum(x < 0.001 for x in areas))
    (output / 'metadata.json').write_text(json.dumps(identity, indent=2) + '\n')
    print(json.dumps(identity, indent=2))


if __name__ == '__main__':
    main()
