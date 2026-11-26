"""Frozen, source-only Qwen3.5 features for the train/validation latent bridge."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers
from PIL import Image, ImageOps
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from experiments.qwen35.fetch_assets import verify
from training.paper_utils import read_records, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--manifests', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit', type=int, help='Bounded source-only feature extraction smoke')
    a = p.parse_args()
    output, model_path = Path(a.output), Path(a.model)
    output.mkdir(parents=True, exist_ok=True)
    origin = json.loads((model_path / 'origin.json').read_text())
    for entry in origin['files']:
        if not verify(model_path / entry['path'], entry):
            raise ValueError(f'Model asset differs from official pinned object: {entry["path"]}')
    manifests = Path(a.manifests)
    records = read_records(manifests / 'train.jsonl') + read_records(manifests / 'validation.jsonl')
    identity = {'model_origin': origin, 'transformers': transformers.__version__, 'torch': torch.__version__,
                'source_only': True, 'generated_reasoning_text': False,
                'preprocessing': 'EXIF_RGB_bicubic_square_512', 'visual_grid': [16, 16],
                'readout': 'last_assistant_prefix_hidden_state',
                'text_features': 'non_image_token_hidden_states_last_128',
                'max_text_tokens': 128,
                'source_sha256': sha256(__file__),
                'manifest_sha256': {s: sha256(manifests / (s + '.jsonl')) for s in ('train', 'validation')}}
    if (output / 'identity.json').exists():
        if json.loads((output / 'identity.json').read_text()) != identity:
            raise ValueError('Feature cache identity changed')
    else:
        (output / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    device = torch.device(a.device)
    torch.cuda.set_device(device)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).to(device).eval().requires_grad_(False)
    # Feature extraction calls the multimodal base; the vocabulary head is unused.
    if model.lm_head.weight is not model.get_input_embeddings().weight:
        model.lm_head.to('cpu')
    selected = records[:a.limit] if a.limit else records
    for index, record in enumerate(selected):
        dest = output / (record['id'] + '.pt')
        instruction_sha = hashlib.sha256(record['instruction'].encode()).hexdigest()
        if dest.exists():
            item = torch.load(dest, map_location='cpu', weights_only=True)
            if item['source_sha256'] != record['source_sha256'] or item['instruction_sha256'] != instruction_sha:
                raise ValueError('Cached source/instruction mismatch')
            continue
        source = Path(a.data) / record['source']
        if sha256(source) != record['source_sha256']:
            raise ValueError('Source image changed since audit')
        with Image.open(source) as im:
            image = ImageOps.exif_transpose(im).convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                                                {'type': 'text', 'text': record['instruction']}]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                                tokenize=True, return_dict=True, return_tensors='pt').to(device)
        with torch.inference_mode():
            result = model.model(**inputs, use_cache=False, return_dict=True)
            hidden = result.last_hidden_state[0]
            image_positions = inputs['input_ids'][0] == model.config.image_token_id
            if image_positions.sum().item() != 256:
                raise ValueError(f'Expected 16x16 native visual grid, got {image_positions.sum().item()} tokens')
            text_positions = ~image_positions & inputs['attention_mask'][0].bool()
            item = {'visual': hidden[image_positions].to(torch.bfloat16).cpu(),
                    'readout': hidden[-1].to(torch.bfloat16).cpu(),
                    'text': hidden[text_positions][-128:].to(torch.bfloat16).cpu(),
                    'input_ids': inputs['input_ids'][0].cpu(),
                    'source_sha256': record['source_sha256'], 'instruction_sha256': instruction_sha}
        torch.save(item, dest.with_suffix('.tmp'))
        dest.with_suffix('.tmp').replace(dest)
        print(f'Frozen Qwen features {index + 1}/{len(selected)}, id={record["id"]}', flush=True)
    if not a.limit:
        hashes = {r['id']: sha256(output / (r['id'] + '.pt')) for r in records}
        (output / 'FEATURES_COMPLETE.json').write_text(json.dumps({'count': len(records), 'sha256': hashes}, indent=2) + '\n')
    print('Feature extraction finished', flush=True)


if __name__ == '__main__':
    main()
