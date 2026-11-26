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


FRAMING = ('Consider the source image and the implicit image-editing instruction below. '
           'Identify the visible target that the instruction refers to, and the requested visual operation. Instruction: ')
CUE = 'The intended visible target object is'


def encode_pair_source(model, processor, source, instruction, device):
    with Image.open(source) as im:
        image = ImageOps.exif_transpose(im).convert('RGB').resize((512,512),Image.Resampling.BICUBIC)
    messages = [{'role':'user','content':[{'type':'image','image':image},
        {'type':'text','text':FRAMING+instruction}]}]
    inputs = processor.apply_chat_template(messages,add_generation_prompt=True,enable_thinking=False,
        tokenize=True,return_dict=True,return_tensors='pt')
    suffix = processor.tokenizer(CUE,add_special_tokens=False,return_tensors='pt').input_ids
    inputs['input_ids'] = torch.cat([inputs.input_ids,suffix],1)
    for key in ('attention_mask','mm_token_type_ids','token_type_ids'):
        if key in inputs:
            inputs[key] = torch.cat([inputs[key],torch.full_like(suffix,1 if key == 'attention_mask' else 0)],1)
    inputs = inputs.to(device)
    image_mask = inputs.input_ids[0] == model.config.image_token_id
    if image_mask.sum().item() != 256:
        raise ValueError('Expected the trained native 16x16 visual grid')
    captured_visual,captured_attention = [],[]
    def visual_hook(module,args,output):
        captured_visual.append(output.pooler_output.detach())
    def attention_hook(module,args,output):
        if output[1] is not None:
            score = output[1][0,:,-1,:][:,image_mask].float().detach()
            mass = score.sum(-1,keepdim=True)
            keep = mass[:,0] > 1e-4
            if keep.any():
                score,mass = score[keep],mass[keep]
            captured_attention.append((score/mass.clamp_min(1e-8)).mean(0))
    handles = [model.model.visual.register_forward_hook(visual_hook)]
    full = [layer.self_attn for layer in model.model.language_model.layers if getattr(layer,'self_attn',None) is not None]
    handles += [layer.register_forward_hook(attention_hook) for layer in full[-4:]]
    try:
        with torch.inference_mode():
            hidden = model.model(**inputs,use_cache=False,return_dict=True).last_hidden_state[0]
            if len(captured_visual) != 1 or len(captured_attention) != 4:
                raise RuntimeError('Native visual/attention extraction differs from the training interface')
            native = torch.stack(captured_attention).mean(0)
            native = native/native.sum().clamp_min(1e-8)
            text_mask = ~image_mask & inputs.attention_mask[0].bool()
            result = dict(input_ids=inputs.input_ids[0].cpu(), visual=captured_visual[0].bfloat16().cpu(),readout=hidden[-1].bfloat16().cpu(),
                text=hidden[text_mask][-128:].bfloat16().cpu(),native_prior=native.cpu(),
                source_sha256=sha256(source),instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest())
            if not all(torch.isfinite(result[k]).all() for k in ('visual','readout','text','native_prior')):
                raise FloatingPointError('Nonfinite native multimodal features')
    finally:
        for handle in handles:
            handle.remove()
    return result


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
    if origin != json.loads(Path('experiments/qwen35/assets.json').read_text()):
        raise ValueError('Expected the pinned Qwen3.5-4B training backbone')
    for entry in origin['files']:
        if not verify(model_path / entry['path'], entry):
            raise ValueError(f'Model asset differs from official pinned object: {entry["path"]}')
    manifests = Path(a.manifests)
    records = read_records(manifests / 'train.jsonl') + read_records(manifests / 'validation.jsonl')
    identity = {'model_origin': origin, 'transformers': transformers.__version__, 'torch': torch.__version__,
                'source_only': True, 'generated_reasoning_text': False,
                'preprocessing': 'EXIF_RGB_bicubic_square_512', 'visual_grid': [16, 16],
                'readout': 'fixed_target_cue_last_hidden_state',
                'visual_features': 'native_vision_pooler_output',
                'fixed_readout_cue': CUE, 'input_framing': FRAMING,
                'native_prior': 'last_four_full_attention_layers_mean_image_normalized_heads',
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
    model.model.language_model.config._attn_implementation = 'eager'
    selected = records[:a.limit] if a.limit else records
    for index, record in enumerate(selected):
        dest = output / (record['id'] + '.pt')
        instruction_sha = hashlib.sha256(record['instruction'].encode()).hexdigest()
        if dest.exists():
            item = torch.load(dest, map_location='cpu', weights_only=True)
            if ('native_prior' not in item or item['source_sha256'] != record['source_sha256']
                    or item['instruction_sha256'] != instruction_sha):
                raise ValueError('Cached source/instruction mismatch')
            continue
        source = Path(a.data) / record['source']
        if sha256(source) != record['source_sha256']:
            raise ValueError('Source image changed since audit')
        item = encode_pair_source(model, processor, source, record['instruction'], device)
        torch.save(item, dest.with_suffix('.tmp'))
        dest.with_suffix('.tmp').replace(dest)
        print(f'Frozen Qwen features {index + 1}/{len(selected)}, id={record["id"]}', flush=True)
    if not a.limit:
        hashes = {r['id']: sha256(output / (r['id'] + '.pt')) for r in records}
        (output / 'FEATURES_COMPLETE.json').write_text(json.dumps({'count': len(records), 'sha256': hashes}, indent=2) + '\n')
    print('Feature extraction finished', flush=True)


if __name__ == '__main__':
    main()
