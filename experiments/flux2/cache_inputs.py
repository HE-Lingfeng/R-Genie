"""Frozen native FLUX text/VAE cache for training and validation; no test access."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, FlowMatchEulerDiscreteScheduler
from PIL import Image, ImageOps
from transformers import AutoTokenizer, Qwen3ForCausalLM

from experiments.qwen35.fetch_assets import verify
from training.paper_utils import read_records, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--manifests', type=Path, default=Path('artifacts/reditbench-v1'))
    p.add_argument('--output', required=True)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--wait-for-download', action='store_true')
    a = p.parse_args()
    model_root, output = Path(a.model), Path(a.output)
    origin = model_root/'origin.json'
    while not origin.exists():
        if not a.wait_for_download:
            raise FileNotFoundError('Model assets incomplete')
        time.sleep(15)
    assets = json.loads(origin.read_text())
    assert assets == json.loads(Path('experiments/flux2/assets.json').read_text())
    for item in assets['files']:
        assert verify(model_root/item['path'], item), item['path']
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(assets=assets, resolution=512, source_sha256=sha256(__file__),
        text_preprocessing='native_Qwen3_chat_template_and_layers_9_18_27_at_512_tokens',
        image_preprocessing='EXIF_RGB_square_bicubic_512', vae_precision='FP32',
        text_precision='BF16', deterministic_VAE_posterior_mode=True,
        source_only_prompt=True, cached_targets_for_training_only=True,
        manifest_sha256={s:sha256(a.manifests/(s+'.jsonl')) for s in ('train','validation')})
    if (output/'identity.json').exists():
        assert json.loads((output/'identity.json').read_text()) == identity
    else:
        (output/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    torch.set_num_threads(2)
    device = torch.device(f'cuda:{a.device}')
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_root/'tokenizer', local_files_only=True)
    text_encoder = Qwen3ForCausalLM.from_pretrained(model_root/'text_encoder', torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    vae = AutoencoderKLFlux2.from_pretrained(model_root/'vae', torch_dtype=torch.float32,
        local_files_only=True).to(device).eval().requires_grad_(False)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_root/'scheduler', local_files_only=True)
    pipe = Flux2KleinPipeline(scheduler, vae, text_encoder, tokenizer, transformer=None)
    records = read_records(a.manifests/'train.jsonl')+read_records(a.manifests/'validation.jsonl')
    started = time.monotonic()
    with torch.inference_mode():
        empty = pipe._get_qwen3_prompt_embeds(text_encoder, tokenizer, '', device=device)
        if not (output/'empty-prompt.pt').exists():
            torch.save(empty[0].cpu(), output/'empty-prompt.pt')
        for index, r in enumerate(records):
            dest = output/(r['id']+'.pt')
            if dest.exists():
                item = torch.load(dest, weights_only=True)
                assert item['source_sha256'] == r['source_sha256'] and item['target_sha256'] == r['target_sha256']
                assert item['instruction_sha256'] == hashlib.sha256(r['instruction'].encode()).hexdigest()
                continue
            prompt = pipe._get_qwen3_prompt_embeds(text_encoder, tokenizer, r['instruction'], device=device)
            latents = []
            for key in ('source', 'target'):
                file = Path(a.data)/r[key]
                assert sha256(file) == r[key+'_sha256']
                with Image.open(file) as image:
                    image = ImageOps.exif_transpose(image).convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
                pixels = pipe.image_processor.preprocess(image, height=512, width=512).to(device, dtype=torch.float32)
                latent = pipe._encode_vae_image(pixels, generator=None)
                latents.append(pipe._pack_latents(latent)[0].bfloat16().cpu())
            assert latents[0].shape == (1024, 128) and prompt.shape == (1, 512, 7680)
            item = dict(source=latents[0], target=latents[1], prompt=prompt[0].cpu(),
                source_sha256=r['source_sha256'], target_sha256=r['target_sha256'],
                instruction_sha256=hashlib.sha256(r['instruction'].encode()).hexdigest())
            assert all(torch.isfinite(item[k]).all() for k in ('source','target','prompt'))
            torch.save(item, dest.with_suffix('.tmp'))
            dest.with_suffix('.tmp').replace(dest)
            if index == 0 or (index+1)%10 == 0:
                print('Cached native FLUX',index+1,len(records),time.monotonic()-started,flush=True)
    complete = dict(count=len(records), sha256={r['id']:sha256(output/(r['id']+'.pt')) for r in records},
        empty_prompt_sha256=sha256(output/'empty-prompt.pt'))
    (output/'CACHE_COMPLETE.json').write_text(json.dumps(complete, indent=2)+'\n')
    print('CACHE_COMPLETE',len(records),time.monotonic()-started,flush=True)


if __name__ == '__main__':
    main()
