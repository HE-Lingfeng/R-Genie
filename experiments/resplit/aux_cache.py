"""Encode eligible explicit optimization pairs and fixed development plans for FLUX continuation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from accelerate import cpu_offload
from diffusers import AutoencoderKLFlux2,Flux2KleinPipeline
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from transformers import AutoTokenizer,Qwen3ForCausalLM

from training.train_paper import atomic_save
from experiments.resplit.spatial import rgb
from training.paper_utils import read_records, sha256


FRAME = ('Edit the reference photograph according to this instruction: {plan}. '
         'For any object removal, reconstruct the unobstructed background naturally. '
         'Preserve all other objects and the camera view.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rank',type=int,required=True,choices=(0,1))
    parser.add_argument('--model',type=Path,default=Path('hf_model/FLUX.2-klein-base-4B'))
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--manifests',type=Path,default=Path('artifacts/reditbench-resplit-v2'))
    parser.add_argument('--validation-prompts',type=Path,required=True,
                        help='Directory of cached validation prompt embeddings, one <id>.pt per source')
    parser.add_argument('--output',type=Path,default=Path('artifacts/aux-cache'))
    a=parser.parse_args()
    torch.set_num_threads(2);device=torch.device('cuda',a.rank);torch.cuda.set_device(device)
    root=a.model
    folder=a.output;folder.mkdir(parents=True, exist_ok=True)
    opt=a.manifests/'aux-explicit-train.jsonl'
    lock=json.loads((a.manifests/'aux-protocol.json').read_text())
    assert sha256(opt)==lock['input_manifest']
    train=list(map(json.loads,opt.read_text().splitlines()))
    val=read_records(a.manifests/'validation.jsonl')
    assert {r['id'] for r in train}.isdisjoint(r['id'] for r in val+read_records(a.manifests/'test.jsonl'))
    encoder=Qwen3ForCausalLM.from_pretrained(root/'text_encoder',dtype=torch.bfloat16,attn_implementation='sdpa',
                                              local_files_only=True).eval().requires_grad_(False)
    cpu_offload(encoder,execution_device=device,offload_buffers=True)
    tok=AutoTokenizer.from_pretrained(root/'tokenizer',local_files_only=True)
    vae=AutoencoderKLFlux2.from_pretrained(root/'vae',dtype=torch.float32,local_files_only=True).to(device).eval()
    proc=Flux2ImageProcessor(vae_scale_factor=16)
    mu=vae.bn.running_mean.reshape(1,-1,1,1);std=(vae.bn.running_var.reshape(1,-1,1,1)+vae.config.batch_norm_eps).sqrt()
    def encode(im):
        pixels=proc.preprocess(im,height=512,width=512).to(device)
        grid=Flux2KleinPipeline._patchify_latents(vae.encode(pixels).latent_dist.mode())
        return Flux2KleinPipeline._pack_latents((grid-mu)/std)[0].bfloat16().cpu()
    with torch.inference_mode():
        for split,rows in [('train',train),('validation',val)]:
            out=folder/split;out.mkdir(exist_ok=True)
            for n,r in enumerate(rows[a.rank::2]):
                f=out/(r['id']+'.pt')
                if f.exists(): continue
                paths=[a.data/r[k] for k in ('source','target')]
                assert [sha256(x) for x in paths]==[r[k+'_sha256'] for k in ('source','target')]
                s,t=map(rgb,paths)
                delta=np.abs(np.asarray(s,dtype=np.int16)-np.asarray(t,dtype=np.int16)).max(-1)>12.75
                mask=torch.from_numpy(delta.reshape(32,16,32,16).mean((1,3)).astype('float32')).flatten()
                if split=='train':
                    prompt=Flux2KleinPipeline._get_qwen3_prompt_embeds(encoder,tok,FRAME.format(plan=r['plan']),device=device)[0].cpu()
                else:
                    raw=torch.load(a.validation_prompts/(r['id']+'.pt'),weights_only=True)
                    assert raw['source_sha256']==r['source_sha256']
                    prompt=raw['prompt']
                atomic_save(dict(source=encode(s),target=encode(t),prompt=prompt,region=mask,id=r['id']),f)
                if n%25==0:print(split,a.rank,n+1,flush=True)
            (out/f'complete-{a.rank}.json').write_text(json.dumps({'ids':[r['id'] for r in rows[a.rank::2]],'code':sha256(__file__)}))


if __name__=='__main__':
    main()
