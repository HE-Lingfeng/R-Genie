"""Fine-tune the diffusion LoRA on allowed explicit pairs to match plan-conditioned inference."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset,DataLoader,DistributedSampler

from experiments.flux2_lora.model import load_editor
from training.train_paper import atomic_save
from training.paper_utils import sha256


class Pairs(Dataset):
    def __init__(self,cache,split):
        self.path=cache/split
        self.ids=sum([json.loads((self.path/f'complete-{k}.json').read_text())['ids'] for k in range(2)],[])
        self.ids.sort(key=int)
        assert len(self.ids)==len(set(self.ids))==(343 if split=='train' else 100)

    def __len__(self):return len(self.ids)

    def __getitem__(self,i):
        return torch.load(self.path/(self.ids[i]+'.pt'),weights_only=True)


class Flow(nn.Module):
    def __init__(self,editor):
        super().__init__();self.editor=editor

    def forward(self,src,tgt,prompt,region,noise,time):
        xt=(1-time[:,None,None])*tgt+time[:,None,None]*noise
        pred=self.editor.vector_field(xt.to(src.dtype),src,prompt,time.flatten().to(src.dtype),None)
        error=(pred.float()-(noise.float()-tgt.float())).square().mean(-1)
        weights=1+4*region.float()
        return (error*weights).sum(1)/weights.sum(1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=Path('artifacts/aux-cache'))
    parser.add_argument('--manifests',type=Path,default=Path('artifacts/reditbench-resplit-v2'))
    parser.add_argument('--output',type=Path,default=Path('runs/aux-training'))
    args=parser.parse_args()
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    assert world==2
    torch.cuda.set_device(rank);torch.set_num_threads(2);device=torch.device('cuda',rank)
    dist.init_process_group('nccl')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42)
    saved=torch.load(args.parent,map_location='cpu',weights_only=False)
    parent_hash=sha256(args.parent)
    ident=saved['identity']
    assert all(sha256(file)==checksum for file,checksum in ident['source_sha256'].items())
    editor=load_editor(OmegaConf.create(ident['config']))
    assert editor.generation_lora_targets==ident['generation_lora_targets']
    assert {k for k,v in editor.named_parameters() if v.requires_grad}==set(saved['model'])
    editor.load_state_dict(saved['model'],strict=False)
    for k,v in editor.named_parameters():v.requires_grad_('.lora_A.' in k or '.lora_B.' in k)
    params=[v for v in editor.parameters() if v.requires_grad]
    assert sum(v.numel() for v in params)==9338880
    flow=Flow(editor).to(device)
    model=DistributedDataParallel(flow,device_ids=[rank],broadcast_buffers=False)
    opt=torch.optim.AdamW(params,lr=2e-5,betas=(.9,.95),weight_decay=.01)
    train=Pairs(args.cache,'train');val=Pairs(args.cache,'validation')
    sampler=DistributedSampler(train,num_replicas=world,rank=rank,shuffle=True,seed=42)
    loader=DataLoader(train,batch_size=3,sampler=sampler,num_workers=2,pin_memory=True)
    vloader=DataLoader(val,batch_size=3,sampler=range(rank,len(val),world),num_workers=1,pin_memory=True)
    rng=torch.Generator(device=device).manual_seed(2026+rank)
    def batch_loss(batch,validation=False):
        src,tgt,prompt,region=[batch[k].to(device,non_blocking=True) for k in ('source','target','prompt','region')]
        if validation:
            ns=[];ts=[]
            for i in batch['id']:
                g=torch.Generator(device=device).manual_seed(45371+int(i))
                ns.append(torch.randn(tgt.shape[1:],device=device,generator=g,dtype=tgt.dtype))
                ts.append(torch.randn((),device=device,generator=g).sigmoid())
            noise=torch.stack(ns);time=torch.stack(ts)
        else:
            noise=torch.randn(tgt.shape,device=device,dtype=tgt.dtype,generator=rng)
            time=torch.randn(len(tgt),device=device,generator=rng).sigmoid()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return model(src,tgt,prompt,region,noise,time)
    best=float('inf');history=[]
    for epoch in range(4):
        if epoch:
            model.train();sampler.set_epoch(epoch)
            for batch in loader:
                opt.zero_grad(set_to_none=True)
                batch_loss(batch).mean().backward()
                torch.nn.utils.clip_grad_norm_(params,1.)
                opt.step()
        model.eval()
        sums=torch.zeros(2,device=device,dtype=torch.float64)
        with torch.no_grad():
            for batch in vloader:
                loss=batch_loss(batch,True)
                sums[0]+=loss.double().sum();sums[1]+=len(loss)
        dist.all_reduce(sums)
        assert int(sums[1])==100
        score=float(sums[0]/sums[1])
        if rank==0:
            history.append(score)
            print('epoch',epoch,'val',score,flush=True)
            if score<best:
                weights={k:v.detach().cpu() for k,v in editor.named_parameters() if v.requires_grad}
                atomic_save(dict(weights=weights,parent_sha256=parent_hash,epoch=epoch,
                    val_weighted_flow_MSE=score,train_sha256=sha256(args.manifests/'aux-explicit-train.jsonl'),
                    protocol_sha256=sha256(args.manifests/'aux-protocol.json'),code=sha256(__file__)),out/'selected.pt')
            if epoch==3:
                (out/'COMPLETE.json').write_text(json.dumps(dict(validation=history,selected_epoch=min(range(4),key=history.__getitem__),
                    checkpoint_sha256=sha256(out/'selected.pt'),world_size=world,train_unique=343,
                    train_steps=3*len(loader),peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30),indent=2))
        best=min(best,score);dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':main()
