"""Torch-only latent intention interface for the modern diffusion environment.

The module names/shapes match the Show-o bridge so localization can be warm
started, without importing that package's legacy Transformers implementation.
"""
import math

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


class CrossBlock(nn.Module):
    def __init__(self, dimension, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(dimension), nn.LayerNorm(dimension)
        self.ff = nn.Sequential(nn.Linear(dimension, 4*dimension), nn.GELU(), nn.Linear(4*dimension, dimension))

    def forward(self, query, memory):
        query = self.norm1(query+self.attn(query, memory, memory, need_weights=False)[0])
        return self.norm2(query+self.ff(query))


class GroundedConditioner(nn.Module):
    def __init__(self, hidden=2560, dimension=256, heads=8, depth=4):
        super().__init__()
        self.dimension = dimension
        self.triplet = nn.Linear(hidden, 3*dimension)
        nn.init.orthogonal_(self.triplet.weight)
        nn.init.zeros_(self.triplet.bias)
        self.visual = nn.Linear(hidden, dimension)
        self.hrm = nn.ModuleList([CrossBlock(dimension, heads) for _ in range(depth)])
        self.query = nn.Linear(2*dimension, dimension)
        self.key, self.value = nn.Linear(dimension, dimension), nn.Linear(dimension, dimension)
        self.fusion = nn.Sequential(nn.LayerNorm(2*dimension), nn.Linear(2*dimension, dimension))
        self.region_bias = nn.Linear(2*dimension, 1)
        nn.init.zeros_(self.region_bias.weight)
        nn.init.constant_(self.region_bias.bias, -2)

    def forward(self, readout, visual, native_prior=None, native_prior_strength=1.):
        answer, reason, edit = self.triplet(readout).chunk(3, -1)
        visual = self.visual(visual)
        reason = reason[:, None]
        for layer in self.hrm:
            reason = layer(reason, visual.mean(1, keepdim=True))
        anchor = torch.cat([edit, answer], -1)
        logits = torch.einsum('bd,bnd->bn', self.query(anchor), self.key(visual))/math.sqrt(self.dimension)
        if native_prior is not None:
            if native_prior.shape != logits.shape:
                raise ValueError('Native prior and visual grid differ')
            normalized = native_prior.float()/native_prior.float().sum(1, keepdim=True).clamp_min(1e-8)
            logits = logits.float()+native_prior_strength*(normalized.clamp_min(1e-8)*logits.size(1)).log()
        prior = logits.softmax(-1)
        local = torch.einsum('bn,bnd->bd', prior.to(visual.dtype), self.value(visual))
        return self.fusion(torch.cat([reason[:, 0], local], -1)), prior, logits+self.region_bias(anchor)


class HybridAlignment(nn.Module):
    def __init__(self, hidden=2560, dimension=256, temperature=.07):
        super().__init__()
        self.temperature = temperature
        for name in ('global_visual', 'global_text', 'token_visual', 'token_text'):
            setattr(self, name, nn.Linear(hidden, dimension))

    def nce(self, image, text):
        image, text = F.normalize(image.float(), dim=-1), F.normalize(text.float(), dim=-1)
        if dist.is_available() and dist.is_initialized():
            from torch.distributed.nn.functional import all_gather
            image, text = torch.cat(all_gather(image)), torch.cat(all_gather(text))
        if len(image) < 2:
            raise ValueError('At least two distributed pairs are needed for alignment')
        logits = image@text.T/self.temperature
        labels = torch.arange(len(image), device=image.device)
        return (F.cross_entropy(logits, labels)+F.cross_entropy(logits.T, labels))/2

    def forward(self, visual, text, valid):
        weights = valid.to(text.dtype).unsqueeze(-1)
        pooled = (text*weights).sum(1)/weights.sum(1).clamp_min(1)
        global_loss = self.nce(self.global_visual(visual.mean(1)), self.global_text(pooled))
        visual, text = self.token_visual(visual), self.token_text(text)
        attended = (text@visual.transpose(1, 2)/math.sqrt(text.size(-1))).softmax(-1)@visual
        return (global_loss+self.nce((attended*weights).sum(1)/weights.sum(1).clamp_min(1),
                                     (text*weights).sum(1)/weights.sum(1).clamp_min(1)))/2


def region_loss(prior, logits, region):
    region = region.float().clamp(0, 1)
    use = region.sum(1) > 0
    if not use.any():
        return logits.sum()*0
    prior, logits, region = prior[use], logits[use], region[use]
    target = region/region.sum(1, keepdim=True)
    kl = (target*(target.clamp_min(1e-8).log()-prior.float().clamp_min(1e-8).log())).sum(1)
    bce = F.binary_cross_entropy_with_logits(logits.float(), region, reduction='none').mean(1)
    probability = logits.float().sigmoid()
    dice = 1-(2*(probability*region).sum(1)+1)/(probability.sum(1)+region.sum(1)+1)
    return (.25*kl+bce+dice).mean()


def predicted_mask(probability, threshold=.35, min_ratio=.01, max_ratio=.6):
    count = probability.size(1)
    sizes = (probability >= threshold).sum(1).clamp(max(1, math.ceil(count*min_ratio)), math.ceil(count*max_ratio))
    return probability.argsort(1, descending=True).argsort(1) < sizes[:, None]
