"""Frozen Qwen3.5 latent interface, HRM/RAB and frozen Show-o masked decoding.

The paired-change auxiliary signal is weak training supervision, not a benchmark
ground-truth mask. No explicit rationale/answer labels or generated CoT are used.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.paper_rgenie import HybridAlignment, IntentionConditioner, PaperRGenie


class GroundedConditioner(IntentionConditioner):
    def __init__(self, hidden, dimension=256, heads=8, depth=4):
        super().__init__(hidden, dimension, heads, depth)
        self.region_bias = nn.Linear(dimension * 2, 1)
        nn.init.zeros_(self.region_bias.weight)
        nn.init.constant_(self.region_bias.bias, -2)

    def forward(self, readout, visual, native_prior=None, native_prior_strength=1.0):
        answer, reason, edit = self.triplet(readout).chunk(3, dim=-1)
        visual = self.visual(visual)
        reason = reason[:, None]
        global_visual = visual.mean(1, keepdim=True)
        for layer in self.hrm:
            reason = layer(reason, global_visual)
        anchor = torch.cat([edit, answer], dim=-1)
        query = self.query(anchor)
        logits = torch.einsum('bd,bnd->bn', query, self.key(visual)) / math.sqrt(self.dimension)
        if native_prior is not None:
            if native_prior.shape != logits.shape:
                raise ValueError('Native Qwen attention must match the spatial feature grid')
            # Native attention was computed from source + instruction only. A
            # uniform prior contributes zero; learned RAB logits can refine it.
            normalized = native_prior.float() / native_prior.float().sum(1, keepdim=True).clamp_min(1e-8)
            logits = logits.float() + native_prior_strength * (normalized.clamp_min(1e-8) * logits.size(1)).log()
        prior = logits.softmax(-1)
        local = torch.einsum('bn,bnd->bd', prior.to(visual.dtype), self.value(visual))
        condition = self.fusion(torch.cat([reason[:, 0], local], dim=-1))
        # A query-dependent offset calibrates region area without changing the
        # normalized RAB attention that pools the local visual condition.
        region_logits = logits + self.region_bias(anchor)
        return condition, prior, region_logits


class QwenBridgeEditor(PaperRGenie):
    def __init__(self, llm, feature_hidden=2560, dimension=256, heads=8, depth=4,
                 adapter_width=256, gradient_checkpointing=True, visual_offset=50305,
                 codebook_size=8192, mask_id=58497, region_weight=1, edited_token_weight=4,
                 native_prior_strength=1.0):
        # The inference/training bridge receives frozen Qwen source features. Phi
        # is retained solely as the frozen pretrained discrete image denoiser.
        nn.Module.__init__(self)
        self.llm = llm.requires_grad_(False)
        self.visual_offset, self.codebook_size, self.mask_id = visual_offset, codebook_size, mask_id
        self.gradient_checkpointing = gradient_checkpointing
        self.region_weight, self.edited_token_weight = region_weight, edited_token_weight
        self.native_prior_strength = native_prior_strength
        self.feature_norm = nn.LayerNorm(feature_hidden)
        self.visual_adapter = nn.Sequential(nn.Linear(feature_hidden, adapter_width), nn.GELU(),
                                            nn.Linear(adapter_width, feature_hidden))
        self.conditioner = GroundedConditioner(feature_hidden, dimension, heads, depth)
        self.alignment = HybridAlignment(feature_hidden, dimension)
        self.condition_projections = nn.ModuleList([
            nn.Linear(dimension, llm.config.hidden_size, bias=False) for _ in llm.model.layers])
        for layer in self.condition_projections:
            nn.init.normal_(layer.weight, std=0.001)

    def intention(self, readout, visual, native_prior=None):
        readout, visual = self.feature_norm(readout), self.feature_norm(visual)
        visual = visual + self.visual_adapter(visual)
        condition, prior, region_logits = self.conditioner(readout, visual, native_prior, self.native_prior_strength)
        return condition, prior, region_logits, visual

    @staticmethod
    def region_loss(prior, logits, weak_region):
        weak_region = weak_region.float().clamp(0, 1)
        area = weak_region.sum(1)
        usable = area > 0
        if not usable.any():
            return logits.sum() * 0
        prior, logits, weak_region = prior[usable], logits[usable], weak_region[usable]
        target = weak_region / weak_region.sum(1, keepdim=True)
        kl = (target * (target.clamp_min(1e-8).log() - prior.float().clamp_min(1e-8).log())).sum(1)
        bce = F.binary_cross_entropy_with_logits(logits.float(), weak_region, reduction='none').mean(1)
        probability = logits.float().sigmoid()
        dice = 1 - (2 * (probability * weak_region).sum(1) + 1) / (probability.sum(1) + weak_region.sum(1) + 1)
        return (0.25 * kl + bce + dice).mean()

    def forward(self, target_codes, text_ids, text_valid, special, readout, visual,
                text_features, text_feature_valid, weak_region, compute_alignment=True,
                mask_positions=None, generator=None, native_prior=None):
        condition, prior, region_logits, visual = self.intention(readout, visual, native_prior)
        batch, count = target_codes.shape
        if mask_positions is None:
            fraction = torch.cos(torch.rand(batch, device=target_codes.device, generator=generator) * math.pi / 2)
            numbers = (fraction * count).round().long().clamp(1, count)
            order = torch.rand(batch, count, device=target_codes.device, generator=generator).argsort(1).argsort(1)
            masked = order < numbers[:, None]
        else:
            if mask_positions.dtype != torch.bool or mask_positions.shape != target_codes.shape:
                raise ValueError('Invalid explicit validation corruption')
            masked = mask_positions.to(target_codes.device)
            numbers = masked.sum(1)
            if (numbers < 1).any():
                raise ValueError('A validation sample has no masked tokens')
        corrupted = (target_codes + self.visual_offset).masked_fill(masked, self.mask_id)
        labels = (target_codes + self.visual_offset).masked_fill(~masked, -100)
        logits = self.denoise(corrupted, text_ids, text_valid, special, condition)
        token_ce = F.cross_entropy(logits.transpose(1, 2), labels, reduction='none')
        reconstruction = token_ce.sum(1) / numbers
        weights = (1 + self.edited_token_weight * weak_region.float()) * masked
        edited_reconstruction = (token_ce * weights).sum(1) / weights.sum(1).clamp_min(1)
        ha = self.alignment(visual, self.feature_norm(text_features), text_feature_valid) if compute_alignment else logits.new_zeros(())
        localization = self.region_loss(prior, region_logits, weak_region)
        keep = 1 - numbers.float() / count
        loss = (keep * edited_reconstruction).mean() + (1 - keep).mean() * ha + self.region_weight * localization
        return {'loss': loss, 'reconstruction': reconstruction.mean(),
                'edited_reconstruction': edited_reconstruction.mean(), 'alignment': ha,
                'localization': localization, 'region_probability': region_logits.float().sigmoid(),
                'attention': prior}

    @staticmethod
    def predicted_mask(probability, threshold=0.35, min_ratio=0.01, max_ratio=0.6):
        if not 0 < min_ratio <= max_ratio <= 1 or not 0 < threshold < 1:
            raise ValueError('Invalid adaptive region configuration')
        batch, count = probability.shape
        minimum, maximum = max(1, math.ceil(count * min_ratio)), max(1, math.ceil(count * max_ratio))
        sizes = (probability >= threshold).sum(1).clamp(minimum, maximum)
        order = probability.argsort(dim=1, descending=True).argsort(dim=1)
        return order < sizes[:, None]
