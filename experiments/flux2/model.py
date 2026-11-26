"""Latent intention -> HRM/RAB -> additive conditioning of frozen FLUX.2.

This is a continuous-flow extension of R-Genie, not a Show-o reproduction. The
native Qwen3 text conditioning and image-reference stream remain pretrained.
Only the latent interface, alignment and per-block bias projections are updated.
"""
import contextvars
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from experiments.flux2.conditioning import GroundedConditioner, HybridAlignment, region_loss


_CONDITION = contextvars.ContextVar('flux_rgenie_condition', default=None)


def conditioned_checkpoint(module, *inputs):
    condition = _CONDITION.get()
    if condition is None:
        return checkpoint(module, *inputs, use_reentrant=False)

    def run(*values):
        token = _CONDITION.set(values[-1])
        try:
            return module(*values[:-1])
        finally:
            _CONDITION.reset(token)
    # The condition is an explicit checkpoint input; recomputation never reads
    # mutable state left over by another forward/microbatch.
    return checkpoint(run, *inputs, condition, use_reentrant=False)


class FluxIntentionEditor(nn.Module):
    def __init__(self, transformer, feature_hidden=2560, dimension=256, heads=8,
                 depth=4, adapter_width=256, gradient_checkpointing=True):
        super().__init__()
        self.transformer = transformer.requires_grad_(False)
        self.feature_norm = nn.LayerNorm(feature_hidden)
        self.visual_adapter = nn.Sequential(nn.Linear(feature_hidden, adapter_width), nn.GELU(),
                                            nn.Linear(adapter_width, feature_hidden))
        self.conditioner = GroundedConditioner(feature_hidden, dimension, heads, depth)
        self.alignment = HybridAlignment(feature_hidden, dimension)
        blocks = [*transformer.transformer_blocks, *transformer.single_transformer_blocks]
        self.condition_projections = nn.ModuleList([nn.Linear(dimension, transformer.inner_dim, bias=False) for _ in blocks])
        for layer in self.condition_projections:
            nn.init.zeros_(layer.weight)
        # Hook attention outputs rather than changing vendor source. Initial
        # zero projections preserve the exact pretrained vector field.
        self._hooks = []
        for i, block in enumerate(blocks):
            def hook(module, inputs, output, index=i):
                condition = _CONDITION.get()
                if condition is None:
                    return output
                bias = self.condition_projections[index](condition)[:, None]
                if isinstance(output, tuple):
                    return tuple(item+bias.to(item.dtype) for item in output)
                return output+bias.to(output.dtype)
            self._hooks.append(block.attn.register_forward_hook(hook))
        if gradient_checkpointing:
            transformer.enable_gradient_checkpointing(gradient_checkpointing_func=conditioned_checkpoint)

    def intention(self, readout, visual, native_prior=None):
        readout, visual = self.feature_norm(readout), self.feature_norm(visual)
        visual = visual+self.visual_adapter(visual)
        condition, prior, logits = self.conditioner(readout, visual, native_prior)
        return condition, prior, logits, visual

    def vector_field(self, latents, source, prompt_embeds, timestep, condition):
        batch, count, _ = latents.shape
        side = math.isqrt(count)
        if side*side != count or source.shape != latents.shape:
            raise ValueError('Expected aligned square source/target latent grids')
        from diffusers import Flux2KleinPipeline
        grid = latents.new_empty(batch, 1, side, side)
        ids = Flux2KleinPipeline._prepare_latent_ids(grid).to(latents.device)
        reference_ids = ids.clone()
        reference_ids[..., 0] = 10
        text_ids = Flux2KleinPipeline._prepare_text_ids(prompt_embeds).to(latents.device)
        token = _CONDITION.set(condition)
        try:
            return self.transformer(hidden_states=torch.cat([latents, source], 1),
                encoder_hidden_states=prompt_embeds, timestep=timestep, guidance=None,
                img_ids=torch.cat([ids, reference_ids], 1), txt_ids=text_ids, return_dict=False)[0][:, :count]
        finally:
            _CONDITION.reset(token)

    def forward(self, source, target, prompt_embeds, readout, visual, text_features,
                text_valid, weak_region, native_prior, noise=None, timestep=None,
                compute_alignment=True, condition_keep=None):
        condition, prior, logits, adapted_visual = self.intention(readout, visual, native_prior)
        if condition_keep is not None:
            condition = condition*condition_keep[:, None].to(condition.dtype)
        noise = torch.randn_like(target) if noise is None else noise
        if timestep is None:
            timestep = torch.randn(len(target), device=target.device).sigmoid()
        mixed = (1-timestep[:, None, None])*target+timestep[:, None, None]*noise
        prediction = self.vector_field(mixed, source, prompt_embeds, timestep, condition)
        errors = (prediction.float()-(noise-target).float()).square().mean(-1)
        side = math.isqrt(target.size(1))
        region = F.interpolate(weak_region.reshape(-1, 1, 16, 16).float(), (side, side), mode='bilinear', align_corners=False).flatten(1)
        weights = 1+4*region
        flow = ((errors*weights).sum(1)/weights.sum(1)).mean()
        localization = region_loss(prior, logits, weak_region)
        ha = self.alignment(adapted_visual, self.feature_norm(text_features), text_valid) if compute_alignment else flow.new_zeros(())
        return dict(loss=flow+localization+.05*ha, flow_MSE=errors.mean(), edited_flow_MSE=flow,
                    localization=localization, alignment=ha, region_probability=logits.float().sigmoid())

    def trainable_state_dict(self):
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {n: x.detach().cpu() for n, x in self.state_dict().items() if n in names}
