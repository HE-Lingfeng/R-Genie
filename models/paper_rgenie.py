"""Two-pass latent intention -> HRM/RAB -> frozen Show-o denoiser.

Uses explicit per-pass adapter state and per-layer conditioning arguments so
activation-checkpoint recomputation cannot accidentally enable denoising LoRA.
This is a new implementation of the supplied paper, not a released checkpoint.
"""
import contextlib
import contextvars
import math

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


_REASONING_PASS = contextvars.ContextVar('rgenie_reasoning_pass', default=False)


@contextlib.contextmanager
def reasoning_adapters(enabled):
    token = _REASONING_PASS.set(enabled)
    try:
        yield
    finally:
        _REASONING_PASS.reset(token)


class PassLoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.base(x)
        if _REASONING_PASS.get():
            y = y + F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b) * self.scale
        return y


class CrossBlock(nn.Module):
    def __init__(self, dimension, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(dimension), nn.LayerNorm(dimension)
        self.ff = nn.Sequential(nn.Linear(dimension, dimension * 4), nn.GELU(),
                                nn.Linear(dimension * 4, dimension))

    def forward(self, query, memory):
        query = self.norm1(query + self.attn(query, memory, memory, need_weights=False)[0])
        return self.norm2(query + self.ff(query))


class IntentionConditioner(nn.Module):
    def __init__(self, hidden, dimension=256, heads=8, depth=4):
        super().__init__()
        self.triplet = nn.Linear(hidden, dimension * 3)
        nn.init.orthogonal_(self.triplet.weight)
        nn.init.zeros_(self.triplet.bias)
        self.visual = nn.Linear(hidden, dimension)
        self.hrm = nn.ModuleList([CrossBlock(dimension, heads) for _ in range(depth)])
        self.query = nn.Linear(dimension * 2, dimension)
        self.key, self.value = nn.Linear(dimension, dimension), nn.Linear(dimension, dimension)
        self.fusion = nn.Sequential(nn.LayerNorm(dimension * 2), nn.Linear(dimension * 2, dimension))
        self.dimension = dimension

    def forward(self, readout, visual):
        answer, reason, edit = self.triplet(readout).chunk(3, dim=-1)
        visual = self.visual(visual)
        global_visual = visual.mean(1, keepdim=True)
        reason = reason[:, None]
        for layer in self.hrm:
            reason = layer(reason, global_visual)
        query = self.query(torch.cat([edit, answer], dim=-1))
        attention = torch.einsum('bd,bnd->bn', query, self.key(visual)) / math.sqrt(self.dimension)
        attention = attention.softmax(-1)
        local = torch.einsum('bn,bnd->bd', attention, self.value(visual))
        condition = self.fusion(torch.cat([reason[:, 0], local], dim=-1))
        return condition, attention, (answer, reason[:, 0], edit)


def _gather_with_grad(x):
    if dist.is_available() and dist.is_initialized():
        from torch.distributed.nn.functional import all_gather
        return torch.cat(all_gather(x), dim=0)
    return x


class HybridAlignment(nn.Module):
    def __init__(self, hidden, dimension=256, temperature=0.07):
        super().__init__()
        self.global_visual = nn.Linear(hidden, dimension)
        self.global_text = nn.Linear(hidden, dimension)
        self.token_visual = nn.Linear(hidden, dimension)
        self.token_text = nn.Linear(hidden, dimension)
        self.temperature = temperature

    def nce(self, visual, text, gather):
        visual, text = F.normalize(visual.float(), dim=-1), F.normalize(text.float(), dim=-1)
        if gather:
            visual, text = _gather_with_grad(visual), _gather_with_grad(text)
        if len(visual) < 2:
            raise ValueError('Hybrid alignment needs >=2 image-instruction pairs per distributed microbatch')
        logits = visual @ text.T / self.temperature
        labels = torch.arange(len(logits), device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    def forward(self, visual, text, valid, gather=True):
        weights = valid.to(text.dtype).unsqueeze(-1)
        pooled_text = (text * weights).sum(1) / weights.sum(1).clamp_min(1)
        global_loss = self.nce(self.global_visual(visual.mean(1)), self.global_text(pooled_text), gather)
        visual, text = self.token_visual(visual), self.token_text(text)
        attn = (text @ visual.transpose(1, 2) / math.sqrt(text.size(-1))).softmax(-1)
        attended = attn @ visual
        token_visual = (attended * weights).sum(1) / weights.sum(1).clamp_min(1)
        token_text = (text * weights).sum(1) / weights.sum(1).clamp_min(1)
        return (global_loss + self.nce(token_visual, token_text, gather)) / 2


def attention_bias(valid, image_prefix=0, image_suffix_start=None):
    """Causal text, bidirectional image blocks; padded keys are never visible."""
    batch, length = valid.shape
    allowed = torch.ones(length, length, dtype=torch.bool, device=valid.device).tril()
    if image_prefix:
        allowed[:image_prefix, :image_prefix] = True
    if image_suffix_start is not None:
        allowed[image_suffix_start:, image_suffix_start:] = True
    allowed = allowed[None] & valid[:, None, :].bool()
    # A padded query has a harmless self edge, avoiding all-masked softmax rows.
    eye = torch.eye(length, dtype=torch.bool, device=valid.device)[None]
    allowed = allowed | (eye & ~valid[:, :, None].bool())
    return torch.zeros(batch, 1, length, length, device=valid.device).masked_fill_(
        ~allowed[:, None], torch.finfo(torch.float32).min)


class PaperRGenie(nn.Module):
    def __init__(self, llm, visual_offset=50305, codebook_size=8192, mask_id=58497,
                 dimension=256, heads=8, depth=4, adapter_width=320,
                 lora_rank=16, lora_alpha=32, lora_dropout=0.05, gradient_checkpointing=True):
        super().__init__()
        self.llm = llm.requires_grad_(False)
        hidden = llm.config.hidden_size
        self.visual_offset, self.codebook_size, self.mask_id = visual_offset, codebook_size, mask_id
        self.gradient_checkpointing = gradient_checkpointing
        for layer in self.llm.model.layers:
            for name in ('q_proj', 'k_proj', 'v_proj', 'dense'):
                base = getattr(layer.self_attn, name)
                setattr(layer.self_attn, name, PassLoRALinear(base, lora_rank, lora_alpha, lora_dropout))
        self.reason_token = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.visual_adapter = nn.Sequential(nn.Linear(hidden, adapter_width), nn.GELU(),
                                            nn.Linear(adapter_width, hidden))
        self.conditioner = IntentionConditioner(hidden, dimension, heads, depth)
        self.alignment = HybridAlignment(hidden, dimension)
        self.condition_projections = nn.ModuleList([
            nn.Linear(dimension, hidden, bias=False) for _ in self.llm.model.layers])
        for p in self.condition_projections:
            nn.init.normal_(p.weight, std=0.001)

    def run_layers(self, embeddings, mask, adapters, condition=None):
        hidden = self.llm.model.embed_dropout(embeddings)
        positions = torch.arange(hidden.size(1), device=hidden.device)[None]
        for index, layer in enumerate(self.llm.model.layers):
            bias = None if condition is None else self.condition_projections[index](condition)

            def block(h, c, layer=layer):
                with reasoning_adapters(adapters):
                    return layer(h, attention_mask=mask.to(h.dtype), position_ids=positions,
                                 use_cache=False, condition_bias=c)[0]

            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(block, hidden, bias, use_reentrant=False)
            else:
                hidden = block(hidden, bias)
        return self.llm.model.final_layernorm(hidden)

    def reason(self, source_codes, text_ids, text_valid, special):
        batch, count = source_codes.shape
        embed = self.llm.get_input_embeddings()
        visual = embed(source_codes + self.visual_offset)
        visual = visual + self.visual_adapter(visual)
        def tag(name):
            return embed(torch.full((batch, 1), special[name], device=text_ids.device, dtype=torch.long))
        prefix = torch.cat([tag('mmu'), tag('soi'), visual, tag('eoi')], dim=1)
        sequence = torch.cat([prefix, embed(text_ids), self.reason_token.expand(batch, -1, -1)], dim=1)
        valid = torch.cat([torch.ones(batch, prefix.size(1), device=text_ids.device, dtype=torch.bool),
                           text_valid.bool(), torch.ones(batch, 1, device=text_ids.device, dtype=torch.bool)], dim=1)
        hidden = self.run_layers(sequence, attention_bias(valid, image_prefix=prefix.size(1)), adapters=True)
        visual_hidden = hidden[:, 2:2 + count]
        text_hidden = hidden[:, prefix.size(1):-1]
        condition, prior, triplet = self.conditioner(hidden[:, -1], visual_hidden)
        return condition, prior, triplet, visual_hidden, text_hidden

    def denoise(self, codes, text_ids, text_valid, special, condition):
        batch = len(codes)
        # Show-o pretrained format: [t2i, BOS, text, EOS, soi, visual tokens, eoi].
        prefix = F.pad(text_ids, (1, 0), value=special['pad'])
        prefix_valid = F.pad(text_valid.bool(), (1, 0), value=False)
        task_positions = (~text_valid.bool()).sum(1)
        rows = torch.arange(batch, device=codes.device)
        prefix[rows, task_positions] = special['t2i']
        prefix_valid[rows, task_positions] = True
        sequence = torch.cat([prefix,
                              torch.full((batch, 1), special['soi'], device=codes.device, dtype=torch.long),
                              codes, torch.full((batch, 1), special['eoi'], device=codes.device, dtype=torch.long)], dim=1)
        valid = torch.cat([prefix_valid,
                           torch.ones(batch, codes.size(1) + 2, device=codes.device, dtype=torch.bool)], dim=1)
        start = prefix.size(1) + 1
        hidden = self.run_layers(self.llm.get_input_embeddings()(sequence),
                                 attention_bias(valid, image_suffix_start=start - 1),
                                 adapters=False, condition=condition)
        return self.llm.lm_head(hidden[:, start:start + codes.size(1)]).float()

    def forward(self, source_codes, target_codes, text_ids, text_valid, special,
                generator=None, compute_alignment=True, mask_ratio=None, mask_positions=None):
        condition, prior, triplet, visual, text = self.reason(source_codes, text_ids, text_valid, special)
        batch, count = target_codes.shape
        if mask_positions is not None:
            if mask_positions.shape != target_codes.shape or mask_positions.dtype != torch.bool:
                raise ValueError('Explicit corruption masks must be Boolean and match target token shape')
            masked = mask_positions.to(target_codes.device)
            mask_count = masked.sum(1)
            if (mask_count < 1).any():
                raise ValueError('Every sample must contain at least one supervised masked position')
        else:
            if mask_ratio is None:
                time = torch.rand(batch, device=target_codes.device, generator=generator)
                mask_ratio = torch.cos(time * math.pi / 2).clamp(min=1 / count, max=1)
            else:
                mask_ratio = torch.as_tensor(mask_ratio, device=target_codes.device).expand(batch)
            mask_count = (mask_ratio * count).round().long().clamp(1, count)
            order = torch.rand(batch, count, device=target_codes.device, generator=generator).argsort(1).argsort(1)
            masked = order < mask_count[:, None]
        corrupted = (target_codes + self.visual_offset).masked_fill(masked, self.mask_id)
        logits = self.denoise(corrupted, text_ids, text_valid, special, condition)
        labels = (target_codes + self.visual_offset).masked_fill(~masked, -100)
        token_loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction='none')
        rec = token_loss.sum(1) / mask_count
        ha = self.alignment(visual, text, text_valid) if compute_alignment else rec.new_zeros(())
        # Eq. (10): alpha_bar is the probability of retaining a token.
        keep = 1 - mask_count.float() / count
        loss = (keep * rec).mean() + (1 - keep).mean() * ha
        return {'loss': loss, 'reconstruction': rec.mean(), 'alignment': ha,
                'mask_fraction': (1 - keep).mean(), 'attention': prior}


    def trainable_state_dict(self):
        names = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {name: value.detach().cpu() for name, value in self.state_dict().items() if name in names}
