import copy

import pytest
import torch
from transformers import PhiConfig

from models.paper_rgenie import PaperRGenie, attention_bias
from models.phi import PhiForCausalLM
from training.paper_utils import validation_masks


SPECIAL = dict(mmu=5, soi=6, eoi=7, t2i=8, pad=0)


def tiny(checkpointing=True):
    torch.manual_seed(10)
    config = PhiConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                       num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                       partial_rotary_factor=0.5, resid_pdrop=0, embd_pdrop=0, attention_dropout=0)
    config._attn_implementation = 'sdpa'
    return PaperRGenie(PhiForCausalLM(config), visual_offset=40, codebook_size=16,
                       mask_id=63, dimension=8, heads=2, depth=2, adapter_width=8,
                       lora_rank=2, lora_alpha=4, lora_dropout=0,
                       gradient_checkpointing=checkpointing)


def inputs():
    source = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 2, 1]])
    target = source.roll(1, dims=1)
    text = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    return source, target, text, text != 0


def test_reconstruction_reaches_reasoning_but_never_updates_base():
    model = tiny()
    source, target, text, valid = inputs()
    before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    result = model(source, target, text, valid, SPECIAL, compute_alignment=False, mask_ratio=0.5)
    result['loss'].backward()
    assert model.llm.model.layers[0].self_attn.v_proj.lora_b.grad.norm() > 0
    assert model.conditioner.triplet.weight.grad.norm() > 0
    assert model.visual_adapter[0].weight.grad.norm() > 0
    assert model.condition_projections[0].weight.grad.norm() > 0
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name in before:
            assert parameter.grad is None
            assert torch.equal(before[name], parameter)


def test_denoiser_disables_lora_even_with_nonzero_adapter_weights():
    model = tiny().eval()
    source, _, text, valid = inputs()
    condition = torch.randn(2, 8)
    codes = source + 40
    reference = model.denoise(codes, text, valid, SPECIAL, condition).detach()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_b' in name:
                parameter.normal_(std=2)
    changed = model.denoise(codes, text, valid, SPECIAL, condition)
    torch.testing.assert_close(reference, changed)
    changed.sum().backward()
    for name, parameter in model.named_parameters():
        if 'lora_' in name:
            assert parameter.grad is None


def test_checkpoint_recomputation_matches_eager_gradients():
    eager = tiny(False)
    with torch.no_grad():
        for name, p in eager.named_parameters():
            if 'lora_b' in name:
                p.normal_(std=0.01)
    saved = copy.deepcopy(eager)
    saved.gradient_checkpointing = True
    args = inputs()
    for model in (eager, saved):
        result = model(*args, SPECIAL, generator=torch.Generator().manual_seed(33), mask_ratio=0.5)
        result['loss'].backward()
    for (name, a), (_, b) in zip(eager.named_parameters(), saved.named_parameters()):
        if a.requires_grad:
            assert a.grad is not None, name
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-6, msg=name)




def test_attention_mask_hides_padding_and_future_text():
    valid = torch.tensor([[False, True, True, True, True]])
    mask = attention_bias(valid, image_suffix_start=3)[0, 0]
    assert mask[1, 2] < -1e20
    assert mask[3, 4] == 0
    assert mask[4, 0] < -1e20
    assert mask[0, 0] == 0


def test_alignment_rejects_single_pair_without_negatives():
    model = tiny()
    args = tuple(x[:1] for x in inputs())
    with pytest.raises(ValueError, match='>=2'):
        model(*args, SPECIAL)


def test_validation_masks_do_not_depend_on_batching_or_sharding():
    together = validation_masks(['a', 'b', 'c'], 256, 12345, 'cpu')
    reordered = validation_masks(['c', 'a'], 256, 12345, 'cpu')
    assert torch.equal(together[0], reordered[1])
    assert torch.equal(together[2], reordered[0])
    assert together.any(1).all()


def test_explicit_validation_corruption_ignores_random_generator():
    model = tiny().eval()
    args = inputs()
    mask = validation_masks(['a', 'b'], 9, 12345, 'cpu')
    one = model(*args, SPECIAL, compute_alignment=False, mask_positions=mask,
                generator=torch.Generator().manual_seed(1))
    two = model(*args, SPECIAL, compute_alignment=False, mask_positions=mask,
                generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(one['reconstruction'], two['reconstruction'])
