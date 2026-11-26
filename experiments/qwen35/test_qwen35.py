import torch
from transformers import PhiConfig

from experiments.qwen35.model import GroundedConditioner, QwenBridgeEditor
from experiments.qwen35.prepare_weak_regions import change_grid
from models.phi import PhiForCausalLM


def test_change_grid_retains_spatial_coordinates():
    source = torch.zeros(3, 256, 256)
    target = source.clone()
    target[:, 16:32, 32:48] = 1
    grid = change_grid(source, target)
    assert grid[18] == 1
    assert grid.sum() == 1


def test_adaptive_mask_obeys_area_limits():
    p = torch.tensor([[0.9, 0.8, 0.1, 0.0], [0.0, 0.0, 0.0, 0.0]])
    mask = QwenBridgeEditor.predicted_mask(p, threshold=0.35, min_ratio=0.25, max_ratio=0.5)
    assert torch.equal(mask.sum(1), torch.tensor([2, 1]))


def test_native_attention_is_a_source_spatial_prior_and_uniform_is_neutral():
    torch.manual_seed(9)
    model = GroundedConditioner(24, dimension=8, heads=2, depth=1).eval()
    readout, visual = torch.randn(2, 24), torch.randn(2, 9, 24)
    condition, prior, logits = model(readout, visual)
    uniform = torch.full((2, 9), 1/9)
    condition2, prior2, logits2 = model(readout, visual, uniform)
    assert torch.allclose(condition, condition2, atol=1e-6)
    assert torch.allclose(prior, prior2, atol=1e-6)
    assert torch.allclose(logits, logits2, atol=1e-6)
    focused = torch.full((2, 9), .01)
    focused[:, 4] = .92
    _, guided, guided_logits = model(readout, visual, focused)
    assert (guided[:, 4] > prior[:, 4]).all()
    assert (guided_logits[:, 4] > logits[:, 4]).all()


def test_qwen_bridge_reconstruction_and_region_gradients_with_frozen_denoiser():
    torch.manual_seed(123)
    config = PhiConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=4, partial_rotary_factor=0.5)
    config._attn_implementation = 'sdpa'
    model = QwenBridgeEditor(PhiForCausalLM(config), feature_hidden=24, dimension=8, heads=2,
                             depth=2, adapter_width=8, visual_offset=40, codebook_size=16, mask_id=63)
    target = torch.randint(0, 16, (2, 9))
    text = torch.randint(1, 5, (2, 4))
    readout, visual, tf = torch.randn(2, 24), torch.randn(2, 9, 24), torch.randn(2, 4, 24)
    weak = torch.zeros(2, 9)
    weak[:, 3:5] = 1
    special = dict(mmu=5, soi=6, eoi=7, t2i=8, pad=0)
    native = torch.full((2, 9), .01)
    native[:, 3:5] = .465
    out = model(target, text, text != 0, special, readout, visual, tf, text != 0, weak, native_prior=native)
    out['loss'].backward()
    assert model.conditioner.region_bias.weight.grad.norm() > 0
    assert model.conditioner.triplet.weight.grad.norm() > 0
    assert model.condition_projections[0].weight.grad.norm() > 0
    assert model.visual_adapter[0].weight.grad.norm() > 0
    assert all(p.grad is None for p in model.llm.parameters())
