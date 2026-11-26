import copy

import torch
from diffusers import Flux2Transformer2DModel

from experiments.flux2.model import FluxIntentionEditor


def model(checkpointed):
    base = Flux2Transformer2DModel(in_channels=8, num_layers=2, num_single_layers=2,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=24,
        timestep_guidance_channels=32, axes_dims_rope=[4, 4, 4, 4], mlp_ratio=2)
    return FluxIntentionEditor(base, feature_hidden=24, dimension=8, heads=2, depth=1,
                                adapter_width=8, gradient_checkpointing=checkpointed)


def inputs():
    return dict(source=torch.randn(2, 16, 8), target=torch.randn(2, 16, 8),
        prompt_embeds=torch.randn(2, 12, 24), readout=torch.randn(2, 24),
        visual=torch.randn(2, 256, 24), text_features=torch.randn(2, 4, 24),
        text_valid=torch.ones(2, 4, dtype=torch.bool), weak_region=torch.rand(2, 256),
        native_prior=torch.rand(2, 256), noise=torch.randn(2, 16, 8), timestep=torch.tensor([.3, .7]))


def test_zero_condition_projections_preserve_native_vector_field():
    torch.manual_seed(10)
    m, data = model(False).eval(), inputs()
    x = data['target']
    without = m.vector_field(x, data['source'], data['prompt_embeds'], data['timestep'], None)
    with_bias = m.vector_field(x, data['source'], data['prompt_embeds'], data['timestep'], torch.randn(2, 8))
    assert torch.equal(without, with_bias)


def test_checkpoint_recomputation_preserves_latent_and_projection_gradients():
    torch.manual_seed(12)
    eager, data = model(False), inputs()
    for layer in eager.condition_projections:
        torch.nn.init.normal_(layer.weight, std=.01)
    ckpt = model(True)
    ckpt.load_state_dict(copy.deepcopy(eager.state_dict()))
    snapshots = []
    for m in (eager, ckpt):
        base_before = {n: p.detach().clone() for n, p in m.transformer.named_parameters()}
        result = m(**data)
        result['loss'].backward()
        gradients = {n: p.grad.clone() for n, p in m.named_parameters() if p.requires_grad}
        assert gradients['conditioner.triplet.weight'].norm() > 0
        assert gradients['visual_adapter.0.weight'].norm() > 0
        assert gradients['condition_projections.0.weight'].norm() > 0
        assert all(p.grad is None and torch.equal(p, base_before[n]) for n, p in m.transformer.named_parameters())
        snapshots.append((result['loss'].detach(), gradients))
    assert torch.allclose(snapshots[0][0], snapshots[1][0])
    for name, gradient in snapshots[0][1].items():
        assert torch.allclose(gradient, snapshots[1][1][name], atol=2e-5, rtol=2e-4), name
