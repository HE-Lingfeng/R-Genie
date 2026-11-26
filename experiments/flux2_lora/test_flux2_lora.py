import torch

from experiments.flux2.test_model import model,inputs
from experiments.flux2_lora.model import add_generation_lora


def test_generation_lora_starts_neutral_and_updates_only_adapters_and_latent_modules():
    torch.manual_seed(31)
    editor,data = model(True),inputs()
    with torch.no_grad():
        original = editor.vector_field(data['target'],data['source'],data['prompt_embeds'],data['timestep'],None)
    editor = add_generation_lora(editor,rank=2,alpha=4)
    with torch.no_grad():
        adapted = editor.vector_field(data['target'],data['source'],data['prompt_embeds'],data['timestep'],None)
    assert torch.equal(original,adapted)
    base = {n:p.detach().clone() for n,p in editor.transformer.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in editor.parameters() if p.requires_grad],lr=.001,weight_decay=0)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = editor(**data)
        result['loss'].backward()
        b = [p.grad for n,p in editor.transformer.named_parameters() if '.lora_B.' in n]
        assert any(x is not None and x.norm()>0 for x in b)
        optimizer.step()
    assert all(p.grad is None and torch.equal(p,base[n]) for n,p in editor.transformer.named_parameters() if not p.requires_grad)
    names = set(editor.trainable_state_dict())
    assert any('transformer.' in n and '.lora_B.' in n for n in names)
    assert not any('.base_layer.' in n for n in names)
