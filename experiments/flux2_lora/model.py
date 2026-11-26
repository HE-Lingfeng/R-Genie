"""Optional generation LoRA on the same latent intention/HRM/RAB pathway."""
import torch
from peft import LoraConfig
from torch import nn

from experiments.flux2.train import load_editor as load_frozen_editor


def add_generation_lora(editor,rank=8,alpha=16):
    transformer = editor.transformer
    names = [name for name,module in transformer.named_modules() if isinstance(module,nn.Linear)
        and '.attn.' in name and name.startswith(('transformer_blocks.','single_transformer_blocks.'))]
    expected = 8*len(transformer.transformer_blocks)+2*len(transformer.single_transformer_blocks)
    if len(names) != expected:
        raise ValueError(f'Unexpected native attention projections: {len(names)} != {expected}')
    transformer.add_adapter(LoraConfig(r=rank,lora_alpha=alpha,lora_dropout=0.,bias='none',target_modules=names))
    for name,parameter in transformer.named_parameters():
        if parameter.requires_grad:
            if '.lora_A.' not in name and '.lora_B.' not in name:
                raise ValueError('A pretrained generation weight was unexpectedly unfrozen: '+name)
            parameter.data = parameter.data.float()
    editor.generation_lora_targets = names
    return editor


def load_editor(config):
    editor = load_frozen_editor(config)
    return add_generation_lora(editor,config.model.lora_rank,config.model.lora_alpha)
