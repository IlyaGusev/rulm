import sys
import json
import os

import torch
from peft import PeftModel
from transformers import LlamaForCausalLM

base_model_path = sys.argv[1]
peft_model_path = sys.argv[2]
output_path = sys.argv[3]

base_model = LlamaForCausalLM.from_pretrained(
    base_model_path,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={'': 'cpu'},
)

lora_model = PeftModel.from_pretrained(
    base_model,
    peft_model_path,
    device_map={'': 'cpu'},
    torch_dtype=torch.float16,
)

lora_model = lora_model.merge_and_unload()
lora_model.train(False)

if '7b' in base_model_path:
    n_layers = 32
    n_heads = 32
    dim = 4096
elif '13b' in base_model_path:
    n_layers = 40
    n_heads = 40
    dim = 5120
elif '30b' in base_model_path:
    n_layers = 60
    n_heads = 52
    dim = 6656
else:
    raise NotImplementedError


def unpermute(w):
    return (
        w.view(n_heads, 2, dim // n_heads // 2, dim)
        .transpose(1, 2)
        .reshape(dim, dim)
    )


def translate_state_dict_key(k):  # noqa: C901
    k = k.replace('base_model.model.', '')
    if k == 'model.embed_tokens.weight':
        return 'tok_embeddings.weight'
    elif k == 'model.norm.weight':
        return 'norm.weight'
    elif k == 'lm_head.weight':
        return 'output.weight'
    elif k.startswith('model.layers.'):
        layer = k.split('.')[2]
        if k.endswith('.self_attn.q_proj.weight'):
            return f'layers.{layer}.attention.wq.weight'
        elif k.endswith('.self_attn.k_proj.weight'):
            return f'layers.{layer}.attention.wk.weight'
        elif k.endswith('.self_attn.v_proj.weight'):
            return f'layers.{layer}.attention.wv.weight'
        elif k.endswith('.self_attn.o_proj.weight'):
            return f'layers.{layer}.attention.wo.weight'
        elif k.endswith('.mlp.gate_proj.weight'):
            return f'layers.{layer}.feed_forward.w1.weight'
        elif k.endswith('.mlp.down_proj.weight'):
            return f'layers.{layer}.feed_forward.w2.weight'
        elif k.endswith('.mlp.up_proj.weight'):
            return f'layers.{layer}.feed_forward.w3.weight'
        elif k.endswith('.input_layernorm.weight'):
            return f'layers.{layer}.attention_norm.weight'
        elif k.endswith('.post_attention_layernorm.weight'):
            return f'layers.{layer}.ffn_norm.weight'
        elif k.endswith('rotary_emb.inv_freq') or 'lora' in k:
            return None
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


lora_model_sd = lora_model.state_dict()
new_state_dict = {}
for k, v in lora_model_sd.items():
    new_k = translate_state_dict_key(k)
    if new_k is not None:
        if 'wq' in new_k or 'wk' in new_k:
            new_state_dict[new_k] = unpermute(v)
        else:
            new_state_dict[new_k] = v

os.makedirs(output_path, exist_ok=True)
torch.save(new_state_dict, f'{output_path}/consolidated.00.pth')
