import argparse

import torch
from peft import PeftModel, PeftConfig
from transformers import LlamaForCausalLM
from tqdm.auto import tqdm


def unpermute(w, n_heads, dim):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('output_path')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--enable_offloading', action='store_true')
    args = parser.parse_args()

    assert args.output_path.endswith(".pt")

    config = PeftConfig.from_pretrained(args.model_name)
    base_model_path = config.base_model_name_or_path

    base_model = LlamaForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={'': args.device},
    )

    lora_model = PeftModel.from_pretrained(
        base_model,
        args.model_name,
        device_map={'': args.device},
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

    lora_model_sd = lora_model.state_dict()
    del lora_model, base_model
    total = len(lora_model_sd)
    with tqdm(list(lora_model_sd.keys())) as progress_bar:
        for i, k in enumerate(progress_bar):
            new_k = translate_state_dict_key(k)
            if new_k is None:
                continue
            v = lora_model_sd.pop(k)
            if 'wq' in new_k or 'wk' in new_k:
                lora_model_sd[new_k] = unpermute(v, n_heads=n_heads, dim=dim)
            else:
                lora_model_sd[new_k] = v

            if args.enable_offloading and i <= total // 2:
                # offload half of all tensors to RAM
                lora_model_sd[new_k] = lora_model_sd[new_k].cpu()

    print('Saving state_dict...')
    torch.save(lora_model_sd, f'{args.output_path}')


if __name__ == '__main__':
    main()
