## Training
Steps:
* Install a dev version of transformers, peft and bitsandbytes.
* Prepare your data as two JSONL files, with three fields: "instruction", "input", "output".
* Download some base model, for example, decapoda-research/llama-7b-hf. 
* Fix pad, bos, eos tokens everywhere. And also a name of the tokenizer.
* Run training.

Installation:
```
sudo apt-get install git-lfs
pip install git+https://github.com/huggingface/transformers peft bitsandbytes
```

Downloading a base model:
```
git clone https://huggingface.co/decapoda-research/llama-7b-hf
```

Correct tokenizer_config.json:
```
{
    "model_max_length": 1000000000000000019884624838656,
    "tokenizer_class": "LlamaTokenizer"
}
```


Correct config.json:
```
{
  "pad_token_id": 0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "architectures": [
    "LLaMAForCausalLM"
  ],
  "hidden_act": "silu",
  "hidden_size": 4096,
  "intermediate_size": 11008,
  "initializer_range": 0.02,
  "max_sequence_length": 2048,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "rms_norm_eps": 1e-06,
  "torch_dtype": "float16",
  "transformers_version": "4.27.0.dev0",
  "use_cache": true,
  "vocab_size": 32000
}
```

Correct generation_config.json:
```
{
  "_from_model_config": true,
  "pad_token_id": 0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "transformers_version": "4.27.0.dev0"
}
```

An example of a training script run:

```python
python3 scripts/train.py --config-file configs/llama_7b_lora.json --train-file train.jsonl --val-file val.jsonl  --output-dir models/llama_7b_lora
```

A config example:
```
{
    "trainer": {
        "evaluation_strategy": "steps",
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "gradient_accumulation_steps": 8,
        "eval_steps": 75,
        "save_steps": 75,
        "logging_steps": 5,
        "learning_rate": 0.0003,
        "num_train_epochs": 3,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 50,
        "fp16": true,
        "bf16": false,
        "torch_compile": false,
        "optim": "adamw_torch"
    },
    "lora": {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM"
    },
    "load_in_8bit": true,
    "only_target_loss": false,
    "model_name": "models/llama-7b-hf",
    "model_type": "causal",
    "template_category": "causal_newlines",
    "max_source_tokens_count": 256,
    "max_target_tokens_count": 512
}
```
