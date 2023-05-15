## Training
Steps:
* Install a version of transformers, peft and bitsandbytes for requirements.txt
* Prepare your data as two JSONL files, with three fields: "instruction", "input", "output".
* Download some base model, for example, huggyllama/llama-7b. 
* Fix pad, bos, eos tokens everywhere. And also a name of the tokenizer.
* Run training.

Install libraries:
```
sudo apt-get install git-lfs
pip install -r ../requirements.txt
```

Download a base model:
```
git clone https://huggingface.co/huggyllama/llama-7b
```

Fix base model's tokenizer_config.json:
```
{
    "tokenizer_class": "LlamaTokenizer",
    "model_max_length": 2048,
    "padding_side": "left",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "clean_up_tokenization_spaces": false,
    "special_tokens_map_file": "special_tokens_map.json"  
}
```


Fix base model's special_tokens_map.json:
```
{
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<unk>",
    "sep_token": "<s>",
    "unk_token": "<unk>"
}
```


Fix base model's generation_config.json:
```
{
  "_from_model_config": true,
  "pad_token_id": 0,
  "bos_token_id": 1,
  "eos_token_id": 2
}
```

A training config example:
```
{
    "trainer": {
        "evaluation_strategy": "steps",
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 32,
        "eval_steps": 50,
        "save_steps": 50,
        "logging_steps": 5,
        "learning_rate": 0.0003,
        "num_train_epochs": 5,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 30,
        "fp16": true,
        "bf16": false,
        "torch_compile": false,
        "optim": "adamw_torch"
    },
    "lora": {
        "r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "task_type": "CAUSAL_LM"
    },
    "load_in_8bit": true,
    "only_target_loss": true,
    "mode": "chat",
    "templates_path": "internal_prompts/saiga_v2.json",
    "model_name": "models/llama-7b",
    "model_type": "causal",
    "max_tokens_count": 2000
}
```

Run training script:

```python
python3 -m src.train --config-file configs/saiga_7b.json --train-file train.jsonl --val-file val.jsonl  --output-dir models/saiga_7b
```
