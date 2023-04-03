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

An example of a training script run:

```python
python3 scripts/train.py --config-file configs/llama_7b_lora.json --train-file train.jsonl --val-file val.jsonl  --output-dir models/llama_7b_lora
```
