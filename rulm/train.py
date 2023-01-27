import argparse
import json
import random
from itertools import chain

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


def tokenize(examples, tokenizer):
    outputs = tokenizer(
        examples["text"],
        truncation=False,
        max_length=None,
        padding=False
    )
    return outputs


def group(examples, block_size):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # Padding for the last example is handled in data collator
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def train(
    dataset_path,
    train_path,
    val_path,
    tokenizer_path,
    output_dir,
    checkpoint,
    sample_rate,
    config_path,
    local_rank
):
    assert dataset_path or (train_path and val_path)

    with open(config_path) as r:
        config = json.load(r)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.padding_side = "right"

    if dataset_path:
        datasets = load_dataset(dataset_path, streaming=True)
    else:
        datasets = load_dataset("rulm/jsonl_loader.py", data_files={
            "train": [train_path],
            "val": [val_path]
        }, streaming=True)

    block_size = config["block_size"]
    datasets = datasets.filter(
        lambda x: random.random() < sample_rate
    ).map(
        lambda x: tokenize(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    ).map(
        lambda x: group(x, block_size),
        batched=True
    )

    train_dataset = datasets["train"]
    val_dataset = datasets["validation"]
    train_dataset.shuffle(seed=42, buffer_size=10000)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False,
        pad_to_multiple_of=block_size
    )

    model_params = config["model"]
    model_type = model_params.pop("type")
    model_config = AutoConfig.from_pretrained(
        model_type,
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **model_params
    )
    print(model_config)

    model = AutoModelForCausalLM.from_config(model_config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")

    model.config.num_beams = 5
    model.config.max_length = block_size

    trainer_config = config["trainer"]
    deepspeed_config = config.get("deepspeed")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=trainer_config.get("batch_size", 32),
        per_device_eval_batch_size=trainer_config.get("batch_size", 32),
        logging_steps=trainer_config.get("logging_steps", 100),
        eval_steps=trainer_config.get("eval_steps", 5000),
        evaluation_strategy="steps",
        save_steps=trainer_config.get("eval_steps", 5000),
        learning_rate=trainer_config.get("learning_rate", 5e-4),
        weight_decay=trainer_config.get("weight_decay", 0.1),
        lr_scheduler_type=trainer_config.get("lr_scheduler_type", "cosine"),
        warmup_steps=trainer_config.get("warmup_steps", 1000),
        num_train_epochs=trainer_config.get("num_train_epochs", None),
        max_steps=trainer_config.get("max_steps", 100000),
        gradient_accumulation_steps=trainer_config.get("gradient_accumulation_steps", 8),
        fp16=trainer_config.get("fp16", False),
        bf16=trainer_config.get("bf16", False),
        half_precision_backend=trainer_config.get("half_precision_backend", "auto"),
        fp16_opt_level=trainer_config.get("fp16_opt_level", "O2"),
        gradient_checkpointing=trainer_config.get("gradient_checkpointing", False),
        optim=trainer_config.get("optim", "adamw_apex_fused"),
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="none",
        deepspeed=deepspeed_config
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    result = trainer.train(checkpoint)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--val-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    train(**vars(args))
