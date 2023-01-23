import argparse
import json

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


def tokenize(element, tokenizer, block_size):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=block_size,
        return_overflowing_tokens=True,
        return_length=True,
        padding=True
    )
    return {"input_ids": outputs["input_ids"]}


def train(
    train_path,
    val_path,
    tokenizer_path,
    output_dir,
    checkpoint,
    sample_rate,
    config_path
):
    with open(config_path) as r:
        config = json.load(r)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    datasets = load_dataset("rulm/jsonl_loader.py", data_files={
        "train": [train_path],
        "val": [val_path]
    }, streaming=True)

    block_size = config["block_size"]
    local_tokenize = lambda x: tokenize(x, tokenizer, block_size)
    tokenized_datasets = datasets.map(
        local_tokenize, batched=True, remove_columns=["text"]
    )

    train_dataset = tokenized_datasets["train"]
    train_dataset.shuffle(seed=42, buffer_size=10000)
    val_dataset = tokenized_datasets["validation"]

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    model_config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=block_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = GPT2LMHeadModel(model_config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")

    model.config.num_beams = 5
    model.config.max_length = block_size

    trainer_config = config["trainer"]
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
        max_steps=trainer_config.get("max_steps", 12000000),
        gradient_accumulation_steps=trainer_config.get("gradient_accumulation_steps", 8),
        fp16=trainer_config.get("fp16", False),
        bf16=trainer_config.get("bf16", False),
        gradient_checkpointing=trainer_config.get("gradient_checkpointing", False),
        optim=trainer_config.get("optim", "adamw_hf"),
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="none"
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
    print_summary(result)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    args = parser.parse_args()
    train(**vars(args))
