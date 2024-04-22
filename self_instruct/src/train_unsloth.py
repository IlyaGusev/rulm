import json

import fire
import wandb
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
from unsloth import FastLanguageModel

from src.dataset import ChatDataset


def train(config_path: str, train_path: str, val_path: str, output_dir: str):
    with open(config_path) as r:
        config = json.load(r)

    max_seq_length = config["max_tokens_count"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_8bit=config["load_in_8bit"],
        load_in_4bit=config["load_in_4bit"]
    )
    tokenizer.add_special_tokens({'additional_special_tokens': ["<|im_start|>", "<|im_end|>", "<|reserved_special_token_0|>"]})
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.bos_token = "<|im_start|>"
    tokenizer.pad_token = "<|reserved_special_token_0|>"
    tokenizer.padding_side = "left"
    tokenizer.save_pretrained(output_dir)

    lora_config = config.get("lora")
    if lora_config:
        model = FastLanguageModel.get_peft_model(
            model,
            **config["lora"],
            max_seq_length=max_seq_length
        )
    mapping = {
        "bot": "assistant"
    }
    with open(train_path) as r:
        train_records = [json.loads(line) for line in r]
        for r in train_records:
            r["messages"] = [{"content": m["content"], "role": mapping.get(m["role"], m["role"])} for m in r["messages"]]
    with open(val_path) as r:
        val_records = [json.loads(line) for line in r]
        for r in val_records:
            r["messages"] = [{"content": m["content"], "role": mapping.get(m["role"], m["role"])} for m in r["messages"]]

    train_dataset = ChatDataset(
        train_records,
        tokenizer,
        max_tokens_count=max_seq_length,
        templates_path=config["templates_path"],
        only_target_loss=config["only_target_loss"]
    )

    val_dataset = ChatDataset(
        val_records,
        tokenizer,
        max_tokens_count=max_seq_length,
        templates_path=config["templates_path"],
        only_target_loss=config["only_target_loss"]
    )
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    trainer_config = config["trainer"]
    if trainer_config.get("report_to", "wandb") == "wandb":
        wandb.init(project="rulm_self_instruct", name=config_path)
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        args=TrainingArguments(
            **trainer_config,
            output_dir=output_dir
        ),
    )
    trainer.train()
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
