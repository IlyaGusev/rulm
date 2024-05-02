import json
import random
import fire
from typing import List, Dict

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, TrainingArguments, Trainer
from trl import RewardTrainer, RewardConfig

from src.util.io import read_jsonl


class ChatRewardDataset(Dataset):
    def __init__(
        self,
        original_records: List[Dict],
        tokenizer: AutoTokenizer,
        max_tokens_count: int,
        sample_rate: float = 1.0
    ):
        self.original_records = original_records
        self.tokenizer = tokenizer
        self.max_tokens_count = max_tokens_count
        self.sample_rate = sample_rate

        self.records = []
        for record in tqdm(original_records):
            if random.random() > self.sample_rate:
                continue

            prompt_messages = record["prompt"]
            chosen_messages = prompt_messages + record["chosen"]
            rejected_messages = prompt_messages + record["rejected"]
            chosen_tensors = self.convert_messages(chosen_messages)
            rejected_tensors = self.convert_messages(rejected_messages)
            if not chosen_tensors or not rejected_tensors:
                continue
            self.records.append({
                "input_ids_chosen": chosen_tensors["input_ids"],
                "attention_mask_chosen": chosen_tensors["attention_mask"],
                "input_ids_rejected": rejected_tensors["input_ids"],
                "attention_mask_rejected": rejected_tensors["attention_mask"],
            })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def convert_messages(self, messages):
        data = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False)
        input_ids = torch.LongTensor(data)
        if len(input_ids) > self.max_tokens_count:
            return None
        attention_mask = input_ids.new_ones(input_ids.size())
        assert input_ids.size(0) == attention_mask.size(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }


def train(
    config_file: str,
    train_path: str,
    eval_path: str,
    output_dir: str,
    sample_rate: float = 1.0
):
    with open(config_file, "r") as r:
        config = json.load(r)

    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = "<|begin_of_text|>"
    tokenizer.eos_token = "<|eot_id|>"
    tokenizer.padding_side = "left"
    tokenizer.save_pretrained(output_dir)

    max_tokens_count = config["max_tokens_count"]
    max_seq_length = config.get("max_seq_length", max_tokens_count)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_8bit=config["load_in_8bit"],
        load_in_4bit=config["load_in_4bit"],
        attn_implementation="flash_attention_2",
    )
    lora_config = config["lora"]
    lora_config = LoraConfig(**lora_config)
    train_records = read_jsonl(train_path)
    train_dataset = ChatRewardDataset(
        train_records,
        tokenizer=tokenizer,
        max_tokens_count=max_tokens_count,
        sample_rate=sample_rate
    )
    eval_records = read_jsonl(eval_path)
    eval_dataset = ChatRewardDataset(
        eval_records,
        tokenizer=tokenizer,
        max_tokens_count=max_tokens_count,
        sample_rate=sample_rate
    )
    print(train_dataset[0])

    trainer_config = config.get("trainer")
    training_args = RewardConfig(
        output_dir=output_dir,
        report_to="wandb",
        **trainer_config
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config
    )

    trainer.train()
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
