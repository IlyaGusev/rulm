import json
import random
import fire
from typing import List, Dict

import wandb
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments, Trainer
from trl import ORPOConfig, ORPOTrainer
from datasets import Dataset as HFDataset
from unsloth import PatchDPOTrainer, FastLanguageModel

from src.util.io import read_jsonl

PatchDPOTrainer()

DEFAULT_SYSTEM_MESSAGE = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."


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
            if prompt_messages[0]["role"] != "system":
                prompt_messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE})
            prompt = self.tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                tokenize=False
            )
            prompt = prompt.replace(self.tokenizer.bos_token, "")
            prompt_tokens = self.tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                tokenize=True
            )
            chosen = record["chosen"][0]["content"]
            chosen_tokens = self.tokenizer(chosen)["input_ids"]

            rejected = record["rejected"][0]["content"]
            rejected_tokens = self.tokenizer(rejected)["input_ids"]

            if len(prompt_tokens) + len(chosen_tokens) > self.max_tokens_count:
                continue
            if len(prompt_tokens) + len(rejected_tokens) > self.max_tokens_count:
                continue

            self.records.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def train(
    config_file: str,
    train_path: str,
    eval_path: str,
    output_dir: str,
    sample_rate: float = 1.0
):
    with open(config_file, "r") as r:
        config = json.load(r)

    max_tokens_count = config["max_tokens_count"]
    max_seq_length = config.get("max_seq_length", max_tokens_count)
    model_name = config["model_name"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_8bit=config["load_in_8bit"],
        load_in_4bit=config["load_in_4bit"],
        attn_implementation="flash_attention_2",
    )
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = "<|begin_of_text|>"
    tokenizer.eos_token = "<|eot_id|>"
    tokenizer.padding_side = "left"
    tokenizer.save_pretrained(output_dir)

    lora_config = config["lora"]
    #lora_config = LoraConfig(**lora_config)
    if lora_config:
        model = FastLanguageModel.get_peft_model(
            model,
            **config["lora"],
            max_seq_length=max_seq_length
        )


    #model = AutoModelForCausalLM.from_pretrained(
    #    model_name,
    #    torch_dtype=torch.bfloat16,
    #    device_map="auto",
    #    load_in_8bit=config["load_in_8bit"],
    #    load_in_4bit=config["load_in_4bit"],
    #    attn_implementation="flash_attention_2",
    #)
    train_records = read_jsonl(train_path)
    train_dataset = ChatRewardDataset(
        train_records,
        tokenizer=tokenizer,
        max_tokens_count=max_tokens_count,
        sample_rate=sample_rate
    )
    train_dataset = HFDataset.from_list(train_dataset)
    eval_records = read_jsonl(eval_path)
    eval_dataset = ChatRewardDataset(
        eval_records,
        tokenizer=tokenizer,
        max_tokens_count=max_tokens_count,
        sample_rate=sample_rate
    )
    eval_dataset = HFDataset.from_list(eval_dataset)
    print(train_dataset[0])

    trainer_config = config.get("trainer")
    if trainer_config.get("report_to", "wandb") == "wandb":
        wandb.init(project="rulm_self_instruct", name=config_file)

    training_args = ORPOConfig(
        output_dir=output_dir,
        report_to="wandb",
        **config["orpo"],
        **trainer_config
    )

    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        #peft_config=lora_config
    )

    trainer.train()
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
