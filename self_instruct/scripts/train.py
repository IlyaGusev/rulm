import argparse
import random
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, logging
from peft import get_peft_model, LoraConfig

from dataset import InstructDataset
from utils import set_random_seed, fix_tokenizer, fix_model, read_jsonl

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(
    config_file,
    checkpoint,
    train_file,
    val_file,
    train_sample_rate,
    val_sample_rate,
    output_dir,
    report_to,
    seed,
    local_rank
):
    set_random_seed(seed)
    logging.set_verbosity_info()
    with open(config_file, "r") as r:
        config = json.load(r)

    deepspeed_config = config.get("deepspeed")
    trainer_config = config["trainer"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to=report_to,
        deepspeed=deepspeed_config,
        **trainer_config
    )

    model_name = config["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = fix_tokenizer(tokenizer)
    tokenizer.save_pretrained(output_dir)

    train_records = read_jsonl(train_file)
    val_records = read_jsonl(val_file)
    random.shuffle(train_records)
    print(train_records[0])

    model_type = config.get("model_type", "causal")
    max_source_tokens_count = config["max_source_tokens_count"]
    max_target_tokens_count = config["max_target_tokens_count"]
    train_dataset = InstructDataset(
        train_records,
        tokenizer,
        max_source_tokens_count=max_source_tokens_count,
        max_target_tokens_count=max_target_tokens_count,
        sample_rate=train_sample_rate,
        input_type=model_type
    )
    print(train_dataset[0])

    val_dataset = InstructDataset(
        val_records,
        tokenizer,
        max_source_tokens_count=max_source_tokens_count,
        max_target_tokens_count=max_target_tokens_count,
        sample_rate=val_sample_rate,
        input_type=model_type
    )

    model_types = {
        "causal": AutoModelForCausalLM,
        "seq2seq": AutoModelForSeq2SeqLM
    }
    load_in_8bit = bool(config.get("load_in_8bit", False))
    if load_in_8bit:
        model = model_types[model_type].from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto"
        )
    else:
        model = model_types[model_type].from_pretrained(model_name)
    model = fix_model(model, tokenizer, max_target_tokens_count)

    # Default model generation params
    model.config.num_beams = 5
    max_tokens_count = max_target_tokens_count + max_source_tokens_count
    model.config.max_length = max_tokens_count if model_type == "causal" else max_target_tokens_count

    lora_config = config.get("lora")
    if lora_config:
        lora_config = LoraConfig(**lora_config)
        model = get_peft_model(model, lora_config)

    deepspeed_config = config.get("deepspeed")
    trainer_config = config["trainer"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to=report_to,
        deepspeed=deepspeed_config,
        **trainer_config
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train(checkpoint)
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--val-file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-sample-rate", type=float, default=1.0)
    parser.add_argument("--val-sample-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", type=str, default="none")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    train(**vars(args))
