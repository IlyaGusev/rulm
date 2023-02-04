import argparse
import json
import os
import random
from itertools import chain

import wandb
from datasets import load_dataset, load_from_disk
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from rulm.preprocess import tokenize, group, MAX_TOKENS


os.environ["WANDB_LOG_MODEL"] = "checkpoint"


def train(
    dataset_path,
    tokenizer_path,
    output_dir,
    checkpoint,
    sample_rate,
    config_path,
    report_to,
    local_rank,
    preprocess,
    streaming,
    from_disk,
    in_memory
):
    with open(config_path) as r:
        config = json.load(r)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if not from_disk:
        datasets = load_dataset(dataset_path, streaming=streaming)
    else:
        datasets = load_from_disk(dataset_path, keep_in_memory=in_memory)

    block_size = config["block_size"]
    if preprocess:
        position_ids = [i % block_size for i in range(MAX_TOKENS)]
        datasets = datasets.filter(
            lambda x: random.random() < sample_rate
        ).map(
            lambda x: tokenize(x, tokenizer, position_ids),
            batched=True,
            remove_columns=["text"]
        ).map(
            lambda x: group(x, block_size),
            batched=True
        )

    train_dataset = datasets["train"]
    val_dataset = datasets["validation"]

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

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
    deepspeed_config = config.pop("deepspeed", None)
    training_args = TrainingArguments(
        output_dir=output_dir,
        load_best_model_at_end=True,
        save_total_limit=1,
        report_to=report_to,
        deepspeed=deepspeed_config,
        **trainer_config
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    with wandb.init(project="rulm", name=config_path) as run:
        trainer.train(checkpoint)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        wandb.save(output_dir + "/*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--from-disk", action="store_true", default=False)
    parser.add_argument("--in-memory", action="store_true", default=None)
    parser.add_argument("--preprocess", action="store_true", default=False)
    parser.add_argument("--streaming", action="store_true", default=False)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    train(**vars(args))
