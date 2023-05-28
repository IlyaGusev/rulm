import argparse
import random
import json
import os

import wandb
import torch
import numpy as np
import bitsandbytes as bnb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, DataCollatorForTokenClassification, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments, logging, TrainerCallback, TrainerState, TrainerControl, BitsAndBytesConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import get_peft_model, LoraConfig, prepare_model_for_int8_training

from src.dataset import InstructDataset, ChatDataset
from src.util.dl import set_random_seed, fix_tokenizer, fix_model
from src.util.io import read_jsonl

os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TrainerNoBaseSave(Trainer):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def _save_checkpoint(self, model, trial, metrics=None):
        print("Running custom _save_checkpoint")
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]
            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        os.makedirs(output_dir, exist_ok=True)
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        return control


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
    local_rank,
    omit_base_model_save
):
    set_random_seed(seed)
    logging.set_verbosity_info()
    with open(config_file, "r") as r:
        config = json.load(r)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    deepspeed_config = config.get("deepspeed")
    trainer_config = config["trainer"]
    lora_config = config.get("lora")
    callbacks = [SavePeftModelCallback] if lora_config else []
    training_args = TrainingArguments(
        output_dir=output_dir,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to=report_to,
        ddp_find_unused_parameters=False if ddp else None,
        deepspeed=deepspeed_config,
        **trainer_config
    )
    model_name = config["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer = fix_tokenizer(tokenizer)
    tokenizer.save_pretrained(output_dir)

    train_records = read_jsonl(train_file)
    val_records = read_jsonl(val_file)
    random.shuffle(train_records)
    print(train_records[0])

    model_type = config.get("model_type", "causal")
    templates_path = config.get("templates_path", "ru_alpaca_template.json")
    only_target_loss = config.get("only_target_loss", True)
    mode = config.get("mode", "instruct")
    if mode == "instruct":
        max_source_tokens_count = config["max_source_tokens_count"]
        max_target_tokens_count = config["max_target_tokens_count"]
        target_field = config.get("target_field", "output")
        source_field = config.get("source_field", "input")

        train_dataset = InstructDataset(
            train_records,
            tokenizer,
            max_source_tokens_count=max_source_tokens_count,
            max_target_tokens_count=max_target_tokens_count,
            sample_rate=train_sample_rate,
            input_type=model_type,
            templates_path=templates_path,
            target_field=target_field,
            source_field=source_field,
            only_target_loss=only_target_loss
        )

        val_dataset = InstructDataset(
            val_records,
            tokenizer,
            max_source_tokens_count=max_source_tokens_count,
            max_target_tokens_count=max_target_tokens_count,
            sample_rate=val_sample_rate,
            input_type=model_type,
            templates_path=templates_path,
            target_field=target_field,
            source_field=source_field,
            only_target_loss=only_target_loss
        )
    elif mode == "chat":
        max_tokens_count = config["max_tokens_count"]

        train_dataset = ChatDataset(
            train_records,
            tokenizer,
            max_tokens_count=max_tokens_count,
            sample_rate=train_sample_rate,
            templates_path=templates_path,
            only_target_loss=only_target_loss
        )

        val_dataset = ChatDataset(
            val_records,
            tokenizer,
            max_tokens_count=max_tokens_count,
            sample_rate=train_sample_rate,
            templates_path=templates_path,
            only_target_loss=only_target_loss
        )
    else:
        assert False

    if "seq2seq" in model_type:
        data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    print("INPUT_IDS")
    print(data_collator([train_dataset[0], train_dataset[1]])["input_ids"][0])
    print("MASK")
    print(data_collator([train_dataset[0], train_dataset[1]])["attention_mask"][0])
    print("LABELS")
    print(data_collator([train_dataset[0], train_dataset[1]])["labels"][0])

    model_types = {
        "causal": AutoModelForCausalLM,
        "seq2seq": AutoModelForSeq2SeqLM
    }
    load_in_8bit = bool(config.get("load_in_8bit", False))
    load_in_4bit = bool(config.get("load_in_4bit", False))
    if load_in_8bit:
        assert not load_in_4bit
        model = model_types[model_type].from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map=device_map
        )
        model = fix_model(model, tokenizer, use_resize=False)
        model = prepare_model_for_int8_training(model)
    elif load_in_4bit:
        assert not load_in_8bit
        use_bf16 = trainer_config.get("bf16", False)
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        model = model_types[model_type].from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map=device_map,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            ),
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float32
        )
        model = fix_model(model, tokenizer, use_resize=False)
        model = prepare_model_for_int8_training(model)
    else:
        model = model_types[model_type].from_pretrained(model_name)
        model = fix_model(model, tokenizer)

    # Default model generation params
    model.config.num_beams = 5
    if mode == "instruct":
        max_tokens_count = max_target_tokens_count + max_source_tokens_count + 1
    model.config.max_length = max_tokens_count if model_type == "causal" else max_target_tokens_count

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    if lora_config:
        lora_config = LoraConfig(**lora_config)
        model = get_peft_model(model, lora_config)

    trainer_class = Trainer if not omit_base_model_save else TrainerNoBaseSave
    print("Trainer class:", trainer_class)
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=callbacks,
        data_collator=data_collator
    )

    with wandb.init(project="rulm_self_instruct", name=config_file) as run:
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
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--omit-base-model-save", action="store_true", default=False)
    args = parser.parse_args()
    train(**vars(args))
