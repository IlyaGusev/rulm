import json

import fire
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
from unsloth import FastLanguageModel

from src.dataset import ChatDataset


def train(model_dir, max_seq_length: int = 8192, output_dir: str = "unsloth_model"):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_8bit=False,
        load_in_4bit=True
    )
    tokenizer.add_special_tokens({'additional_special_tokens': ["<|im_start|>", "<|im_end|>", "<|reserved_special_token_0|>"]})
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.pad_token = "<|reserved_special_token_0|>"
    tokenizer.padding_side = "left"
    tokenizer.save_pretrained(output_dir)

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=1337,
        max_seq_length=max_seq_length,
        use_gradient_checkpointing=True
    )
    mapping = {
        "bot": "assistant"
    }
    with open("train.jsonl") as r:
        train_records = [json.loads(line) for line in r]
        for r in train_records:
            r["messages"] = [{"content": m["content"], "role": mapping.get(m["role"], m["role"])} for m in r["messages"]]
    with open("val.jsonl") as r:
        val_records = [json.loads(line) for line in r]
        for r in val_records:
            r["messages"] = [{"content": m["content"], "role": mapping.get(m["role"], m["role"])} for m in r["messages"]]


    train_dataset = ChatDataset(
        train_records,
        tokenizer,
        max_tokens_count=max_seq_length,
        templates_path="internal_prompts/chaml.json",
        only_target_loss=True
    )

    val_dataset = ChatDataset(
        val_records,
        tokenizer,
        max_tokens_count=max_seq_length,
        templates_path="internal_prompts/chaml.json",
        only_target_loss=True
    )
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        args=TrainingArguments(
            num_train_epochs=3,
            per_device_train_batch_size=3,
            per_device_eval_batch_size=3,
            gradient_accumulation_steps=42,
            warmup_steps=4,
            learning_rate=0.0002,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            eval_steps=8,
            save_steps=8,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=1337,
            output_dir=output_dir,
            evaluation_strategy="steps",
            load_best_model_at_end=True
        ),
    )
    trainer.train()
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
