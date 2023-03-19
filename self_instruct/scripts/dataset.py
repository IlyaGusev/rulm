import random
from typing import List, Dict, Tuple, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm


TEMPLATES_WITH_INPUT = [
    ("{instruction}\nВход: {inp}\nВыход:", " {out}"),
    ("{instruction}\n\nВход: {inp}\n\nОтвет: ", "{out}"),
    ("Задание: {instruction}\nВход: {inp}\nВыход:", " {out}"),
    ("Инструкция: {instruction}\nДано: {inp}\nВыход:", " {out}"),
    ("{instruction}\n\n{inp}\n\nОтвет:", " {out}"),
    ("{instruction}\n\n{inp}\n\n", "{out}"),
    ("{instruction}\n{inp}\n\n", "{out}"),
    ("{instruction}\n{inp}\n", "{out}"),
    ("Задание: {instruction}\n\n{inp}\n\n", "{out}"),
]

TEMPLATES_NO_INPUT = [
    ("{instruction} Ответ:", " {out}"),
    ("{instruction} Выход:", " {out}"),
    ("{instruction}\nВыход:", " {out}"),
    ("{instruction}\n\nОтвет:", " {out}"),
    ("{instruction}\n", "{out}"),
    ("{instruction}\n\n", "{out}"),
    ("Задание: {instruction}\n\n", "{out}"),
    ("Инструкция: {instruction}\n\n", "{out}"),
]


class InstructDataset(Dataset):
    def __init__(
        self,
        original_records: List[Dict],
        tokenizer: AutoTokenizer,
        max_source_tokens_count: int,
        max_target_tokens_count: int,
        sample_rate: float = 1.0,
        only_target_loss: bool = True
    ):
        self.original_records = original_records
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.max_source_tokens_count = max_source_tokens_count
        self.max_target_tokens_count = max_target_tokens_count
        self.only_target_loss = only_target_loss

        self.records = []
        for record in tqdm(original_records):
            if random.random() > self.sample_rate:
                continue
            tensors = self.convert_record(record)
            self.records.append(tensors)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def convert_record(self, record):
        instruction = record["instruction"]
        inp = record["input"]
        out = record["output"]
        if inp.strip() != "":
            prompt_template, completion_template = random.choice(TEMPLATES_WITH_INPUT)
            source = prompt_template.format(instruction=instruction.strip(), inp=inp.strip())
        else:
            prompt_template, completion_template = random.choice(TEMPLATES_NO_INPUT)
            source = prompt_template.format(instruction=instruction.strip())
        target = completion_template.format(out=out.strip())

        source_tokens = self.tokenizer(
            source,
            add_special_tokens=False,
            max_length=self.max_source_tokens_count,
            padding=False,
            truncation=True
        )["input_ids"]
        input_ids = source_tokens + []
        if target is not None:
            target_tokens = self.tokenizer(
                target,
                add_special_tokens=False,
                max_length=self.max_target_tokens_count,
                padding=False,
                truncation=True
            )["input_ids"]
            input_ids += target_tokens + [self.tokenizer.eos_token_id]
            max_length = self.max_source_tokens_count + self.max_target_tokens_count + 2
            padding = [self.tokenizer.pad_token_id for i in range(len(input_ids), max_length)]
            input_ids.extend(padding)
        input_ids = torch.LongTensor(input_ids)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        if self.only_target_loss:
            for i in range(len(source_tokens) + 1):
                labels[i] = -100
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }
