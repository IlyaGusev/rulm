import random
import json
from typing import Optional
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm import tqdm

from src.util.chat import Conversation


class InstructDataset(Dataset):
    def __init__(
        self,
        original_records: List[Dict],
        tokenizer: AutoTokenizer,
        max_source_tokens_count: int,
        max_target_tokens_count: int,
        templates_path: str,
        sample_rate: float = 1.0,
        only_target_loss: bool = True,
        input_type: str = "causal",
        target_field: str = "output",
        source_field: str = "input",
        use_padding: bool = False
    ):
        self.original_records = original_records
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.max_source_tokens_count = max_source_tokens_count
        self.max_target_tokens_count = max_target_tokens_count
        self.only_target_loss = only_target_loss
        self.input_type = input_type
        self.target_field = target_field
        self.source_field = source_field
        self.use_padding = use_padding
        self.is_printed = False

        with open(templates_path) as r:
            self.templates = json.load(r)

        self.records = []
        for record in tqdm(original_records):
            if random.random() > self.sample_rate:
                continue
            tensors = self.convert_record(record)
            if tensors is None:
                continue
            self.records.append(tensors)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def convert_record(self, record):
        instruction = record["instruction"]
        inp = record[self.source_field]
        out = record[self.target_field]
        if inp.strip() != "":
            templates = self.templates["prompts_input"]
            prompt_template = random.choice(templates)
            source = prompt_template.format(instruction=instruction.strip(), inp=inp.strip())
        else:
            templates = self.templates["prompts_no_input"]
            prompt_template = random.choice(templates)
            source = prompt_template.format(instruction=instruction.strip())
        target = out.strip()
        if not self.is_printed:
            print("Source and target examples")
            print(source)
            print(target)
            self.is_printed = True
        if self.input_type == "causal":
            return self.convert_causal(source, target)
        elif self.input_type == "seq2seq":
            return self.convert_seq2seq(source, target)
        else:
            assert False

    def convert_causal(self, source, target=None):
        source_tokens = self.tokenizer(
            source,
            add_special_tokens=False,
            max_length=self.max_source_tokens_count,
            padding=False,
            truncation=True
        )["input_ids"]
        if self.tokenizer.bos_token_id:
            source_tokens.insert(0, self.tokenizer.bos_token_id)
        input_ids = source_tokens[:]
        actual_length = len(input_ids)
        max_length = self.max_source_tokens_count + self.max_target_tokens_count + 2
        if target is not None:
            target_tokens = self.tokenizer(
                target,
                add_special_tokens=False,
                max_length=self.max_target_tokens_count,
                padding=False,
                truncation=True
            )["input_ids"]
            input_ids += target_tokens + [self.tokenizer.eos_token_id]
            actual_length = len(input_ids)
            if self.use_padding:
                padding = [self.tokenizer.pad_token_id for i in range(len(input_ids), max_length)]
                input_ids.extend(padding)

        input_ids = torch.LongTensor(input_ids)
        labels = input_ids.clone()
        attention_mask = input_ids.new_ones(input_ids.size())
        if self.use_padding:
            labels[actual_length:] = -100
            attention_mask[actual_length:] = 0
        if self.only_target_loss:
            labels[:len(source_tokens)] = -100
        assert input_ids.size(0) == labels.size(0) == attention_mask.size(0) <= max_length

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

    def convert_seq2seq(self, source, target=None):
        inputs = self.tokenizer(
            source,
            add_special_tokens=True,
            max_length=self.max_source_tokens_count,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        if target is not None:
            outputs = self.tokenizer(
                target,
                add_special_tokens=True,
                max_length=self.max_target_tokens_count,
                padding=False,
                truncation=True,
                return_tensors="pt"
            )
            labels = outputs["input_ids"].squeeze(0).tolist()
            if labels[-1] != self.tokenizer.eos_token_id:
                labels.append(self.tokenizer.eos_token_id)
            inputs["labels"] = torch.LongTensor(labels)
        return inputs


class ChatDataset(Dataset):
    def __init__(
        self,
        original_records: List[Dict],
        tokenizer: AutoTokenizer,
        max_tokens_count: int,
        templates_path: str,
        sample_rate: float = 1.0,
        only_target_loss: bool = True,
        add_global_bos: bool = False,
        add_global_eos: bool = False
    ):
        self.templates_path = templates_path
        self.original_records = original_records
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.max_tokens_count = max_tokens_count
        self.only_target_loss = only_target_loss
        self.is_printed = False
        self.add_global_bos = add_global_bos
        self.add_global_eos = add_global_eos

        self.records = []
        for record in tqdm(original_records):
            if random.random() > self.sample_rate:
                continue
            tensors = self.convert_record(record)
            if tensors is None:
                continue
            self.records.append(tensors)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def get_tokens(self, text):
        return self.tokenizer(
            text,
            add_special_tokens=False,
            padding=False,
            truncation=False
        )["input_ids"]

    def convert_record(self, record):
        conversation = Conversation.from_template(self.templates_path)
        conversation.expand(record["messages"])
        full_text = conversation.get_prompt(self.tokenizer)

        if not self.is_printed:
            print("Prompt example")
            print(full_text)
            self.is_printed = True

        input_ids = self.get_tokens(full_text)
        if self.add_global_bos:
            input_ids.insert(0, self.tokenizer.bos_token_id)
        input_ids = input_ids[:self.max_tokens_count-1]
        if self.add_global_eos or input_ids[-1] != self.tokenizer.eos_token_id:
            input_ids.append(self.tokenizer.eos_token_id)
        actual_length = len(input_ids)

        input_ids = torch.LongTensor(input_ids)
        labels = input_ids.clone()
        attention_mask = input_ids.new_ones(input_ids.size())

        if self.only_target_loss:
            start_token_id = conversation.get_start_token_id()
            end_token_id = conversation.get_end_token_id()
            bot_token_id = conversation.get_bot_token_id()

            spans = []
            cur_start_idx = -1
            cur_end_idx = -1
            cur_is_bot = False

            input_ids = input_ids.tolist()
            while True:
                try:
                    cur_start_idx = input_ids.index(start_token_id, cur_start_idx + 1)
                    cur_end_idx = input_ids.index(end_token_id, cur_start_idx + 1) + 1
                    cur_is_bot = input_ids[cur_start_idx:cur_end_idx].count(bot_token_id) >= 1
                    if not cur_is_bot:
                        spans.append((cur_start_idx, cur_end_idx))
                except ValueError:
                    break
            for start_idx, end_idx in spans:
                start_idx = max(0, start_idx)
                end_idx = min(len(input_ids), end_idx)
                labels[start_idx: end_idx] = -100

            if (labels == start_token_id).sum() == 0:
                return None

            assert (labels == start_token_id).sum() == (labels == end_token_id).sum()
            assert (labels == bot_token_id).sum() >= (labels == start_token_id).sum()

        input_ids = torch.LongTensor(input_ids)
        assert input_ids.size(0) == labels.size(0) == attention_mask.size(0) <= self.max_tokens_count
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }
