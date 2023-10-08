import random
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from src.util.chat import Conversation


class ChatDataset(Dataset):
    def __init__(
        self,
        original_records: List[Dict],
        tokenizer: AutoTokenizer,
        max_tokens_count: int,
        templates_path: str,
        sample_rate: float = 1.0,
        only_target_loss: bool = True,
        add_global_bos: bool = True,
        add_global_eos: bool = True,
        labels_pad_token_id: int = -100,
        truncation_side: str = "left"
    ):
        self.templates_path = templates_path
        self.original_records = original_records
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.max_tokens_count = max_tokens_count
        self.only_target_loss = only_target_loss
        self.labels_pad_token_id = labels_pad_token_id
        self.add_global_bos = add_global_bos
        self.add_global_eos = add_global_eos
        self.truncation_side = truncation_side
        self.is_printed = False

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

        input_ids, labels = [], []
        for message, role in conversation.iter_messages():
            message_input_ids = self.get_tokens(message)
            message_labels = message_input_ids
            if len(input_ids) + len(message_input_ids) > self.max_tokens_count:
                break

            labels_mask = [self.labels_pad_token_id for _ in range(len(message_input_ids))]
            if role != conversation.bot_role and self.only_target_loss:
                message_labels = labels_mask

            input_ids.extend(message_input_ids)
            labels.extend(message_labels)

        if self.add_global_bos and input_ids[0] != self.tokenizer.bos_token_id:
            input_ids.insert(0, self.tokenizer.bos_token_id)
            labels.insert(0, self.labels_pad_token_id)

        if self.add_global_eos and input_ids[-1] != self.tokenizer.eos_token_id:
            input_ids.append(self.tokenizer.eos_token_id)
            labels.append(self.tokenizer.eos_token_id)

        if not self.is_printed:
            print(input_ids)
            print(labels)
            print("Full prompt:", self.tokenizer.decode(input_ids, skip_special_tokens=False))
            self.is_printed = True

        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        attention_mask = input_ids.new_ones(input_ids.size())
        assert input_ids.size(0) == labels.size(0) == attention_mask.size(0) <= self.max_tokens_count
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }
