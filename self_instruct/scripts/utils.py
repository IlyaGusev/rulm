import logging
import os
import json
import time
import random
from dataclasses import dataclass
from typing import Optional, Sequence
from multiprocessing.pool import ThreadPool

import torch
import numpy as np
import openai
import copy


@dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 2560
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


def openai_completion(
    messages,
    decoding_args,
    model_name,
    sleep_time
):
    decoding_args = copy.deepcopy(decoding_args)
    assert decoding_args.n == 1
    while True:
        try:
            completions = openai.ChatCompletion.create(
                messages=messages,
                model=model_name,
                **decoding_args.__dict__
            )
            break
        except openai.error.OpenAIError as e:
            logging.warning(f"OpenAIError: {e}.")
            if "Please reduce" in str(e):
                decoding_args.max_tokens = int(decoding_args.max_tokens * 0.8)
                logging.warning(f"Reducing target length to {decoding_args.max_tokens}, Retrying...")
            else:
                logging.warning("Hit request rate limit; retrying...")
                time.sleep(sleep_time)
    return completions.choices[0]


def openai_batch_completion(
    batch,
    decoding_args: OpenAIDecodingArguments,
    model_name="gpt-3.5-turbo",
    sleep_time=2
):
    completions = []
    with ThreadPool(len(batch)) as pool:
        results = pool.starmap(openai_completion, [
            (messages, decoding_args, model_name, sleep_time) for messages in batch
        ])
        for result in results:
            completions.append(result)
    return completions


def read_jsonl(file_name):
    with open(file_name) as r:
        return [json.loads(line) for line in r]


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fix_tokenizer(tokenizer):
    # Fixing broken tokenizers
    special_tokens = dict()
    for token_id in range(1000):
        token = tokenizer.convert_ids_to_tokens(token_id)
        if tokenizer.pad_token_id in (None, tokenizer.vocab_size) and "pad" in token:
            special_tokens["pad_token"] = token
        if tokenizer.bos_token_id in (None, tokenizer.vocab_size) and "<s>" in token:
            special_tokens["bos_token"] = token
        if tokenizer.eos_token_id in (None, tokenizer.vocab_size) and "</s>" in token:
            special_tokens["eos_token"] = token
        if tokenizer.unk_token_id in (None, tokenizer.vocab_size) and "unk" in token:
            special_tokens["unk_token"] = token
        if tokenizer.sep_token_id in (None, tokenizer.vocab_size) and "sep" in token:
            special_tokens["sep_token"] = token

    if tokenizer.sep_token_id in (None, tokenizer.vocab_size) and "bos_token" in special_tokens:
        special_tokens["sep_token"] = special_tokens["bos_token"]

    if tokenizer.pad_token_id in (None, tokenizer.vocab_size) and "pad_token" not in special_tokens:
        if tokenizer.unk_token_id is not None:
            special_tokens["pad_token"] = tokenizer.unk_token
        else:
            special_tokens["pad_token"] = "<|pad|>"

    if tokenizer.sep_token_id in (None, tokenizer.vocab_size) and "sep_token" not in special_tokens:
        if tokenizer.bos_token_id is not None:
            special_tokens["sep_token"] = tokenizer.bos_token
        else:
            special_tokens["sep_token"] = "<|sep|>"

    tokenizer.add_special_tokens(special_tokens)

    print("Vocab size: ", tokenizer.vocab_size)
    print("PAD: ", tokenizer.pad_token_id, tokenizer.pad_token)
    print("BOS: ", tokenizer.bos_token_id, tokenizer.bos_token)
    print("EOS: ", tokenizer.eos_token_id, tokenizer.eos_token)
    print("UNK: ", tokenizer.unk_token_id, tokenizer.unk_token)
    print("SEP: ", tokenizer.sep_token_id, tokenizer.sep_token)
    return tokenizer


def fix_model(model, tokenizer, use_resize=True):
    model.config.pad_token_id = tokenizer.pad_token_id
    assert model.config.pad_token_id is not None

    bos_candidates = (
        tokenizer.bos_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.unk_token_id
    )
    for bos_candidate in bos_candidates:
        model.config.bos_token_id = bos_candidate
        if bos_candidate is not None:
            break
    assert model.config.bos_token_id is not None
    model.config.decoder_start_token_id = model.config.bos_token_id

    eos_candidates = (tokenizer.eos_token_id, tokenizer.sep_token_id)
    for eos_candidate in eos_candidates:
        model.config.eos_token_id = eos_candidate
        if eos_candidate is not None:
            break
    assert model.config.eos_token_id is not None

    if use_resize:
        model.resize_token_embeddings(len(tokenizer))

    return model


DEFAULT_MESSAGE_TEMPLATE = "<start>{role}\n{content} <end>\n",
DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
DEFAULT_START_TOKEN_ID = 2962
DEFAULT_END_TOKEN_ID = 355
DEFAULT_BOT_TOKEN_ID = 7451


class Conversation:
    def __init__(
        self,
        message_template=DEFAULT_MESSAGE_TEMPLATE,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        role_mapping=None,
        start_token_id=DEFAULT_START_TOKEN_ID,
        end_token_id=DEFAULT_END_TOKEN_ID,
        bot_token_id=DEFAULT_BOT_TOKEN_ID
    ):
        self.message_template = message_template
        self.role_mapping = role_mapping or {}
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.bot_token_id = bot_token_id
        self.messages = [{
            "role": "system",
            "content": system_prompt
        }]

    def get_end_token_id(self):
        return self.end_token_id

    def get_start_token_id(self):
        return self.start_token_id

    def get_bot_token_id(self):
        return self.bot_token_id

    def add_user_message(self, message):
        self.messages.append({
            "role": "user",
            "content": message
        })

    def add_bot_message(self, message):
        self.messages.append({
            "role": "bot",
            "content": message
        })

    def get_prompt(self):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        return final_text.strip()

    @classmethod
    def from_template(cls, file_name):
        with open(file_name) as r:
            template = json.load(r)
        return Conversation(
            **template
        )

    def expand(self, messages):
        for message in messages:
            self.messages.append({
                "role": self.role_mapping.get(message["role"], message["role"]),
                "content": message["content"]
            })
