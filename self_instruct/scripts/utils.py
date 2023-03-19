import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import openai
import copy


openai_org = os.getenv("OPENAI_ORG")
if openai_org is not None:
    openai.organization = openai_org
    logging.warning(f"Switching to organization: {openai_org} for OAI API key.")


@dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 2560
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


def openai_completion(
    messages,
    decoding_args: OpenAIDecodingArguments,
    model_name="gpt-3.5-turbo",
    sleep_time=2
):
    assert decoding_args.n == 1
    completions = []
    sample_decoding_args = copy.deepcopy(decoding_args)
    while True:
        try:
            completion = openai.ChatCompletion.create(
                messages=messages,
                model=model_name,
                **sample_decoding_args.__dict__
            )
            completions = completion.choices
            break
        except openai.error.OpenAIError as e:
            logging.warning(f"OpenAIError: {e}.")
            if "Please reduce" in str(e):
                sample_decoding_args.max_tokens = int(sample_decoding_args.max_tokens * 0.8)
                logging.warning(f"Reducing target length to {sample_decoding_args.max_tokens}, Retrying...")
            else:
                logging.warning("Hit request rate limit; retrying...")
                time.sleep(sleep_time)
    return completions[0]
