import logging
import time
from dataclasses import dataclass
from typing import Optional, Sequence
from multiprocessing.pool import ThreadPool

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

DEFAULT_ARGS = OpenAIDecodingArguments()

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
    decoding_args: OpenAIDecodingArguments = DEFAULT_ARGS,
    model_name="gpt-3.5-turbo",
    sleep_time=20
):
    completions = []
    with ThreadPool(len(batch)) as pool:
        results = pool.starmap(openai_completion, [
            (messages, decoding_args, model_name, sleep_time) for messages in batch
        ])
        for result in results:
            completions.append(result)
    return completions
