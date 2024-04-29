import logging
import time
import os
import json
import inspect
from typing import Optional

from anthropic import Anthropic, APIError

DEFAULT_MODEL = "claude-3-haiku-20240307"
DEFAULT_SLEEP_TIME = 20


def anthropic_completion(
    messages,
    model_name: str = DEFAULT_MODEL,
    sleep_time: int = DEFAULT_SLEEP_TIME,
    api_key: Optional[str] = None,
    max_tokens: int = 2048,
    **kwargs,
):
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", None)

    system_message = ""
    if messages[0]["role"] == "system":
        system_message = messages[0]["content"]
        messages = messages[1:]

    while True:
        try:
            client = Anthropic(api_key=api_key)
            completion = client.messages.create(
                system=system_message,
                messages=messages,
                model=model_name,
                max_tokens=max_tokens,
                **kwargs,
            )
            break
        except APIError as e:
            logging.warning(f"Anthropic error: {e}.")
            time.sleep(sleep_time)
    return completion.content[0].text


def anthropic_tokenize(text: str, api_key: Optional[str] = None):
    client = Anthropic(api_key)
    tokenizer = client.get_tokenizer()
    return tokenizer.encode(text)


def anthropic_list_models():
    models = (
        inspect.signature(Anthropic().messages.create).parameters["model"].annotation
    )
    models = models[models.find("Literal") + len("Literal"): -1]
    models = models.replace("'", '"')
    models = json.loads(models)
    return models


def anthropic_get_key(model_settings):
    env_key = os.getenv("ANTHROPIC_API_KEY", None)
    local_key = model_settings.anthropic_api_key
    if local_key:
        return local_key
    return env_key
