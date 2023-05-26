from typing import Optional

import json

import fire
from llama_cpp import Llama
from tqdm.auto import tqdm

SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."

SYSTEM_TOKEN = 1788
USER_TOKEN = 1404
BOT_TOKEN = 9225
LINEBREAK_TOKEN = 13

ROLE_TOKENS = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}


def read_jsonl(file_name):
    with open(file_name) as r:
        return [json.loads(line) for line in r]


def get_message_tokens(model, role, content):
    message_tokens = model.tokenize(content.encode("utf-8"))
    message_tokens.insert(1, ROLE_TOKENS[role])
    message_tokens.insert(2, LINEBREAK_TOKEN)
    message_tokens.append(model.token_eos())
    return message_tokens


def get_system_tokens(model):
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
    return get_message_tokens(model, **system_message)


def infer(
    model_name: str,
    input_path: str,
    output_path: str,
    n_ctx: int = 2000,
    top_k: int = 30,
    top_p: float = 0.9,
    temperature: float = 0.2,
    repeat_penalty: float = 1.15,
    max_new_tokens: Optional[int] = None,
):
    model = Llama(
        model_path=model_name,
        n_ctx=n_ctx,
        n_parts=1,
        use_mmap=False,
    )

    records = read_jsonl(input_path)
    with open(output_path, "w") as w, tqdm(records) as progress_bar:
        for record in progress_bar:
            tokens = get_system_tokens(model)[:]
            tokens.append(LINEBREAK_TOKEN)

            if "instruction" in record and "messages" not in record:
                user_message = record["instruction"]
                if "input" in record and record["input"]:
                    user_message += "\nДано: " + record["input"]
                record["messages"] = [{
                    "role": "user",
                    "content": user_message
                }]

            for message in record["messages"]:
                message_tokens = get_message_tokens(model=model, **message)
                tokens.extend(message_tokens)
                tokens.append(LINEBREAK_TOKEN)

            role_tokens = [model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
            tokens.extend(role_tokens)
            progress_bar.write(model.detokenize(tokens).decode("utf-8", "ignore"))
            generator = model.generate(
                tokens,
                top_k=top_k,
                top_p=top_p,
                temp=temperature,
                repeat_penalty=repeat_penalty,
                reset=True,
            )
            completion_tokens = []
            for i, token in enumerate(generator):
                completion_tokens.append(token)
                if token == model.token_eos() or (max_new_tokens is not None and i >= max_new_tokens):
                    break
                token_str = model.detokenize([token]).decode("utf-8", "ignore")
                progress_bar.write(token_str, end="")
            progress_bar.write('\n\n')
            record["answer"] = model.detokenize(completion_tokens).decode("utf-8")
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    fire.Fire(infer)
