import json

import fire
from llama_cpp import Llama

from src.util.io import read_jsonl


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


def get_message_tokens(model, role, content):
    message_tokens = model.tokenize(content.encode("utf-8"))
    message_tokens.insert(1, ROLE_TOKENS[role])
    message_tokens.insert(2, LINEBREAK_TOKEN)
    message_tokens.append(model.token_eos())
    return message_tokens


def get_system_tokens(model):
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT
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
    repeat_penalty: float = 1.15
):
    model = Llama(
        model_path=model_name,
        n_ctx=n_ctx,
        n_parts=1,
    )

    records = read_jsonl(input_path)
    with open(output_path, "w") as w:
        for record in records:
            system_tokens = get_system_tokens(model)
            tokens = system_tokens
            model.eval(tokens)

            user_message = record["instruction"]
            if "input" in record and record["input"]:
                user_message += "\nДано: " + record["input"]

            print(user_message)
            print()
            message_tokens = get_message_tokens(model=model, role="user", content=user_message)
            role_tokens = [model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
            tokens += message_tokens + role_tokens
            generator = model.generate(
                tokens,
                top_k=top_k,
                top_p=top_p,
                temp=temperature,
                repeat_penalty=repeat_penalty
            )
            output = ""
            for token in generator:
                token_str = model.detokenize([token]).decode("utf-8")
                tokens.append(token)
                if token == model.token_eos():
                    break
                print(token_str, end="", flush=True)
                output += token_str
            print()
            print()
            record["answer"] = output
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
            model.reset()


if __name__ == "__main__":
    fire.Fire(infer)
