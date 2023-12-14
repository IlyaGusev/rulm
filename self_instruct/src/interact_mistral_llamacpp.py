import fire
from llama_cpp import Llama

SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."


def get_message_tokens(model, role, content):
    content = f"{role}\n{content}\n</s>"
    content = content.encode("utf-8")
    message_tokens = model.tokenize(content, special=True)
    return message_tokens


def get_system_tokens(model):
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
    return get_message_tokens(model, **system_message)


def interact(
    model_path,
    n_ctx=2000,
    top_k=30,
    top_p=0.9,
    temperature=0.2,
    repeat_penalty=1.1
):
    model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_parts=1,
    )

    system_tokens = get_system_tokens(model)
    tokens = system_tokens
    model.eval(tokens)

    while True:
        user_message = input("User: ")
        message_tokens = get_message_tokens(model=model, role="user", content=user_message)
        role_tokens = model.tokenize("bot\n".encode("utf-8"), special=True)
        tokens += message_tokens + role_tokens
        full_prompt = model.detokenize(tokens)
        generator = model.generate(
            tokens,
            top_k=top_k,
            top_p=top_p,
            temp=temperature,
            repeat_penalty=repeat_penalty
        )
        for token in generator:
            token_str = model.detokenize([token]).decode("utf-8", errors="ignore")
            tokens.append(token)
            if token == model.token_eos():
                break
            print(token_str, end="", flush=True)
        print()


if __name__ == "__main__":
    fire.Fire(interact)
