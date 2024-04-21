import fire
from llama_cpp import Llama

SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."


def interact(
    model_path,
    n_ctx=8192,
    top_k=30,
    top_p=0.9,
    temperature=0.6,
    repeat_penalty=1.1
):
    model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_parts=1,
        verbose=True,
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        user_message = input("User: ")
        messages.append({"role": "user", "content": user_message})
        for part in model.create_chat_completion(
            messages,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stream=True,
        ):
            delta = part["choices"][0]["delta"]
            if "content" in delta:
                print(delta["content"], end="", flush=True)
        print()


if __name__ == "__main__":
    fire.Fire(interact)
