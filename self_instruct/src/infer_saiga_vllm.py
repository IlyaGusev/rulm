import json

import fire
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from src.util.io import read_jsonl

SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."


def infer_saiga_vllm(
    model_name: str,
    input_path: str,
    output_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 30,
    max_tokens: int = 2048,
    repetition_penalty: float = 1.1
):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model=model_name, max_context_len_to_capture=8192)
    records = read_jsonl(input_path)
    prompts = []
    role_mapping = {
        "bot": "assistant",
        "gpt": "assistant",
        "human": "user",
    }
    actual_records = []
    for r in records:
        if "instruction" in r:
            query = r["instruction"]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]
        elif "messages" in r:
            messages = r["messages"]
            for m in messages:
                m["role"] = role_mapping.get(m["role"], m["role"])
            if messages[-1]["role"] == "assistant":
                messages = messages[:-1]
        else:
            assert False, "Wrong input format!"
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        actual_records.append(r)
        prompts.append(prompt)
    outputs = llm.generate(prompts, sampling_params)
    with open(output_path, "w") as w:
        for record, output in zip(actual_records, outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(prompt)
            print(generated_text)
            print()
            print()
            record["answer"] = generated_text.encode("utf-8").decode("utf-8", "ignore")
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    fire.Fire(infer_saiga_vllm)
