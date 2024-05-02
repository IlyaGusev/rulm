import json

import fire
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from src.util.io import read_jsonl

def infer_saiga_vllm(
    model_name: str,
    input_path: str,
    output_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 30,
    max_tokens: int = 2048,
    repetition_penalty: float = 1.12
):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model=model_name)
    records = read_jsonl(input_path)
    prompts = []
    for r in records:
        messages = [{"role": "user", "content": r["instruction"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    outputs = llm.generate(prompts, sampling_params)
    with open(output_path, "w") as w:
        for record, output in zip(records, outputs):
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
