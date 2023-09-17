import copy
import json
from tqdm import tqdm

import fire
from transformers import AutoTokenizer

from src.util.io import read_jsonl
from src.util.chat import Conversation
from src.util.dl import gen_batch
from src.util.load import load_saiga
from src.util.generate import generate


def generate_answers(
    model_name: str,
    template_path: str,
    input_path: str,
    output_path: str,
    batch_size: int = 1,
    use_4bit: bool = False,
    torch_dtype: str = None,
    is_lora: bool = True
):
    model, tokenizer, generation_config = load_saiga(
        model_name,
        use_4bit=use_4bit,
        torch_dtype=torch_dtype,
        is_lora=is_lora
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if batch_size > 1:
        assert tokenizer.padding_side == "left", "Batched inference for right padding side is impossible"
    records = read_jsonl(input_path)

    default_conversation = Conversation.from_template(template_path)
    with open(output_path, "w") as w:
        for batch in tqdm(gen_batch(records, batch_size)):
            prompts = []
            for record in batch:
                conversation = copy.deepcopy(default_conversation)
                user_message = record["instruction"]
                if "input" in record and record["input"]:
                    user_message += "\nДано: " + record["input"]
                conversation.add_user_message(user_message)
                prompt = conversation.get_prompt(tokenizer)
                prompts.append(prompt)
            outputs = generate(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                generation_config=generation_config
            )
            for record, prompt, output in zip(batch, prompts, outputs):
                print(prompt)
                print(output)
                record["instruction"] = record["instruction"].encode("utf-8").decode("utf-8", "ignore")
                if "input" in record and record["input"]:
                    record["input"] = record["input"].encode("utf-8").decode("utf-8", "ignore")
                record["answer"] = output.encode("utf-8").decode("utf-8", "ignore")
                w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    fire.Fire(generate_answers)
