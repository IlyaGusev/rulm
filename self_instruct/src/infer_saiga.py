import sys
import json
import random
from tqdm import tqdm

import fire
import torch
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from peft import PeftConfig, PeftModel

from src.util.io import read_jsonl
from src.util.chat import Conversation
from src.util.dl import gen_batch


def generate(model, tokenizer, prompts, generation_config):
    data = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation_config
    )
    outputs = []
    for sample_output_ids, sample_input_ids in zip(output_ids, data["input_ids"]):
        sample_output_ids = sample_output_ids[len(sample_input_ids):]
        sample_output = tokenizer.decode(sample_output_ids, skip_special_tokens=True)
        sample_output = sample_output.replace("</s>", "").strip()
        outputs.append(sample_output)
    return outputs


def generate_answers(
    model_name: str,
    template_path: str,
    input_path: str,
    output_path: str,
    batch_size: int = 1
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if batch_size > 1:
        assert tokenizer.padding_side == "left", "Batched inference for right padding side is impossible"
    generation_config = GenerationConfig.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        config = PeftConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(
            model,
            model_name,
            torch_dtype=torch.float16
        )
    elif device == "cpu":
        config = PeftConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            device_map={"": device},
            low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            model_name,
            device_map={"": device}
        )

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    records = read_jsonl(input_path)
    prompts = []
    for record in records:
        conversation = Conversation.from_template(template_path)
        user_message = record["instruction"]
        if "input" in record and record["input"]:
            user_message += "\nДано: " + record["input"]
        conversation.add_user_message(user_message)
        prompt = conversation.get_prompt(tokenizer)
        prompts.append(prompt)

    all_outputs = []
    for batch in tqdm(gen_batch(prompts, batch_size)):
        outputs = generate(
            model=model,
            tokenizer=tokenizer,
            prompts=batch,
            generation_config=generation_config
        )
        for prompt, output in zip(batch, outputs):
            print(prompt)
            print(output)
            all_outputs.append(output)

    with open(output_path, "w") as w:
        for record, output in zip(records, all_outputs):
            record["answer"] = output
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    fire.Fire(generate_answers)
