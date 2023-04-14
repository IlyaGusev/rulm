import sys

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel, PeftConfig

from src.util.chat import Conversation


def generate(model, tokenizer, prompt, generation_config, eos_token_id: int = None):
    data = tokenizer(prompt, return_tensors="pt")
    data = {k: v.to(model.device) for k, v in data.items()}
    if eos_token_id is not None:
        generation_config.eos_token_id = eos_token_id
    output_ids = model.generate(
        **data,
        generation_config=generation_config
    )[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids)
    return output


def interact(model_name, template_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generation_config = GenerationConfig.from_pretrained(model_name)

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

    conversation = Conversation.from_template(template_path)
    while True:
        user_message = input("User: ")
        if user_message.strip() == "/reset":
            conversation = Conversation.from_template(template_path)
            print("History reset completed!")
            continue
        conversation.add_user_message(user_message)
        prompt = conversation.get_prompt()
        prompt += "\n<start>"
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            generation_config=generation_config,
            eos_token_id=conversation.get_end_token_id()
        )
        output = output.replace("<end", "").replace("bot", "").strip()
        conversation.add_bot_message(output)
        print("Saiga:", output)


if __name__ == "__main__":
    fire.Fire(interact)
