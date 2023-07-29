import sys

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel, PeftConfig

from src.util.chat import Conversation
from src.util.load import load_saiga

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
    output = tokenizer.decode(output_ids).replace("</s>", "").strip()
    return output


def interact(model_name, template_path):
    model, tokenizer, generation_config = load_saiga(model_name)
    conversation = Conversation.from_template(template_path)
    while True:
        user_message = input("User: ")
        if user_message.strip() == "/reset":
            conversation = Conversation.from_template(template_path)
            print("History reset completed!")
            continue
        conversation.add_user_message(user_message)
        prompt = conversation.get_prompt(tokenizer)
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            generation_config=generation_config,
            eos_token_id=conversation.get_end_token_id()
        )
        conversation.add_bot_message(output)
        print("Saiga:", output)


if __name__ == "__main__":
    fire.Fire(interact)
