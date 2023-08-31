import sys
import json

from tqdm import tqdm
from datasets import load_dataset
from fasttext import load_model as ft_load_model

from src.util.io import read_jsonl

output_path = sys.argv[1]

role_mapping = {
    "assistant": "bot",
    "prompter": "user",
    "user": "user"
}


class FasttextClassifier:
    def __init__(self, model_path):
        self.model = ft_load_model(model_path)
        self.label_offset = len("__label__")

    def __call__(self, text):
        text = text.replace("\xa0", " ").strip()
        text = " ".join(text.split())

        text_sample = text[:150]
        (label,), (prob,) = self.model.predict(text_sample, k=1)
        label = label[self.label_offset:]
        return label, prob

lang_detect = FasttextClassifier("models/lid.176.bin")

with open(output_path, "w") as w:
    for i, record in enumerate(load_dataset("OpenAssistant/oasst_top1_2023-08-25", split="train")):
        text = record["text"]
        messages = text.split("<|im_start|>")
        ru_message_count, message_count = 0, len(messages)
        fixed_messages = []
        for message in messages:
            message = message.strip()
            if not message:
                continue
            orig_role = message.split()[0]
            role = role_mapping[orig_role]
            message = message[len(orig_role):].strip()
            content = message.replace("<|im_end|>", "").strip()
            if "assistant" in content.lower():
                continue

            language, prob = lang_detect(content)
            if language == "ru" and prob > 0.4:
                ru_message_count += 1
            fixed_messages.append({"role": role, "content": content})
        messages = fixed_messages

        ru_part = ru_message_count / message_count
        if ru_part < 0.3:
            continue
        if messages[-1]["role"] == "user":
            messages = messages[:-1]
        if not messages:
            continue

        w.write(json.dumps({
            "messages": messages,
            "id": str(i)
        }, ensure_ascii=False).strip() + "\n")
