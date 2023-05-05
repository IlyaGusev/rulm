import json
import os
import shutil

import fire
from jinja2 import Template
from tqdm import tqdm

from src.util.io import read_jsonl, write_jsonl
from src.util.openai import openai_batch_completion, OpenAIDecodingArguments


def encode_prompt(char, topic, template_path):
    with open(template_path) as f:
        template = Template(f.read())
    fields = ("name", "context", "greeting", "example_dialogue")
    char = {k: v for k, v in char.items() if k in fields}
    return template.render(
        char_json=json.dumps(char, ensure_ascii=False),
        topic=topic
    ).strip() + "\n"


def get_char_key(char):
    return (char["name"].strip(), char["context"].strip())


def get_dialogue_key(char, topic):
    return (char["name"].strip(), char["context"].strip(), topic)


def parse_chat(result):
    try:
        chat = json.loads(result)
    except Exception:
        print("Incorrect JSON:", result)
        return None

    if isinstance(chat, dict):
        keys = list(chat.keys())
        if len(keys) > 1:
            print("Too many keys:", result)
            return None
        key = keys[0]
        chat = chat[key]
    if not isinstance(chat, list):
        print("Not a list:", chat)
        return None

    prev_role = None
    for message in chat:
        if not "role" in message:
            print("No role in message:", message)
            return None
        if not "content" in message:
            print("No content in message:", message)
            return None
        if message["role"] not in ("user", "char"):
            print("Incorrect role:", message)
            return None
        if message["role"] == prev_role:
            print("Two messages from the same role:", chat)
            return None
        prev_role = message["role"]
    return chat


def process_batch(batch, model_name, template_path):
    prompts = [[
        {"role": "user", "content": encode_prompt(char, topic, template_path)}
    ] for char, topic in batch]
    results = openai_batch_completion(
        batch=prompts,
        model_name=model_name,
        decoding_args=OpenAIDecodingArguments(
            max_tokens=3074
        )
    )

    output_chars = dict()
    for (char, topic), prompt, result in zip(batch, prompts, results):
        result = result.message["content"]
        print(prompt[-1]["content"])
        print(result)
        print()
        print("=============")
        print()
        chat = parse_chat(result)
        if chat is None:
            continue
        chat = {
            "topic": topic,
            "chat": chat,
            "model_name": model_name
        }
        if "dialogues" not in char:
            char["dialogues"] = []
        char["dialogues"].append(chat)
        output_chars[get_char_key(char)] = char
    return output_chars


def main(
    chars_path,
    output_path,
    template_path,
    request_batch_size=4
):
    existing_keys = set()
    output_records = dict()
    if os.path.exists(output_path):
        with open(output_path) as f:
            output_records = [json.loads(line) for line in f]
            for record in output_records:
                for dialogue in record["dialogues"]:
                    topic = dialogue["topic"]
                    existing_keys.add(get_dialogue_key(record, topic))
            output_records = {get_char_key(char): char for char in output_records}
    print(f"Existing keys: {len(existing_keys)}")

    batch = []
    chars = read_jsonl(chars_path)
    for idx, char in enumerate(chars):
        key = get_char_key(char)
        if key in output_records:
            chars[idx] = output_records[key]

    for char in tqdm(chars):
        topics = char["topics"]

        # GPT-4 for the first topic
        first_topic = topics[0]
        key = get_dialogue_key(char, first_topic)
        if key in existing_keys:
            print(f"Skipping {key}")
        else:
            updated_chars = process_batch(
                [(char, first_topic)], "gpt-4", template_path
            )
            for key, char in updated_chars.items():
                output_records[key] = char

        # GPT-3.5 for all other topics
        for topic in topics[1:]:
            key = get_dialogue_key(char, topic)
            if key in existing_keys:
                print(f"Skipping {key}")
                continue
            batch.append((char, topic))
            if len(batch) != request_batch_size:
                continue
            updated_chars = process_batch(batch, "gpt-3.5-turbo", template_path)
            for key, char in updated_chars.items():
                output_records[key] = char
            batch = []
            write_jsonl(list(output_records.values()), output_path + "_tmp")
            shutil.move(output_path + "_tmp", output_path)

    if batch:
        updated_chars = process_batch(batch, "gpt-3.5-turbo", template_path)
        for key, char in updated_chars.items():
            output_records[key] = char
        write_jsonl(list(output_records.values()), output_path + "_tmp")
        shutil.move(output_path + "_tmp", output_path)


if __name__ == "__main__":
    fire.Fire(main)
