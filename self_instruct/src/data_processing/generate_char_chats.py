import json
import copy
import os
import shutil
from collections import defaultdict

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
        if "role" not in message:
            print("No role in message:", message)
            return None
        if "content" not in message:
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
    print("Processing batch...")
    print([r["name"] for (r, topic) in batch])
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

    dialogues = defaultdict(list)
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
        key = get_char_key(char)
        dialogues[key].append(chat)
    return dialogues


def fix_output_records(records):
    for char in records:
        unique_dialogues = dict()
        topics = char["topics"]
        if "dialogues" in char:
            for dialogue in char["dialogues"]:
                topic = dialogue["topic"]
                if topic in topics:
                    unique_dialogues[dialogue["topic"]] = dialogue
            char["dialogues"] = list(unique_dialogues.values())
    return records


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
            output_records = fix_output_records(output_records)
            for record in output_records:
                if "dialogues" in record:
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
            chars[idx] = copy.deepcopy(output_records[key])

    key2idx = {get_char_key(char): idx for idx, char in enumerate(chars)}

    def add_dialogues(dialogues):
        for key, char_dialogues in dialogues.items():
            idx = key2idx[key]
            char = chars[idx]
            if "dialogues" not in char:
                char["dialogues"] = []
            char["dialogues"].extend(char_dialogues)

    for char in tqdm(chars):
        topics = char["topics"]

        # GPT-4 for the first topic
        first_topic = topics[0]
        key = get_dialogue_key(char, first_topic)
        if key not in existing_keys:
            dialogues = process_batch([(char, first_topic)], "gpt-4", template_path)
            add_dialogues(dialogues)

        # GPT-3.5 for all other topics
        for topic in topics[1:]:
            key = get_dialogue_key(char, topic)
            if key in existing_keys:
                print(f"Skipping {key}")
                continue
            batch.append((char, topic))
            if len(batch) != request_batch_size:
                continue
            dialogues = process_batch(batch, "gpt-3.5-turbo", template_path)
            add_dialogues(dialogues)
            batch = []

            write_jsonl(chars, output_path + "_tmp")
            shutil.move(output_path + "_tmp", output_path)

    if batch:
        dialogues = process_batch(batch, "gpt-3.5-turbo", template_path)
        add_dialogues(dialogues)

    write_jsonl(chars, output_path + "_tmp")
    shutil.move(output_path + "_tmp", output_path)


if __name__ == "__main__":
    fire.Fire(main)
