import re
import os
import json
import hashlib
import base64
from datetime import datetime, timezone

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from transliterate import translit
from datasets import DatasetDict
from datasets import load_dataset


# JSON example
# {
#    'name': 'Amaliya',
#    'description': 'Amaliya summary',
#    'personality': 'Amaliya personality',
#    'scenario': 'Amaliya scenario',
#    'first_mes': 'Amaliya greeting',
#    'mes_example': 'Amaliya example messages',
#    'metadata': {
#        'version': 1,
#        'created': 1683663135067,
#        'modified': 1683663135067,
#        'source': None,
#        'tool': {
#            'name': 'AI Character Editor',
#            'version': '0.5.0',
#            'url': 'https://zoltanai.github.io/character-editor/'
#        }
#    }
# }


def get_current_ts():
    return int(datetime.now().replace(tzinfo=timezone.utc).timestamp())


def calc_id(row, language="ru"):
    if language == "ru":
        char_id = translit(row["name"], reversed=True).lower()
    else:
        char_id = row["name"].lower()

    char_id = re.sub(r'[^\w\s]', '', char_id)
    char_id = "_".join(char_id.split(" "))
    context_hash = hashlib.md5(row["context"].encode()).hexdigest()[:5]
    char_id = char_id + "_" + context_hash
    return char_id


def add_json_to_png(row, language):
    dialogue = ""
    for message in row["example_dialogue"]:
        role = "User" if message["role"] == "user" else "Character"
        text = "{}: {}".format(role, message["content"])
        dialogue += text + "\n"
    dialogue = dialogue.strip()
    result = {
        "name": row["name"],
        "description": row["context"],
        "first_mes": row["greeting"],
        "personality": "",
        "scenario": "Random encounter",
        "mes_example": dialogue,
        "metadata": {
            "version": 1,
            "created": get_current_ts() * 1000,
            "modified": get_current_ts() * 1000,
            'source': "https://huggingface.co/datasets/IlyaGusev/gpt_roleplay_realm",
            "tool": {
                "name": "GPT Role-play Realm custom converter",
                "version": "0.0.1",
                "url": "https://github.com/IlyaGusev/rulm"
            }
        }
    }
    json_dump = json.dumps(result, ensure_ascii=False)
    b64_encoded = base64.b64encode(json_dump.encode())

    info = PngInfo()
    info.add_text("chara", b64_encoded)

    char_id = calc_id(row, language)
    row["char_id"] = char_id

    tmp_file_name = f"images/{char_id}.png"
    row["image"].save(tmp_file_name, "PNG", pnginfo=info)
    row["image"] = Image.open(tmp_file_name)
    row["image"].load()
    return row


os.makedirs("images", exist_ok=True)
ru_dataset = load_dataset("IlyaGusev/gpt_roleplay_realm", split="ru")
ru_dataset = ru_dataset.map(lambda x: add_json_to_png(x, "ru"))
en_dataset = load_dataset("IlyaGusev/gpt_roleplay_realm", split="en")
en_dataset = en_dataset.map(lambda x: add_json_to_png(x, "en"))

ddict = DatasetDict({
    "en": en_dataset,
    "ru": ru_dataset
})
ddict.push_to_hub("IlyaGusev/gpt_roleplay_realm")
