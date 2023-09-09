import sys
import json

from tqdm import tqdm

from src.util.io import read_jsonl

input_path = sys.argv[1]
output_path = sys.argv[2]

role_mapping = {
    "assistant": "bot",
    "prompter": "user"
}

with open(output_path, "w") as w:
    records = tqdm(read_jsonl(input_path))
    for record in records:
        if record["prompt"]["lang"] != "ru":
            continue
        role = record["prompt"]["role"]
        content = record["prompt"]["text"]
        messages = [{"role": role, "content": content}]

        def collect_replies(current_replies):
            if not current_replies:
                return []
            best_reply = current_replies[0]
            if best_reply["synthetic"]:
                return []
            role = best_reply["role"]
            text = best_reply["text"]
            replies = best_reply["replies"]
            messages = [{"role": role, "content": text}]
            messages += collect_replies(replies)
            return messages

        replies = record["prompt"]["replies"]
        messages += collect_replies(replies)
        for m in messages:
            m["role"] = role_mapping[m.pop("role")]
            m["content"] = m["content"].replace("Ася", "Бот")

        if messages[-1]["role"] == "user":
            messages = messages[:-1]
        w.write(json.dumps({
            "messages": messages,
            "id": record["message_tree_id"]
        }, ensure_ascii=False).strip() + "\n")
