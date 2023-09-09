import sys
import json

from tqdm import tqdm

from src.util.io import read_jsonl
from src.data_processing.bad_substrings import has_bad_ss

input_path = sys.argv[1]
output_path = sys.argv[2]

records = list(read_jsonl(input_path))
with open(output_path, "w") as w:
    skip_count = 0
    for record in tqdm(records):
        output = record.pop("output")
        lines = output.split("\n")
        messages = []
        current_message = ""
        current_agent = None
        is_bad_record = False
        for line in lines:
            if line.startswith("[Пользователь]"):
                if current_agent and current_message:
                    if current_agent != "bot":
                        is_bad_record = True
                        break
                    messages.append({
                        "role": current_agent,
                        "content": current_message.strip()
                    })
                current_agent = "user"
                line = line.replace("[Пользователь]", "")
                current_message = line
            elif line.startswith("[Ассистент]"):
                if current_agent and current_message:
                    if current_agent != "user":
                        is_bad_record = True
                        break
                    messages.append({
                        "role": current_agent,
                        "content": current_message.strip()
                    })
                current_agent = "bot"
                line = line.replace("[Ассистент]", "")
                current_message = line
            else:
                current_message += "\n" + line
        if current_message:
            messages.append({
                "role": current_agent,
                "content": current_message.strip()
            })

        if messages[-1]["role"] == "user":
            messages = messages[:-1]

        sum_len = 0
        for message in messages:
            assert message["role"]
            assert message["content"]
            sum_len += len(message["content"])

        if has_bad_ss(messages):
            is_bad_record = True
            skip_count += 1
            continue

        if sum_len < 750:
            is_bad_record = True
            skip_count += 1
            continue
        record["messages"] = messages
        w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
print(f"Skipped: {skip_count}")
