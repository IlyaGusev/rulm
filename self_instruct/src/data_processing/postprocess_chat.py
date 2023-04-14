import sys
import json

from tqdm import tqdm

from src.util.io import read_jsonl

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
                    if current_agent != "assistant":
                        is_bad_record = True
                        break
                    messages.append({
                        "role": current_agent,
                        "content": current_message.strip()
                    })
                current_agent = "user"
                line = line.replace("[Пользователь]",  "")
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
                current_agent = "assistant"
                line = line.replace("[Ассистент]",  "")
                current_message = line
            else:
                current_message += "\n" + line
        if current_message:
            messages.append({
                "role": current_agent,
                "content": current_message.strip()
            })
        if is_bad_record:
            skip_count += 1
            continue
        messages = messages[:-2]
        record["messages"] = messages
        w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
print(f"Skipped: {skip_count}")
