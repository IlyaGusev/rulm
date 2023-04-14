import sys
import json

from tqdm import tqdm

from src.util.io import read_jsonl

input_path = sys.argv[1]
output_path = sys.argv[2]

BAD_SS = (
    " ул. ",
    " +7",
    "Как ИИ",
    "как ИИ",
    "Как модель ИИ",
    "как модель ИИ",
    "как языковая модель ИИ",
    "Как языковая модель ИИ",
    "как искусственный интеллект",
    "OpenAI",
    "ChatGPT",
    "as a language model"
)

records = list(read_jsonl(input_path))
with open(output_path, "w") as w:
    skip_count = 0
    for record in tqdm(records):
        output = record.pop("output")
        is_bad_record = False
        if any(ss in output for ss in BAD_SS):
            skip_count += 1
            continue
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
                current_agent = "bot"
                line = line.replace("[Ассистент]",  "")
                current_message = line
            else:
                current_message += "\n" + line
        if current_message:
            messages.append({
                "role": current_agent,
                "content": current_message.strip()
            })
        messages = messages[:-2]
        for message in messages:
            assert message["role"]
            assert message["content"]
            if message["role"] == "bot" and len(message["content"]) < 150:
                is_bad_record = True
                break
            
        if is_bad_record:
            skip_count += 1
            continue
        record["messages"] = messages
        w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
print(f"Skipped: {skip_count}")
