import json
from collections import Counter

import fire

from src.util.io import read_jsonl

def process_annotations(input_path, output_path):
    records = read_jsonl(input_path)
    keys = set()
    new_records = []
    counts = Counter()
    for r in records:
        if not r["preference"]:
            continue
        preference = float(r["preference"])
        if 1.4 <= preference <= 1.6:
            continue
        if r["generator_1"] == r["generator_2"]:
            continue
        instruction = r["instruction"]
        if "\n##@@##\n" not in instruction:
            continue
        prompt = instruction.split("\n##@@##\n")
        messages = []
        for m in prompt:
            if m.startswith("system:"):
                role = "system"
            elif m.startswith("user:"):
                role = "user"
            elif m.startswith("assistant:"):
                role = "assistant"
            else:
                assert False
            content = m[len(role) + 1:].strip()
            messages.append({"role": role, "content": content})
        assert len(messages) >= 2
        model_1 = r["generator_1"].replace("saiga_bot_user_multiturn_prompts_", "")
        model_2 = r["generator_2"].replace("saiga_bot_user_multiturn_prompts_", "")
        mapping = {
            "llama3_8b": "llama_3_8b",
            "saiga_llama3_8b_v4": "saiga_llama3_8b",
        }
        model_1 = mapping.get(model_1, model_1)
        model_2 = mapping.get(model_2, model_2)
        models = sorted([model_1, model_2])
        key = (instruction, models[0], models[1])
        if key in keys:
            continue
        keys.add(key)
        output_1 = r["output_1"]
        output_2 = r["output_2"]
        if output_1.strip().replace(" ", "").lower() == output_2.strip().replace(" ", "").lower():
            continue
        winning_model = model_2 if preference > 1.5 else model_1
        losing_model = model_1 if preference > 1.5 else model_2
        counts[(winning_model, losing_model)] += 1
        chosen_output = output_2 if preference > 1.5 else output_1
        rejected_output = output_1 if preference > 1.5 else output_2
        new_records.append({
            "prompt": messages, #[{"role": "user", "content": instruction}],
            "chosen": [{"role": "assistant", "content": chosen_output}],
            "rejected": [{"role": "assistant", "content": rejected_output}],
            "chosen_model": winning_model,
            "rejected_model": losing_model,
            "source": "saiga_bot_multiturn"
        })
    print(counts)
    with open(output_path, "w") as w:
        for r in new_records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(process_annotations)
