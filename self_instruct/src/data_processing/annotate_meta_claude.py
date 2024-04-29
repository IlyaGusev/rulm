import json
import random
import fire
import shutil
import os
import traceback

from src.util.io import read_jsonl, write_jsonl
from src.anthropic_wrapper import anthropic_completion

PROMPT = """Choose one of the following topics for user requests in the following conversation. Evaluate the complexity of user requests.

Topics:
- "facts": questions and answers about some established facts
- "writing": conversations about writing stories, novels, poems, jokes, letters and all kinds of texts
- "extract": extracting information from a text: responding with structural information, summarizing, text-based QA
- "translation": translating from one language to another
- "languages": questions about specifics of languages
- "coding": code writing and debugging, including building web and mobile apps
- "math": questions about solving mathematical problems
- "stem": science, technology, and engineering questions (excluding "math" and "coding")
- "humanities": academic questions that do not fall into "math" or "stem" categories
- "psychology": conversations about user psychological well-being
- "finance": conversations about the economy and managing money
- "travel": questions about tourism and traveling
- "entertainment": questions about movies, fiction books, TV series, computer games
- "ecology": questions about ecology and natural disasters
- "cooking": questions about food and cooking
- "health": questions about health and medical matters
- "law": questions about laws and law practice
- "career": questions about career, CV, resume
- "reasoning": different riddles and questions that are supposed to be tricky and should be solved with common sense
- "brainstorm": conversations about coming up with new ideas
- "roleplay": conversations about pretending to be someone
- "inappropriate": inappropriate questions
- "chit_chat": inconsequential conversation without a specific topic
- "other": all conversations that do not fall into one of the above categories

Do not come up with your own topics!
Select only one topic from the list!
Consider only user requests!

Rate complexity on a scale with 3 values: "easy", "medium", "hard".
"hard" requests can not be responded by an average human, and almost any human should be able to solve "easy" ones.

Conversation:
####
{conversation}
####

First provide an explanation of your decisions in English.
Return only a JSON with the following format: {{"topic_explanation": "...", "topic": "...", "complexity_explanation": "...", "complexity": "..."}}.
"""

def to_conversation(r):
    return "\n\n".join([f'{m["role"]}: {m["content"]}' for m in r["messages"]])


def annotate_meta_claude(
    input_path: str,
    output_path: str,
    model_name: str = "claude-3-sonnet-20240229"
):
    records = read_jsonl(input_path)
    output_records = []
    existing_conversations = set()
    if os.path.exists(output_path):
        output_records = read_jsonl(output_path)
        for r in output_records:
            existing_conversations.add(to_conversation(r))
    random.shuffle(records)
    for i, r in enumerate(records):
        conversation = to_conversation(r)
        if conversation in existing_conversations:
            print(f"Skipping {i}")
            continue
        for _ in range(3):
            try:
                prompt = PROMPT.format(conversation=conversation)
                answer = anthropic_completion(
                    [{"role": "user", "content": prompt}],
                    model_name=model_name
                )
                print(prompt)
                print(answer)
                print()
                print()
                start_index = answer.find("{")
                end_index = answer.rfind("}")
                answer = answer[start_index : end_index + 1]
                answer = json.loads(answer)
                assert isinstance(answer, dict)
                assert "topic" in answer
                break
            except Exception:
                traceback.print_exc()
                continue
        r["topics_answer"] = answer
        output_records.append(r)
        write_jsonl(output_records, output_path + "_tmp")
        shutil.move(output_path + "_tmp", output_path)


if __name__ == "__main__":
    fire.Fire(annotate_meta_claude)
