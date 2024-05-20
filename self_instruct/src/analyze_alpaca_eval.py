import json
from collections import Counter, defaultdict

import fire


def analyze(input_path, tasks_path):
    with open(input_path) as r:
        annotations = json.load(r)
    with open(tasks_path) as r:
        tasks = [json.loads(line) for line in r]
    tasks = {task["instruction"]: task for task in tasks}
    wins_count = Counter()
    wins_by_topics = defaultdict(Counter)
    models = set()
    for r in annotations:
        preference = r["preference"]
        generator_1 = r["generator_1"]
        models.add(generator_1)
        output_1 = r["output_1"]
        generator_2 = r["generator_2"]
        models.add(generator_2)
        output_2 = r["output_2"]
        winning_model = generator_1 if preference < 1.5 else generator_2
        wins_count[winning_model] += 1
        instruction = r["instruction"]
        task = tasks[instruction]
        topic = task["category"]
        wins_by_topics[topic][winning_model] += 1

    models = list(models)
    print("Total wins:")
    for model in models:
        print(model, wins_count[model])
    print()
    for topic, cnts in wins_by_topics.items():
        print(topic)
        for model in models:
            print(model, cnts[model])
        print()



fire.Fire(analyze)
