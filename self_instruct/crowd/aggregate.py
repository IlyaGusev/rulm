import argparse
import os
from collections import defaultdict, Counter

import toloka.client as toloka
from nltk.metrics.agreement import AnnotationTask
from crowdkit.aggregation import DawidSkene
import pandas as pd

from src.util.io import write_jsonl


def get_key(record):
    return (record["instruction"], record["left_model"], record["right_model"])


def get_pool(pool_id, toloka_client):
    records = []
    for assignment in toloka_client.get_assignments(pool_id=pool_id):
        solutions = assignment.solutions
        if not solutions:
            continue
        for task, solution in zip(assignment.tasks, solutions):
            known_solutions = task.known_solutions
            if known_solutions is not None:
                continue
            input_values = task.input_values
            output_values = solution.output_values
            record = {
                "worker_id": assignment.user_id,
                "assignment_id": assignment.id
            }
            record.update(input_values)
            record.update(output_values)
            records.append(record)
    return records


def aggregate(records, overlap=5, min_agreement=0.0):
    results = defaultdict(list)
    records.sort(key=lambda x: x["assignment_id"])
    for r in records:
        results[get_key(r)].append(r["result"])

    for key, votes in results.items():
        results[key] = votes[:overlap]

    data = {get_key(r): r for r in records}
    votes_distribution = Counter()
    res_distribution = Counter()
    votes = dict()
    for key, res in results.items():
        res_count = Counter(res)
        overlap = len(res)
        res_win, votes_win = res_count.most_common(1)[0]
        res_distribution[res_win] += 1
        votes_part = float(votes_win) / overlap
        votes_distribution[votes_part] += 1
        votes[key] = votes_part
        data[key].update({
            "result": res_win,
            "agreement": votes_part
        })

    answers = [(str(hash(get_key(r))), r["result"], r["worker_id"]) for r in records]
    answers_df = pd.DataFrame(answers, columns=["task", "label", "worker"])
    proba = DawidSkene(n_iter=20).fit_predict_proba(answers_df)
    labels = proba.idxmax(axis=1)
    for key in data:
        ds_key = str(hash(key))
        label = labels[ds_key]
        confidence = proba.loc[ds_key, label]
        data[key].update({
            "ds_result": label,
            "ds_confidence": confidence
        })

    print("Aggregation: ")
    total_samples = sum(votes_distribution.values())
    sum_agreement = 0
    for v, sample_count in sorted(votes_distribution.items(), reverse=True):
        print("{}: {}".format(v, sample_count))
        sum_agreement += sample_count * v
    print("Total: ", total_samples)
    print("Average agreement:", sum_agreement / total_samples)
    print("Results: ")
    for res, cnt in res_distribution.items():
        print("{}: {}".format(res, cnt))

    answers = [(r["worker_id"], get_key(r), r["result"]) for r in records]
    t = AnnotationTask(data=answers)
    print("Krippendorff’s alpha: {}".format(t.alpha()))

    answers = [
        (r["worker_id"], get_key(r), r["result"])
        for r in records if votes[get_key(r)] >= min_agreement
    ]
    t = AnnotationTask(data=answers)
    print("Krippendorff’s alpha, border {}: {}".format(min_agreement, t.alpha()))
    print()

    data = {key: r for key, r in data.items()}
    return data


def main(
    token,
    agg_output,
    raw_output,
    pools_file,
    input_fields,
):
    input_fields = input_fields.split(",")

    with open(os.path.expanduser(token), "r") as r:
        toloka_token = r.read().strip()
    toloka_client = toloka.TolokaClient(toloka_token, 'PRODUCTION')

    pool_ids = []
    with open(pools_file, "r") as r:
        for line in r:
            pool_id = line.strip()
            if not pool_id:
                continue
            pool_id = int(pool_id)
            pool_ids.append(pool_id)

    records = []
    for pool_id in pool_ids:
        pool = get_pool(pool_id, toloka_client)
        records.extend(pool)

    agg_records = aggregate(records)
    agg_records = list(agg_records.values())
    agg_records.sort(key=lambda x: (x["agreement"], x["left_model"]), reverse=True)
    agg_header = ["result", "agreement", "ds_result", "ds_confidence"]
    agg_header += input_fields
    agg_records = [{key: r[key] for key in agg_header} for r in agg_records]
    write_jsonl(agg_records, agg_output)

    label_mapping = {
        "left": 1,
        "right": -1,
        "equal": 0,
        "incorrect": 0
    }

    answers = defaultdict(list)
    for record in records:
        key = get_key(record)
        answers[key].append(record["result"])

    agg_scores = defaultdict(lambda: Counter())
    for key, labels in answers.items():
        _, a_model, b_model = key
        if a_model > b_model:
            a_model, b_model = b_model, a_model
            mapping = {
                "left": "right",
                "right": "left",
            }
            labels = [mapping.get(label, label) for label in labels]

        labels = [label_mapping[label] for label in labels]
        result = min(max(sum(labels), -1), 1)
        if result == 1:
            agg_scores[(a_model, b_model)][a_model] += 1
        elif result == -1:
            agg_scores[(a_model, b_model)][b_model] += 1
        elif result == 0:
            agg_scores[(a_model, b_model)]["equal"] += 1
    results = []
    for cnt in agg_scores.values():
        counts = dict(list(cnt.items()))
        models = sorted(list(set(counts.keys()) - {"equal"}), reverse=True)
        result = "{} vs {}: ".format(*models)
        models_plus = [models[0]] + ["equal"] + [models[1]]
        result += "-".join([str(counts[model]) for model in models_plus])
        results.append(result)

    for result in sorted(results):
        print(result)

    raw_records = records
    raw_header = ["result", "worker_id", "assignment_id"] + input_fields
    raw_records = [{key: r[key] for key in raw_header} for r in raw_records]
    write_jsonl(raw_records, raw_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-fields", type=str, default="instruction,input,left_answer,right_answer,left_model,right_model,id")
    parser.add_argument("--token", type=str, default="~/.toloka/nyan_token")
    parser.add_argument("--agg-output", type=str, required=True)
    parser.add_argument("--raw-output", type=str, required=True)
    parser.add_argument("--pools-file", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))

