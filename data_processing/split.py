import argparse
import json
import random

from tqdm.auto import tqdm

from data_processing.util import read_jsonl


def split(
    input_path,
    train_path,
    validation_path,
    test_path,
    val_part,
    test_part
):
    records = read_jsonl(input_path)
    with open(train_path, "w") as train, open(validation_path, "w") as val, open(test_path, "w") as test:
        for r in tqdm(records):
            prob = random.random()
            f = train
            if prob > 1.0 - test_part:
                f = test
            elif prob > 1.0 - val_part - test_part:
                f = val
            f.write(json.dumps(r, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--validation-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--val-part", type=float, default=0.005)
    parser.add_argument("--test-part", type=float, default=0.005)
    args = parser.parse_args()
    split(**vars(args))
