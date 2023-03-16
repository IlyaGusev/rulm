import json
import os
import io

import zstandard
import jsonlines
import datasets

try:
    import simdjson
    parser = simdjson.Parser()
    def parse_json(x):
        try:
            return parser.parse(x).as_dict()
        except ValueError:
            return
except ImportError:
    import json
    def parse_json(x):
        return json.loads(x)


class JsonlDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="default", version=VERSION, description=""),
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        features = datasets.Features({
            "text": datasets.Value("string"),
            "meta": {
                "source": datasets.Value("string"),
                "url": datasets.Value("string")
            }
        })
        return datasets.DatasetInfo(
            features=features
        )

    def _split_generators(self, dl_manager):
        data_files = self.config.data_files
        gens = []
        if "train" in data_files:
            gens.append(datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"files": data_files["train"]}))
        if "test" in data_files:
            gens.append(datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"files": data_files["test"]}))
        if "val" in data_files:
            gens.append(datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"files": data_files["val"]}))
        return gens

    def _generate_examples(self, files):
        global_id = 0
        for f in files:
            with open(f, encoding="utf-8") as f:
                for row in f:
                    data = parse_json(row)
                    yield global_id, {"text": data["text"], "meta": data["meta"]}
                    global_id += 1
