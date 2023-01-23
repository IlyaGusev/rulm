import json
import os

import datasets


_DOCUMENT = "text"


class JsonlDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="default", version=VERSION, description=""),
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        features = datasets.Features(
            {
                _DOCUMENT: datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            features=features,
            supervised_keys=(_DOCUMENT,),
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
                    data = json.loads(row)
                    yield global_id, {_DOCUMENT: data[_DOCUMENT]}
                    global_id += 1
