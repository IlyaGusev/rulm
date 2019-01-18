import os
import argparse

from allennlp.data.vocabulary import Vocabulary
from allennlp.common.params import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from rulm.language_model import LanguageModel


def preprocess(train_path, vocabulary_path, config_path):
    vocabulary_path = vocabulary_path
    config_path = config_path
    params = Params.from_file(config_path)

    vocabulary_params = params.pop("vocabulary", default=Params({}))
    reader_params = params.duplicate().pop("reader", default=Params({}))
    reader = DatasetReader.from_params(reader_params)
    dataset = reader.read(train_path)

    if os.path.isdir(vocabulary_path):
        vocabulary = Vocabulary.from_files(vocabulary_path)
    else:
        vocabulary = Vocabulary.from_params(vocabulary_params, instances=dataset)
        vocabulary.save_to_files(vocabulary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-path', required=True)
    parser.add_argument('--vocabulary-path', required=True)
    parser.add_argument('--config-path', required=True)
    args = parser.parse_args()
    preprocess(args.train_path, args.vocabulary_path, args.config_path)
