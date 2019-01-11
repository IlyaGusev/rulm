import os
import argparse

from allennlp.data.vocabulary import Vocabulary
from allennlp.common.params import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from rulm.language_model import LanguageModel


def train(model_path, train_path, val_path):
    vocabulary_path = os.path.join(model_path, "vocabulary")
    config_path = os.path.join(model_path, "config.json")
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

    train_params = params.pop("train", Params({}))
    model = LanguageModel.from_params(params, vocab=vocabulary)
    model.train(train_path, train_params, serialization_dir=model_path, valid_file_name=val_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--train-path', required=True)
    parser.add_argument('--val-path')
    args = parser.parse_args()
    train(args.model_path, args.train_path, args.val_path)
