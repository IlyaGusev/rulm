import argparse

from rulm.language_model import LanguageModel
from rulm.models.neural_net import NeuralNetLanguageModel
from rulm.models.n_gram import NGramLanguageModel


def measure_ppl(model_path, val_path):
    model = LanguageModel.load(model_path)
    model.measure_perplexity_file(val_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--val-path', required=True)
    args = parser.parse_args()
    measure_ppl(args.model_path, args.val_path)
