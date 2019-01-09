import argparse

from rulm.language_model import LanguageModel


def measure_ppl(model_path, val_path):
    model = LanguageModel.load(model_path)
    model.measure_perplexity(val_path)
    # with open(val_path, "r", encoding="utf-8") as r:
    #    lines = r.readlines()[:200]
    #    model.estimate_parameters(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--val-path', required=True)
    args = parser.parse_args()
    measure_ppl(args.model_path, args.val_path)
