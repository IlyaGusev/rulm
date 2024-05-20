from fasttext import load_model as ft_load_model


class FasttextLanguageDetector:
    def __init__(self, model_path="models/lid.176.bin", max_tokens=50):
        self.model = ft_load_model(model_path)
        self.label_offset = len("__label__")
        self.max_tokens = max_tokens

    def __call__(self, text):
        text = text.replace("\xa0", " ").strip()
        text = " ".join(text.split()[:self.max_tokens])

        (label,), (prob,) = self.model.predict(text, k=1)
        label = label[self.label_offset:]
        return label, prob
