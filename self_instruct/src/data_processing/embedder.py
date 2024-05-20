import torch
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

from src.util.dl import set_random_seed, gen_batch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
        max_length: int = 128,
        device: str = DEVICE,
        pooling_method: str = "default",
        normalize: bool = True,
        text_prefix: str = ""
    ):
        set_random_seed(1337)
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.pooling_method = pooling_method
        self.normalize = normalize
        self.text_prefix = text_prefix

    def __call__(self, texts):
        embeddings = torch.zeros((len(texts), self.model.config.hidden_size))
        total = len(texts) // self.batch_size + 1
        desc = "Calc embeddings"
        if self.text_prefix:
            texts = [self.text_prefix + text for text in texts]
        for batch_num, batch in enumerate(gen_batch(texts, self.batch_size)):
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.model.device)
            with torch.no_grad():
                out = self.model(**inputs)
                if self.pooling_method == "default":
                    batch_embeddings = out.pooler_output
                elif self.pooling_method == "mean":
                    hidden_states = out.last_hidden_state
                    attention_mask = inputs["attention_mask"]
                    last_hidden = hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
                    batch_embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                if self.normalize:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings)
            start_index = batch_num * self.batch_size
            end_index = (batch_num + 1) * self.batch_size
            embeddings[start_index:end_index, :] = batch_embeddings
        return embeddings
