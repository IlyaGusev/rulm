import torch
from allennlp.common.registrable import Registrable

class Embedder(torch.nn.Module, Registrable):
    def __init__(self,
                 input_dim: int,
                 embedding_dim: int):
        super().__init__()

        self._input_dim = input_dim
        self._embedding_dim = embedding_dim

    def get_weight(self) -> torch.nn.Parameter:
        raise NotImplementedError

