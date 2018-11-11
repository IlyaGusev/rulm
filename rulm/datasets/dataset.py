from torch.utils.data import Dataset
from allennlp.common.registrable import Registrable

class Dataset(Dataset, Registrable):
    default_implementation = "stream"

