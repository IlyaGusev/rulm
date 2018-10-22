from rulm.vocabulary import Vocabulary
from rulm.settings import TRAIN_EXAMPLE, TRAIN_VOCAB_EXAMPLE, TEST_EXAMPLE
from rulm.rnnlm import RNNLanguageModel
from rulm.chunk_dataset import ChunkDataset, ChunkDataLoader
#from torch.utils.data.dataloader import DataLoader

#TRAIN = "/media/yallen/My Passport/Datasets/rdt_very_clean/rdt/rdt.train.01.txt"
TRAIN = "rdt.train.01.txt"
vocabulary = Vocabulary()
#vocabulary.add_file(TRAIN)
vocabulary.load("vocab.01.txt")
vocabulary.sort(50000)
dataset = ChunkDataset(vocabulary, ".", "./chunks")
loader = ChunkDataLoader(dataset, batch_size=64)
for batch in loader:
    print(batch)
#print(dataset[10])
#print(dataset[600000])
#model = RNNLanguageModel(vocabulary)
#model.train_file(TRAIN, epochs=2)
#print(model.sample_decoding(["После"]))
# model.measure_perplexity_file(TEST_EXAMPLE)
