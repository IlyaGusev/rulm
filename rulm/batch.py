import torch
from torch.autograd import Variable

from rulm.vocabulary import Vocabulary


class Batch:
    def __init__(self, vocabulary: Vocabulary, max_length: int):
        self.vocabulary = vocabulary  # type: Vocabulary
        self.max_length = max_length
        self.word_indices = []
        self.y = []
        self.lengths = []
        self.reset()

    def reset(self):
        self.word_indices = []
        self.y = []
        self.lengths = []

    def add_sentence(self, indices):
        self.word_indices.append(indices[:self.max_length])
        self.y.append(indices[1:self.max_length+1])
        self.lengths.append(min(len(indices), self.max_length))

    def __len__(self):
        return len(self.word_indices)

    def sort_by_length(self):
        self.word_indices = self.__sort_list_by_length(self.word_indices)
        self.y = self.__sort_list_by_length(self.y)
        self.lengths = self.__sort_list_by_length(self.lengths)

    def pad(self):
        max_length = max(self.lengths)
        self.word_indices = [self.vocabulary.pad_indices(sentence, max_length) for sentence in self.word_indices]
        self.y = [self.vocabulary.pad_indices(sentence, max_length) for sentence in self.y]

    def __sort_list_by_length(self, l):
        return [t[1] for t in sorted(enumerate(l), key=lambda t: self.lengths[t[0]], reverse=True)]


class VarBatch:
    def __init__(self, batch: Batch):
        use_cuda = torch.cuda.is_available()
        LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

        batch.pad()
        batch.sort_by_length()
        self.word_indices = Variable(LongTensor(batch.word_indices).transpose(0, 1), requires_grad=False)
        self.y = Variable(LongTensor(batch.y).transpose(0, 1), requires_grad=False)
        self.lengths = batch.lengths
