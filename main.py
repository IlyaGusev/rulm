from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN

from rulm.settings import TRAIN_EXAMPLE, TRAIN_VOCAB_EXAMPLE, TEST_EXAMPLE, RNNLM_REMEMBER_EXAMPLE
from rulm.stream_reader import LanguageModelingStreamReader

reader = LanguageModelingStreamReader()
#dataset = reader.read(TRAIN_EXAMPLE)
#vocabulary = Vocabulary.from_instances(dataset)
v = Vocabulary.from_files(TRAIN_VOCAB_EXAMPLE)
print(v.get_token_index(DEFAULT_PADDING_TOKEN))
#vocabulary.save_to_files(TRAIN_VOCAB_EXAMPLE)
#vocabulary.load(TRAIN_VOCAB_EXAMPLE)
#vocabulary.sort(1000)
#vocabulary.add_file(RNNLM_REMEMBER_EXAMPLE)
#model = RNNLanguageModel(vocabulary)
#model.train_file(RNNLM_REMEMBER_EXAMPLE, epochs=20)
#model.train_file(TRAIN_EXAMPLE, epochs=10)
