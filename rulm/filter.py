from rulm.vocabulary import Vocabulary

class Filter:
    def __init__(self, vocabulary: Vocabulary):
        self.vocabulary = vocabulary

    def __call__(self, index: int):
        raise NotImplementedError()

    def advance(self, index: int):
        raise NotImplementedError()

class AlphabetOrderFilter(Filter):
    def __init__(self, vocabulary: Vocabulary, language: str="ru"):
        assert language in ("ru", "en"), "Bad language for filter"
        if language == "ru":
            self.current_letter = "а"
        elif language == "en":
            self.current_letter = "a"
        Filter.__init__(self, vocabulary)

    def __call__(self, index: int) -> bool:
        print(index)
        if self.vocabulary.get_word_by_index(index)[0] == self.current_letter:
            return True
        else:
            return False

    def advance(self, index: int) -> None:
        self.current_letter = chr(ord(self.current_letter) + 1)
        if language == "ru" and self.current_letter > 'я':
            self.current_letter = 'а'
        if language == "en" and self.current_letter > 'z':
            self.current_letter = 'a'
