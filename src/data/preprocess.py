from src.data.utils import read_stopwords
from pyvi import ViTokenizer

class Preprocessor(object):
    def __init__(self, stopwords_path = None):
        self.stopwords_path = stopwords_path
        self.special_character = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        self.__set_stopwords()

    def __set_stopwords(self):
        self.stopwords = read_stopwords(self.stopwords_path) if self.stopwords_path else set()

    def segmentation(self, text):
        return ViTokenizer.tokenize(text)

    def split_words(self, input_text):
        text = self.segmentation(input_text)
        try:
            return [x.strip(self.special_character).lower() for x in text.split()]
        except TypeError:
            return []

    def get_words_feature(self, input_text ):
        split_words = self.split_words(input_text)
        return [word for word in split_words if word not in self.stopwords]
