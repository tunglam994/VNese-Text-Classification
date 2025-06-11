from src.data.preprocess import Preprocessor
from src.data.utils import load_dictionary
from gensim import matutils
import os

class FeatureExtraction(object):
    def __init__(self, data):
        self.data = data

    def build_dictionary(self):
#        print('Building dictionary')
        dict_words = []
        i = 0
        for text in self.data:
            i += 1
#             print("Step {} / {}".format(i, len(self.data)))
            words = Preprocessor(text = text['content']).get_words_feature()
            dict_words.append(words)
        FileStore(file_path=DICTIONARY_PATH).store_dictionary(dict_words)

    def __load_dictionary(self):
        if os.path.exists(DICTIONARY_PATH) == False:
            self.build_dictionary()
        self.dictionary = FileReader(DICTIONARY_PATH).load_dictionary()

    def __build_dataset(self):
        self.features = []
        self.labels = []
        i = 0
        for d in self.data:
            i += 1
#             print("Step {} / {}".format(i, len(self.data)))
            self.features.append(self.get_dense(d['content']))
            self.labels.append(d['category'])

    def get_dense(self, text):
        self.__load_dictionary()
        words = Preprocessor(text).get_words_feature()
        # Bag of words
        vec = self.dictionary.doc2bow(words)
        dense = list(matutils.corpus2dense([vec], num_terms=len(self.dictionary)).T[0])
        return dense

    def get_data_and_label(self):
        self.__build_dataset()
        return self.features, self.labels
    

def get_dense(words, dictionary_path):
    dictionary = load_dictionary(dictionary_path)
    vec = dictionary.doc2bow(words)
    dense = list(matutils.corpus2dense([vec], num_terms=len(dictionary)).T[0])
    return dense

