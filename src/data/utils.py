import os
from gensim import corpora
import pickle
        
def read_stopwords(file_path, encoder=None):
    with open(file_path, 'r', encoding=encoder if encoder else 'utf-8') as f:
        stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
    return stopwords

def load_dictionary(file_path):
    return corpora.Dictionary.load_from_text(file_path)

def store_dictionary(file_path, dict_words):
    dictionary = corpora.Dictionary(dict_words)
    dictionary.filter_extremes(no_below=5, no_above=0.2)
    dictionary.save_as_text(file_path)

def convert_label(abbreviation):
    label_dict = {
        '__CTXH__': 'Chính trị Xã hội',
        '__DS__': 'Đời sống',
        '__KH__': 'Khoa học',
        '__KD__': 'Kinh doanh',
        '__PL__': 'Pháp luật',
        '__SK__': 'Sức khỏe',
        '__TG__': 'Thế giới',
        '__TT__': 'Thể thao',
        '__VH__': 'Văn hóa',
        '__VT__': 'Vi tính'
    }
    return label_dict.get(abbreviation, abbreviation)

if __name__ == "__main__":  
    with open('src/models/ckpt/naive_model.pk', 'rb') as f:
        model = pickle.load(f)
    print(model)