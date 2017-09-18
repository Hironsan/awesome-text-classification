import numpy as np
#from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.externals import joblib
from keras.utils import to_categorical

UNK = '<UNK>'
PAD = '<PAD>'


class Preprocessor(object):

    def __init__(self, padding=True):
        self.padding = padding
        self.vocab = None
        self.classes = None
        self.maxlen=1014

    def fit(self, X, y):
        chars = {PAD: 0, UNK: 1}
        tags  = {}

        for doc in X:
            for char in doc:
                if char in chars:
                    continue
                chars[char] = len(chars)

        for t in y:
            if t in tags:
                continue
            tags[t] = len(tags)

        self.vocab = chars
        self.classes = tags

        return self

    def transform(self, X, y=None):
        chars = []
        for doc in X:
            char_ids = []
            for char in doc[:self.maxlen]:
                if char in self.vocab:
                    char_ids.append(self.vocab[char])
                else:
                    char_ids.append(self.vocab[UNK])
            char_ids += [self.vocab[PAD]] * (self.maxlen - len(char_ids))  # padding
            chars.append(char_ids)
        chars = dense_to_one_hot(chars, len(self.vocab))

        if y is not None:
            y = [self.classes[t] for t in y]
            y = to_categorical(y, len(self.classes))

        return (chars, y) if y is not None else chars

    def inverse_transform(self, y):
        indice_tag = {i: t for t, i in self.classes.items()}
        return [indice_tag[y_] for y_ in y]
    """
    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)
        return p
    """

def dense_to_one_hot(labels_dense, num_classes):
    res = []
    for sent in labels_dense:
        L = []
        for char_id in sent:
            ary = np.zeros(num_classes, dtype=np.int32)
            ary[char_id] = 1
            L.append(ary)
        res.append(L)
    return np.asarray(res, dtype=np.int32)
