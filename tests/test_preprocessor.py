import os
import unittest

from src.data.load_data import load_csv, batch_iter
from src.data.preprocess_data import Preprocessor


class TestCharCNN(unittest.TestCase):

    def setUp(self):
        self.filename = os.path.join(os.path.dirname(__file__), 'data/test.csv')

    def test_train(self):
        X, y = load_csv(self.filename)
        p = Preprocessor()
        p.fit(X, y)
        X_, y_ = p.transform(X, y)
        print(X_[0])

    def test_batch_iter(self):
        X, y = load_csv(self.filename)
        p = Preprocessor()
        p.fit(X, y)
        batch_size = 1
        generator, steps = batch_iter(X, y, batch_size, p)
