import os
import unittest

from src.data.load_data import load_csv


class TestLoadData(unittest.TestCase):

    def setUp(self):
        self.filename = os.path.join(os.path.dirname(__file__), '../data/yelp_review_polarity_csv/test.csv')

    def test_load_csv(self):
        docs, labels = load_csv(self.filename)
        self.assertEqual(len(docs), len(labels))
        self.assertIsInstance(docs[0], str)
        self.assertIsInstance(labels[0], int)