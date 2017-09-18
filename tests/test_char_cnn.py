import os
import unittest

from src.data.load_data import load_csv, batch_iter
from src.data.preprocess_data import Preprocessor
from src.models.char_cnn.model import build_model
from src.models.char_cnn.config import ModelConfig, TrainingConfig

from keras.optimizers import Adam


class TestCharCNN(unittest.TestCase):

    def setUp(self):
        self.filename = os.path.join(os.path.dirname(__file__), 'data/test.csv')

    def test_build(self):
        X, y = load_csv(self.filename)
        p = Preprocessor()
        p.fit(X, y)
        model_config = ModelConfig(vocab_size=len(p.vocab), nb_class=len(p.classes))
        model = build_model(model_config.kernel_sizes,
                            model_config.dense_units,
                            model_config.vocab_size,
                            model_config.nb_filter,
                            model_config.nb_class)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam,
                      metrics=['accuracy'])

    def test_train(self):
        X, y = load_csv(self.filename)
        p = Preprocessor()
        p.fit(X, y)
        model_config = ModelConfig(vocab_size=len(p.vocab), nb_class=len(p.classes))
        training_config = TrainingConfig()
        train_batches, train_steps = batch_iter(X, y, training_config.batch_size, p)

        model = build_model(model_config.kernel_sizes,
                            model_config.dense_units,
                            model_config.vocab_size,
                            model_config.nb_filter,
                            model_config.nb_class)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
        model.fit_generator(generator=train_batches,
                            steps_per_epoch=train_steps,
                            epochs=training_config.max_epoch)
