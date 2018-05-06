"""
Model definition.
"""
import json

import keras.backend as K
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, Lambda, Activation
from keras.layers import Conv1D, MaxPooling1D, Flatten, GlobalMaxPooling1D
from keras.layers.merge import Concatenate
from keras.models import Model


class BaseModel(object):

    def __init__(self):
        self.model = None

    def save(self):
        pass

    def load(self):
        pass

    def build(self):
        pass


class SimpleCNN(BaseModel):

    def __init__(self, max_sequence_length, vocab_size, num_tags, embedding_dim=100,
                 filter_sizes=(3, 4, 5), num_filters=(100, 100, 100),
                 num_units=100, keep_prob=0.5):
        super(KimCNN).__init__()
        assert len(filter_sizes) == len(num_filters)
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.num_units = num_units
        self.num_tags = num_tags
        self.keep_prob = keep_prob

    def build(self):
        sequence_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding = Embedding(
            self.vocab_size + 1,  # due to mask_zero
            self.embedding_dim,
            input_length=self.max_sequence_length,
            # weights=[weight],
            trainable=True
        )(sequence_input)

        convs = []
        for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
            conv = Conv1D(filters=num_filter,
                          kernel_size=filter_size,
                          activation='relu')(embedding)
            pool = GlobalMaxPooling1D()(conv)
            convs.append(pool)

        z = Concatenate()(convs)
        z = Dense(self.num_units)(z)
        z = Dropout(self.keep_prob)(z)
        z = Activation('relu')(z)
        pred = Dense(self.num_tags, activation='sigmoid')(z)
        model = Model(inputs=[sequence_input], outputs=[pred])

        return model


class KimCNN(BaseModel):

    def __init__(self, max_sequence_length, vocab_size, num_tags, embedding_dim=100,
                 filter_sizes=(3, 4, 5), num_filters=(100, 100, 100),
                 num_units=100, keep_prob=0.5):
        super(KimCNN).__init__()
        assert len(filter_sizes) == len(num_filters)
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.num_units = num_units
        self.num_tags = num_tags
        self.keep_prob = keep_prob

    def build(self):
        sequence_input = Input(shape=(self.max_sequence_length,), dtype='int32')

        embeddings_static = Embedding(
            self.vocab_size + 1,  # due to mask_zero
            self.embedding_dim,
            input_length=self.max_sequence_length,
            # weights=[weight],
            trainable=False
        )(sequence_input)
        embeddings_static = Dropout(self.keep_prob)(embeddings_static)

        embeddings_non_static = Embedding(
            self.vocab_size + 1,  # due to mask_zero
            self.embedding_dim,
            input_length=self.max_sequence_length,
            # weights=[weight],
            trainable=True
        )(sequence_input)
        embeddings_non_static = Dropout(self.keep_prob)(embeddings_non_static)

        convs = []
        for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
            conv_layer = Conv1D(filters=num_filter,
                                kernel_size=filter_size,
                                activation='relu')(embeddings_static)
            pool_layer = MaxPooling1D()(conv_layer)
            flatten = Flatten()(pool_layer)
            convs.append(flatten)
        for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
            conv_layer = Conv1D(filters=num_filter,
                                kernel_size=filter_size,
                                activation='relu')(embeddings_non_static)
            pool_layer = MaxPooling1D()(conv_layer)
            flatten = Flatten()(pool_layer)
            convs.append(flatten)

        z = Concatenate()(convs)
        z = Dense(self.num_units)(z)
        z = Dropout(self.keep_prob)(z)
        z = Activation('relu')(z)
        pred = Dense(self.num_tags, activation='softmax')(z)
        model = Model(inputs=[sequence_input], outputs=[pred])

        return model
