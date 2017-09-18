import csv

import numpy as np


def load_csv(file_path):
    docs, labels = [], []
    with open(file_path) as f:
        for label, text in csv.reader(f):
            docs.append(text.lower())
            labels.append(int(label))

    return np.asarray(docs), np.asarray(labels)


def batch_iter(data, labels, batch_size, preprocessor, shuffle=True):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    def data_generator():
        """
        Generates a batch iterator for a dataset.
        """
        data_size = len(data)
        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X, y = shuffled_data[start_index: end_index], shuffled_labels[start_index: end_index]
                yield preprocessor.transform(X, y)

    return data_generator(), num_batches_per_epoch
