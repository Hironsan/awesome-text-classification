from keras.layers import Input, Conv1D, Dense, MaxPool1D, Flatten
from keras.models import Model


def build_model(kernel_sizes, dense_units, vocab_size, nb_filter, nb_class):
    maxlen = 1014
    inputs = Input(batch_shape=(None, maxlen, vocab_size))

    conv1 = Conv1D(nb_filter, kernel_sizes[0], activation='relu')(inputs)
    pool1 = MaxPool1D(pool_size=3)(conv1)

    conv2 = Conv1D(nb_filter, kernel_sizes[1], activation='relu')(pool1)
    pool2 = MaxPool1D(pool_size=3)(conv2)

    conv3 = Conv1D(nb_filter, kernel_sizes[2], activation='relu')(pool2)
    conv4 = Conv1D(nb_filter, kernel_sizes[3], activation='relu')(conv3)
    conv5 = Conv1D(nb_filter, kernel_sizes[4], activation='relu')(conv4)
    conv6 = Conv1D(nb_filter, kernel_sizes[5], activation='relu')(conv5)
    pool3 = MaxPool1D(pool_size=3)(conv6)
    pool3 = Flatten()(pool3)

    fc1 = Dense(dense_units[0], activation='relu')(pool3)
    fc2 = Dense(dense_units[1], activation='relu')(fc1)
    pred = Dense(nb_class, activation='softmax')(fc2)

    model = Model(inputs=[inputs], outputs=[pred])

    return model
