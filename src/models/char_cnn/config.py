class ModelConfig(object):

    def __init__(self,
                 vocab_size,
                 nb_class,
                 kernel_sizes=(7, 7, 3, 3, 3, 3),
                 nb_filter=256,
                 dense_units=(1024, 1024),
                 maxlen=1014,
                 ):
        self.vocab_size = vocab_size
        self.nb_class = nb_class
        self.kernel_sizes = kernel_sizes
        self.nb_filter = nb_filter
        self.dense_units = dense_units
        self.maxlen = maxlen


class TrainingConfig(object):

    def __init__(self):
        self.batch_size = 80
        self.lr = 0.9
        self.max_epoch = 15
