import numpy as np
from keras.utils import Sequence, to_categorical
import glob


class DataGenerator(Sequence):
    ''' Class inheriting from keras.utils.sequence used to
        generate the dataset for training and validation'''

    def __init__(self, list_IDs, db_part='train', dim=(10, 128), batch_size=16, n_classes=2, shuffle=True):
        self.list_IDs   = list_IDs
        self.db_part    = db_part
        self.dim        = dim
        self.batch_size = batch_size
        self.n_classes  = n_classes
        self.shuffle    = shuffle

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Update indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'

        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        Y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,]   = np.load('../data/arrays/{}/{}'.format(self.db_part, ID))
            Y[i]    = self.ID_to_label(ID)

        return X, to_categorical(Y, num_classes=self.n_classes)

    def ID_to_label(self, ID):
        'Define de label {0: bg, 1: sn} to the corresponded ID'

        if ID.split('_')[0] == 'sn':
            return 1
        else:
            return 0
        
    @property
    def get_batch(self):
        return self.batch_size