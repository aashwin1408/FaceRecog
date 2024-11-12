import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, image_files, mask_files, batch_size=16, dim=(512, 512), n_channels=4, n_classes=1, shuffle=True):
        self.image_files = image_files
        self.mask_files = mask_files
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        batch_image_files = self.image_files[index * self.batch_size:(index + 1) * self.batch_size]
        batch_mask_files = self.mask_files[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_image_files, batch_mask_files)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            combined = list(zip(self.image_files, self.mask_files))
            np.random.shuffle(combined)
            self.image_files, self.mask_files = zip(*combined)

    def __data_generation(self, batch_image_files, batch_mask_files):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_classes))

        for i, (image_file, mask_file) in enumerate(zip(batch_image_files, batch_mask_files)):
            X[i,] = joblib.load(image_file)
            y[i,] = joblib.load(mask_file)[:,:,0].reshape(512,512,1)

        return X, y
