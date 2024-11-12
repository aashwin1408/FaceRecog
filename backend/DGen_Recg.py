import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import cv2

class SiameseDataGenerator(Sequence):
    def __init__(self, pickle_files, batch_size=32, dim=(512, 512), n_channels=1, shuffle=True):
        """
        Data generator for loading pairs of images and labels from pickle files.
        
        Args:
        - pickle_files: List of paths to pickle files.
        - batch_size: Number of samples per batch.
        - dim: Dimensions of the input images.
        - n_channels: Number of channels in the input images.
        - shuffle: Whether to shuffle data after each epoch.
        """
        self.pickle_files = pickle_files
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.pickle_files) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Select a batch of pickle files
        batch_files = self.pickle_files[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X1, X2, y = self.__data_generation(batch_files)
        
        return (X1, X2), y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.pickle_files)

    def __data_generation(self, batch_files):
        """Generates data containing batch_size samples."""
        # Initialization
        X1 = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        X2 = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size,), dtype=np.float32)

        # Load data from each pickle file
        for i, file_path in enumerate(batch_files):
            # Load the pickle file
            data = joblib.load(file_path)
            
            # Extract the two images and the label
            image1, image2, label = data
            
            # Store in arrays

            X1[i,] = image1.reshape(512,512,1)
            X2[i,] = image2.reshape(512,512,1)
            y[i] = label

        return X1, X2, y
