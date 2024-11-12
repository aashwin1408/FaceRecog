import cv2
import numpy as np
from skimage.transform import resize
from PIL import Image


def get_hogf(img):
    # Read and resize the image
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))

    # Initialize HOG Descriptor with custom parameters
    hog = cv2.HOGDescriptor(
        _winSize=(512, 512),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )

    # Compute HOG features for the 512x512 image
    hog_features = hog.compute(image)

    # Reshape the HOG features for the grid
    num_blocks_x = 63  # number of blocks in x-direction
    num_blocks_y = 63  # number of blocks in y-direction
    features_per_block = 36  # number of HOG features per block

    hog_features_reshaped = hog_features.reshape((num_blocks_y, num_blocks_x, features_per_block))

    # Resize and normalize the HOG features
    hog_as_channel = resize(np.mean(hog_features_reshaped,axis=2), (512, 512))  # Averaging over all channels
    hog_as_channel = hog_as_channel.T  # Transpose to correct orientation


    # Normalize for display and convert to uint8
 
    return hog_as_channel.astype('float32').reshape(512,512,1)




