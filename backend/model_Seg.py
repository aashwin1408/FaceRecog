from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

def unet_model(input_size=(512, 512, 4), l2_reg=0.001):
    he_normal = HeNormal()

    inputs = Input(input_size)
    
    # Encoder Block 1
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', 
                   kernel_initializer=he_normal, kernel_regularizer=l2(l2_reg))(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', 
                   kernel_initializer=he_normal, kernel_regularizer=l2(l2_reg))(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Encoder Block 2
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', 
                   kernel_initializer=he_normal, kernel_regularizer=l2(l2_reg))(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', 
                   kernel_initializer=he_normal, kernel_regularizer=l2(l2_reg))(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Encoder Block 3
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', 
                   kernel_initializer=he_normal, kernel_regularizer=l2(l2_reg))(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', 
                   kernel_initializer=he_normal, kernel_regularizer=l2(l2_reg))(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', 
                   kernel_initializer=he_normal, kernel_regularizer=l2(l2_reg))(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(0.5)(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', 
                   kernel_initializer=he_normal, kernel_regularizer=l2(l2_reg))(conv4)
    conv4 = BatchNormalization()(conv4)

    # Decoder Block 1
    up1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv4)
    up1 = concatenate([up1, conv3])
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same', 
                   kernel_initializer=he_normal, kernel_regularizer=l2(l2_reg))(up1)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same', 
                   kernel_initializer=he_normal, kernel_regularizer=l2(l2_reg))(conv5)
    conv5 = BatchNormalization()(conv5)

    # Decoder Block 2
    up2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    up2 = concatenate([up2, conv2])
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same', 
                   kernel_initializer=he_normal, kernel_regularizer=l2(l2_reg))(up2)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same', 
                   kernel_initializer=he_normal, kernel_regularizer=l2(l2_reg))(conv6)
    conv6 = BatchNormalization()(conv6)

    # Decoder Block 3
    up3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    up3 = concatenate([up3, conv1])
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same', 
                   kernel_initializer=he_normal, kernel_regularizer=l2(l2_reg))(up3)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same', 
                   kernel_initializer=he_normal, kernel_regularizer=l2(l2_reg))(conv7)
    conv7 = BatchNormalization()(conv7)

    # Final Output Layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv7)
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model
