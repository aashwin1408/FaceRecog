from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def create_base_cnn(input_shape=(512, 512, 1), l2_reg=0.001):
    he_normal = HeNormal()
    
    inputs = Input(input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', 
               kernel_initializer=he_normal, kernel_regularizer=l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', 
               kernel_initializer=he_normal, kernel_regularizer=l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same', 
               kernel_initializer=he_normal, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', 
               kernel_initializer=he_normal, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', 
               kernel_initializer=he_normal, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', 
               kernel_initializer=he_normal, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_initializer=he_normal, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    
    return Model(inputs, x)

def create_siamese_network(input_shape=(512, 512, 1), l2_reg=0.001):
    base_cnn = create_base_cnn(input_shape, l2_reg)
    
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # Pass both inputs through the base CNN
    processed_a = base_cnn(input_a)
    processed_b = base_cnn(input_b)

    # Compute the L1 distance between the two outputs
    distance = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]),output_shape=lambda shapes: shapes[0])([processed_a, processed_b])
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(distance)
    
    # Create the Siamese Model
    model = Model(inputs=[input_a, input_b], outputs=outputs)
    
    return model
