import tensorflow.keras.backend as K
from  Unet import unet_model
from tensorflow.keras.callbacks import ModelCheckpoint
from DataGenerator import DataGenerator
from tensorflow.keras.optimizers import Adam
import os
import tensorflow as tf
# loss and accuracy functions

def dice_loss(y_true, y_pred):

    y_true = tf.cast(y_true, dtype=tf.float32)

    numerator = 2 * K.sum(y_true * y_pred)
    denominator = K.sum(y_true + y_pred)
    return 1 - (numerator + 1) / (denominator + 1)

def dice_accuracy(y_true, y_pred):

    y_true = tf.cast(y_true, dtype=tf.float32)

    numerator = 2 * K.sum(y_true * K.round(y_pred))
    denominator = K.sum(y_true + K.round(y_pred))
    return (numerator + 1) / (denominator + 1)


# get model

model = unet_model()
model.load_weights("unet_checkpoint.keras")
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss=dice_loss,
              metrics=[dice_accuracy])

#define inputs and masks

train_image_files = sorted(list(map(lambda z : "inputs/"+z,os.listdir("inputs"))),key = lambda x : int(x.split('.')[0].split('t')[-1]))
val_image_files = sorted(list(map(lambda z : "inputs_test/"+z,os.listdir("inputs_test"))),key = lambda x : int(x.split('.')[0].split('t')[-1]))
train_mask_files = sorted(list(map(lambda z : "outputs/"+z,os.listdir("outputs"))),key = lambda x : int(x.split('.')[0].split('t')[-1]))
val_mask_files = sorted(list(map(lambda z : "outputs_test/"+z,os.listdir("outputs_test"))),key = lambda x : int(x.split('.')[0].split('t')[-1]))


train_generator = DataGenerator(train_image_files, train_mask_files)
val_generator = DataGenerator(val_image_files, val_mask_files)

# Model checkpoint callback
checkpoint = ModelCheckpoint('unet_checkpoint.keras', monitor='val_loss', save_best_only=True, mode='min')

# Train the model
history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=97,
                    callbacks=[checkpoint])
