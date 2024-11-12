import tensorflow as tf 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import os
from tensorflow.keras.metrics import Precision, Recall,Accuracy
from  backend.model_Seg import unet_model
from backend.model_Recg import create_siamese_network
import numpy as np
import backend.haarx as haarx
import backend.hogz as hogz

import os
import joblib
import cv2



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


def contrastive_loss(y_true, y_pred, margin=1.0):
    squared_pred = tf.square(y_pred)
    margin_squared = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean((1 - y_true)* 1.9 * squared_pred + y_true * margin_squared)


def contrastive_accuracy(y_true, y_pred, margin=1.0):
    # For similar pairs (y_true = 0), y_pred should be < margin
    # For dissimilar pairs (y_true = 1), y_pred should be >= margin

    predictions = tf.where(y_pred < margin, 0.0, 1.7)
    correct = tf.equal(y_true, predictions)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy


localizer = unet_model()
localizer.load_weights("./backend/model_localizer.keras")
localizer.compile(optimizer=Adam(learning_rate=0.0001),
              loss=dice_loss,
              metrics=[dice_accuracy])


encoder = create_siamese_network()
encoder.load_weights("./backend/model_encoder.keras")
encoder.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=lambda y_true, y_pred: contrastive_loss(y_true, y_pred, margin=1.9),
    metrics=[lambda y_true, y_pred: contrastive_accuracy(y_true, y_pred, margin=0.6),Accuracy(),Precision(),Recall()]
)

def seg_input(image_path):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(512,512))
    image = image.astype("float32")/255.0

    haarf = haarx.get_haarf(image_path)
    hogf = hogz.get_hogf(image_path)
    image = image.reshape(512,512,3).astype('float32')

    h_image = image*haarf
    g_image = hogf*haarf
    f_image = np.dstack((h_image,g_image)).reshape(1,512,512,4)

    return f_image

def seg_output(image_path):

    imagex = cv2.imread(image_path)
    imagex = cv2.cvtColor(imagex,cv2.COLOR_BGR2RGB)
    imagex = cv2.resize(imagex,(512,512))
    image=seg_input(image_path)
    mask = localizer.predict(image)
    
    



    return (mask>0.05).astype('float32')


def recg_input(image_path1,image_path2):
    seg1 = seg_output(image_path1)
    seg2 = seg_output(image_path2)
    
    image1 = cv2.imread(image_path1)
    image1 = cv2.resize(image1,(512,512))
    image1 = image1.astype("float32")
    image1 = image1.reshape(1,512,512,3)
    
    image2 = cv2.imread(image_path2)
    image2 = cv2.resize(image2,(512,512))
    image2 = image2.astype("float32")
    image2 = image2.reshape(1,512,512,3)
    
    
    recog_input1=image1*seg1.reshape(1,512,512,1)
    recog_input1=cv2.cvtColor(recog_input1[0],cv2.COLOR_RGB2GRAY)
    recog_input1= recog_input1.reshape(1,512,512,1)
    recog_input1 = recog_input1.astype('float32')/ 255.0
    
    recog_input2=image2*seg1.reshape(1,512,512,1)
    recog_input2=cv2.cvtColor(recog_input2[0],cv2.COLOR_RGB2GRAY)
    recog_input2=recog_input2.reshape(1,512,512,1)
    recog_input2 = recog_input2.astype('float32')/ 255.0
    
    return [recog_input1,recog_input2]

def recg_output(image_path1,image_path2):
    
    f_out = recg_input(image_path1,image_path2)
    e=encoder.predict(f_out)
    
    return e
        





if __name__ == '__main__':
	while True :
		x =  input("First Image : ")
		y =  input("Second Image : ")
		print(recg_output(x,y))
		if x == "exit" or y == "exit":
			break






