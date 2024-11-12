import cv2
import glob
import numpy as np






def get_haarf(img):


    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



    image = cv2.imread(img)

    mage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mage = cv2.resize(mage,(512,512))
    faces = face_cascade.detectMultiScale(mage, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces)!= 0:
        x,y,w,h = faces[0]

        haar_zxz = np.zeros([512,512])
        haar_zxz[y:y+h,x:x+w]=255
        haar_zxz=np.reshape(haar_zxz,(512,512,1)).astype('float32')

    else:

        haar_zxz = np.zeros([512,512])
        haar_zxz=np.reshape(haar_zxz,(512,512,1)).astype('float32')


    return haar_zxz







