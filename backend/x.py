import cv2 
from haarx import get_haarf
from hogz import get_hogf
image = 'a_10.jpg'

x = get_hogf(image)
y= get_haarf(image)

image = x.reshape(512,512,1)*y.reshape(512,512,1)


image = image.astype('uint8')

cv2.imwrite("xyzh.jpg",image)
