import joblib
import cv2


x = joblib.load("list9.pkl")

y = x[0]*255
y=y.astype('uint8')

#y[:,:,:3]=(y[:,:,:3]*255).astype("uint8")

cv2.imshow("x.png",y[:,:,:3])
cv2.waitKey(0)

cv2.destroyAllWindows()

