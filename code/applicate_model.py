from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
model = load_model('../data/saved_model1.h5')
# fn = lambda x : 255 if x > thresh else 0
# r = img.convert('L').point(fn, mode='1')

#With Image.open:
# img = Image.open('../data/test_image.png').convert('L')
# pic = np.array(img)

#With cv2:
img = cv2.imread('../data/test_image.png',0)
cv2.imshow("picture:",img)

#Process image:
pic = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
# cv2.imshow("picture after processing", pic)
cv2.imshow("picture after thresh_gaussian",pic)
ret,pic = cv2.threshold(pic,200,255,cv2.THRESH_BINARY)
cv2.imshow("picture after all processing",pic)
# print(img)

pic = 255-pic
pic = pic/255.0
# print(pic)
pic = pic.reshape(-1,28,28,1)
tmp = model.predict(pic)
pic_pre = np.argmax(tmp,axis=1)
print(f"Number of prediction: {pic_pre}")
cv2.waitKey(0)
cv2.destroyWindow()

