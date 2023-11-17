from tensorflow.keras.models import load_model
import argparse
# from PIL import Image
import numpy as np
import cv2

model = load_model('../data/saved_model1.h5')


# fn = lambda x : 255 if x > thresh else 0
# r = img.convert('L').point(fn, mode='1')

def get_args():
    parser = argparse.ArgumentParser("Recognize digit from Image")
    parser.add_argument("--source", type=str, default="../data/test_image.png", help="Path to input image")
    args = parser.parse_args()
    return args

def main(opt):
    # With Image.open:
    # img = Image.open('../data/test_image.png').convert('L')
    # pic = np.array(img)
    path = opt.source

    # With cv2:
    img = cv2.imread(path, 0)
    cv2.imshow("picture:", img)
    # Process image:
    ret, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    pic = img

    # pic = cv2.adaptiveThreshold(pic, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #                             cv2.THRESH_BINARY, 11, 2)
    # cv2.imshow("picture after processing", pic)

    cv2.imshow("picture after thresh_gaussian", pic)
    pic = cv2.resize(pic,(28,28), interpolation= cv2.INTER_AREA)
    cv2.imshow("picture after all processing", pic)
    # print(img)

    pic = 255 - pic
    pic = pic / 255.0
    # print(pic)
    pic = pic.reshape(-1, 28, 28, 1)
    tmp = model.predict(pic)
    pic_pre = np.argmax(tmp, axis=1)
    print(f"Number of prediction: {pic_pre}")
    cv2.waitKey(0)
    cv2.destroyWindow()

if __name__ == '__main__':
    opt = get_args()
    main(opt)