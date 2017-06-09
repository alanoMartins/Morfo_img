import cv2
import numpy as np

def erosion(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

def init():

    cv2.namedWindow('image')

    img = cv2.imread('/home/alano/Pictures/m1', 0)
    cv2.imshow("Original", img)

    while (1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            exit()
        if k == ord('e'):
            res = erosion(img)
            cv2.imshow("Result", res)


init()