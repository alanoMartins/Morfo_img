import numpy as np
import math
import cv2

def erosion(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

def init():

    #cv2.namedWindow('Original')
    cv2.namedWindow('Bin')
    cv2.namedWindow('Result')

    img = cv2.imread('/home/alano/Pictures/Lobos.jpg', 0)
    #cv2.imshow("Original", img)

    while (1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            exit()
        if k == ord('e'):
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            rect, th1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
            cv2.imshow("Bin", th1)
            res = erosion(th1)
            cv2.imshow("Result", res)
        if k == ord('d'):
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            rect, th1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
            cv2.imshow("Bin", th1)
            res = dilate(th1, 1)
            cv2.imshow("Result", res)

def erosion(img):
    for col in range(len(img)):
        for row in range(len(img[col])):
            if img[col, row] == 1:
                if col > 0 and img[col-1][row] == 0: img[col - 1][row] = 2
                if row > 0 and img[col][row - 1] == 0: img[col][row - 1] = 2
                if col + 1 < len(img) and img[col + 1][row] == 0: img[col + 1][row] = 2
                if row + 1 < len(img[col]) and img[col][row + 1] == 0: img[col][row + 1] = 2

    for col in range(len(img)):
        for row in range(len(img[col])):
            if(img[col][row] == 2): img[col][row] = 1

    return img

def dilate(img, k):

    mar = manhattan(img)
    for col in range(len(mar)):
        for row in range(len(mar[col])):
            mar[col][row] = 1 if mar[col][row] <= k else 0

    return mar


def manhattan(img):
    for col in reversed(range(len(img))):
        for row in reversed(range(len(img[col]))):
            if (img[col][row] == 1): img[col][row] = 0
            else:
                img[col][row] = len(img) + len(img[col])
                if col > 0: img[col][row] = min(img[col][row], img[col - 1][row] + 1)
                if row > 0: img[col][row] = min(img[col][row], img[col][row - 1] + 1)

    for col in range(len(img)):
        for row in range(len(img[col])):
            if col + 1 < len(img): img[col][row] = min(img[col][row], img[col + 1][row] + 1)
            if row + 1 < len(img[col]): img[col][row] = min(img[col][row], img[col][row + 1] + 1)

    return img

init()