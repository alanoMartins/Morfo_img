import numpy as np
import cv2
from morfology import Morfology

DEBUG = False

def erosion(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

def init():
    cap = cv2.VideoCapture(0)
    m = Morfology(True)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #bw = prepare(gray)
        ret, bw = cv2.threshold(gray, 127, 255, 0)
        #result = m.flood(bw)
        result = m.extract_boundry(bw)





        # Display the resulting frame
        cv2.imshow('frame', result)
        cv2.imshow('cnt', gray)
        cv2.imshow('bw', bw)
        #cv2.imshow('hull', hull)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def prepare(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    rect, th1 = cv2.threshold(blur, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th1

def contour(gray, bw):
    m = Morfology(True)
    result, contours, hierarchy = m.extract_boundry(bw)

    cv2.drawContours(gray, contours, -1, (0, 255, 0), 3)
    cv2.imshow('cnt', gray)
    cv2.imshow('bw', bw)





init()