import numpy as np
import cv2
from morfology import Morfology
import time

DEBUG = False

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml')

def findFace(gray):
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # bw = prepare(gray)
    # ret, bw = cv2.threshold(gray, 127, 255, 0)
    # result = m.flood(bw)

    for (x, y, w, h) in faces:
        gray = gray[y:y + w, x:x + h]
        gray = cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if len(gray) > 10:
            return gray


def init():
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('sample1.avi')
    m = Morfology(True)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        #frame = cv2.imread('ch_ex.png')

        if frame is None:
            break

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = frame

        basicImage = gray.copy()
        basicOperations(basicImage)

        # chImage = gray.copy()
        # convexHull = m.ch2(chImage)
        # cv2.imshow('ConvexHull', convexHull)

        tkImage = gray.copy()
        thinkness = m.thinkness(tkImage)
        cv2.imshow('Thinkness', thinkness)

        tnImage = gray.copy()
        thinning = m.thinning10(tnImage)
        cv2.imshow('Thinning', thinning)

        skImage = gray.copy()
        skeleton = m.skeleton(skImage)
        cv2.imshow('Skeleton', skeleton)

        brImage = gray.copy()
        border = m.extract_boundry(brImage)
        cv2.imshow('Border', border)

        flImage = gray.copy()
        flood = m.flood(flImage)
        cv2.imshow('Flood', flood)

        thimage = gray.copy()
        top_hat = m.top_hat(thimage)
        cv2.imshow('Top_hat', top_hat)

        bhimage = gray.copy()
        black_hat = m.black_hat(bhimage)
        cv2.imshow('Blakc_hat', black_hat)

        gdimage = gray.copy()
        gradient = m.gradient(gdimage)
        cv2.imshow('Gradient', gradient)


        time.sleep(0.5)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def basicOperations(img):
    m = Morfology(True)

    opened = m.opening(img)
    closed = m.closing(img)
    eroded = m.erode(img)
    dilated = m.dilate(img)
    hitnmiss = m.hitnmiss(img)

    cv2.imshow('Dilatation', dilated)
    cv2.imshow('Erosion', eroded)
    cv2.imshow('Closing', closed)
    cv2.imshow('Openning', opened)
    cv2.imshow('HitnMiss', hitnmiss)

init()