import cv2
import numpy as np

class Morfology:
    def __init__(self, is_debug):
        self.is_debug = is_debug

    def draw(self, name, image):
        if (self.is_debug):
            cv2.imshow(name, image)

    def binarize(self, image):
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        rect, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th1

    def convex_hull(self, img):
        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        hull = cv2.convexHull(cnt)

    def hitnmiss(self, img):
        element = np.array([[0, 1, 0], [1, -1, 1], [0, 1, 0]])
        return cv2.morphologyEx(img, cv2.MORPH_HITMISS, element)

    def flood(self, img):
        mask = np.zeros((len(img)+2,len(img[0])+2),np.uint8)
        return  cv2.floodFill(img, mask, (100, 100), 0)

    def extract_boundry(self, img):
        return cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def components_connected(self, img):
        connectivity = 4
        output = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
        print("0 %d", output[0])
        print("1 %d", output[1])
        print("2 %d", output[2])
        print("3 %d", output[3])
        return img


    def skeleton(self, img):
        size = np.size(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        skel = np.zeros(img.shape, np.uint8)
        done = False

        while (not done):
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True

        return skel