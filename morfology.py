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
        rect, th1 = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY)
        cv2.imshow('Binary', th1)
        return th1

    def opening(self, img):
        img = self.binarize(img)
        kernel = np.ones((5, 5), np.uint8)
        return  cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    def closing(self, img):
        img = self.binarize(img)
        kernel = np.ones((5, 5), np.uint8)
        return  cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    def erode(self, img):
        img = self.binarize(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        return cv2.erode(img, element, iterations=5)

    def dilate(self, img):
        img = self.binarize(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        return cv2.dilate(img, element, iterations=5)

    def hitnmiss(self, img):
        img = self.binarize(img)
        junction1 = np.array([[0, 1, 0], [-1, 1, 1], [-1, -1, 0]])
        r1 = cv2.morphologyEx(img, cv2.MORPH_HITMISS, junction1)
        cv2.imshow("Junction 1", r1)

        junction2 = np.array([[0, 1, 0], [1, 1, -1], [0, -1, -1]])
        r2 = cv2.morphologyEx(img, cv2.MORPH_HITMISS, junction2)
        cv2.imshow("Junction 2", r2)

        junction3 = np.array([[0, -1, -1], [1, 1, -1], [0, 1, 0]])
        r3 = cv2.morphologyEx(img, cv2.MORPH_HITMISS, junction3)
        cv2.imshow("Junction 3", r3)

        junction4 = np.array([[-1, -1, 0], [-1, 1, 1], [0, 1, 0]])
        r4 = cv2.morphologyEx(img, cv2.MORPH_HITMISS, junction4)
        cv2.imshow("Junction 4", r4)

        j12 = cv2.bitwise_or(r1, r2)
        j34 = cv2.bitwise_or(r3, r4)
        return cv2.bitwise_or(j12, j34)

    def conex_hull_image(self, img):
        img = cv2.pyrDown(img)
        # threshold image
        ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                          127, 255, cv2.THRESH_BINARY)
        # get contours from image
        image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)

        # for each contour
        for cnt in contours:
            # get convex hull
            hull = cv2.convexHull(cnt)
            # draw it in red color
            cv2.drawContours(img, [hull], -1, (0, 0, 255), 1)

        return img

    def flood(self, img):
        #img = self.binarize(img)
        mask = np.zeros((len(img)+2, len(img[0])+2),np.uint8)
        seed = (100, 100)
        cv2.floodFill(img, mask, seed, (255, 255, 255), 100, 100, cv2.FLOODFILL_FIXED_RANGE)
        return cv2.circle(img, seed, 2, (0, 0, 255), -1)

    def extract_boundry(self, img):
        gray = cv2.GaussianBlur(img, (5, 5), 0)

        thresh = self.binarize(gray)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        result, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

        return img

    def components_connected(self, img):
        bw = self.binarize(img)
        connectivity = 4
        output = cv2.connectedComponentsWithStats(bw, connectivity, cv2.CV_32S)
        print("0 %d", output[0])
        print("1 %d", output[1])
        print("2 %d", output[2])
        print("3 %d", output[3])

        return img

    def skeleton(self, img):
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        rect, img = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY_INV)
        size = np.size(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        skel = np.zeros(img.shape, np.uint8)
        done = False

        while not done:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True

        return skel

    def thinkness(self, img):
        bw = self.binarize(img)
        return cv2.bitwise_or(bw, self.hitnmiss(img))

    def thinkness10(self, img):
        bw = self.binarize(img)
        res1 = cv2.bitwise_or(bw, self.hitnmiss(img))
        res2 = cv2.bitwise_or(bw, res1)
        res3 = cv2.bitwise_or(bw, res2)
        res4 = cv2.bitwise_or(bw, res3)
        res5 = cv2.bitwise_or(bw, res4)
        res6 = cv2.bitwise_or(bw, res5)
        res7 = cv2.bitwise_or(bw, res6)
        res8 = cv2.bitwise_or(bw, res7)
        res9 = cv2.bitwise_or(bw, res8)
        res10 = cv2.bitwise_or(bw, res9)
        return cv2.bitwise_or(bw, res10)

    def thinning(self, img):
        bw = self.binarize(img)
        return cv2.bitwise_and(bw, cv2.bitwise_not(self.hitnmiss(img)))

    def thinning10(self, img):
        bw = self.binarize(img)
        res1 =  cv2.bitwise_and(bw, cv2.bitwise_not(self.hitnmiss(img)))
        res2 = cv2.bitwise_and(bw, res1)
        res3 = cv2.bitwise_and(bw, res2)
        res4 = cv2.bitwise_and(bw, res3)
        res5 = cv2.bitwise_and(bw, res4)
        res6 = cv2.bitwise_and(bw, res5)
        res7 = cv2.bitwise_and(bw, res6)
        res8 = cv2.bitwise_and(bw, res7)
        res9 = cv2.bitwise_and(bw, res8)
        res10 = cv2.bitwise_and(bw, res9)
        return cv2.bitwise_and(bw, res10)

    def top_hat(self, img):
        element = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, element)

    def black_hat(self, img):
        element = np.ones((5,5),np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, element)

    def gradient(self, img):
        element = np.ones((5,5),np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, element)
