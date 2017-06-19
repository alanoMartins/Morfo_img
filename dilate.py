
class Dilation:

    def __init__(self, is_debug):
        self.is_debug = is_debug

    def manhattan(self, img):
        for col in reversed(range(len(img))):
            for row in reversed(range(len(img[col]))):
                if (img[col][row] == 1):
                    img[col][row] = 0
                else:
                    img[col][row] = len(img) + len(img[col])
                    if col > 0: img[col][row] = min(img[col][row], img[col - 1][row] + 1)
                    if row > 0: img[col][row] = min(img[col][row], img[col][row - 1] + 1)

        for col in range(len(img)):
            for row in range(len(img[col])):
                if col + 1 < len(img): img[col][row] = min(img[col][row], img[col + 1][row] + 1)
                if row + 1 < len(img[col]): img[col][row] = min(img[col][row], img[col][row + 1] + 1)

        return img

    def execute(self, img, k):
        print('Initializing dilate function')
        mar = self.manhattan(img)
        for col in range(len(mar)):
            for row in range(len(mar[col])):
                mar[col][row] = 1 if mar[col][row] <= k else 0

        return mar