
class Erosion:

    def execute(self, img):
        print('Initializing erosion function')
        for col in range(len(img)):
            for row in range(len(img[col])):
                if img[col, row] == 1:
                    if col > 0 and img[col - 1][row] == 0: img[col - 1][row] = 2
                    if row > 0 and img[col][row - 1] == 0: img[col][row - 1] = 2
                    if col + 1 < len(img) and img[col + 1][row] == 0: img[col + 1][row] = 2
                    if row + 1 < len(img[col]) and img[col][row + 1] == 0: img[col][row + 1] = 2

        for col in range(len(img)):
            for row in range(len(img[col])):
                if (img[col][row] == 2): img[col][row] = 1

        return img