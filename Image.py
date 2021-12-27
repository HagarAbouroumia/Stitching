import numpy as np
import cv2


class Image:

    def __init__(self, image):
        self.image = image

    def get_corners(self):
        return np.float32([[0, 0], [self.shape(1), 0], [self.shape(1), self.shape(0)], [0, self.shape(0)]]).reshape(-1,
                                                                                                                    1,
                                                                                                                    2)

    def set_resize(self, h, w):
        self.image = cv2.resize(self.image, (w, h), interpolation=cv2.INTER_AREA)

    def get_resize(self, h, w):
        return cv2.resize(self.image, (w, h), interpolation=cv2.INTER_AREA)

    def shape(self, x):
        return self.image.shape[x]

    def show(self):
        temp = self.get_resize(500, 500)
        cv2.imshow("selfImage", temp)
        cv2.waitKey(0)

    def get_image(self):
        return self.image

    def get_copy(self):
        return self.image.copy()

    def set_pixels(self, x, y, img):
        self.image[0:x, 0:y] = img

    def set_image(self, image):
        self.image = image

    def write_image(self, name):
        cv2.imwrite(name, self.image)
