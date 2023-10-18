import cv2

import numpy as np

image = cv2.imread("C:/Users/hneul/Downloads/img/cat.jpg")
image = cv2.blur(image, (7,7))
gray = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)

kernel = np.ones((3,3), np.uint8)
eroded_image = cv2.erode(binary, kernel,
iterations=6)

kernel_erode = np.ones((3, 3), np.uint8)
dilated_image = cv2.dilate(eroded_image, kernel_erode, iterations=7)
cv2.imshow('img/cat_seg.png', dilated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()