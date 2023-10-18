import cv2
import numpy as np

img = cv2.imread('C:/Users/hneul/Downloads/img2/block.jpg')
# HSV 색공간으로 변환 및 분리
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# s(채도)와 v(밝기)공간에 대해 adaptive threshold 적용하여 이진화
thresh0 = cv2.adaptiveThreshold(s, 255,
cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 11)
thresh1 = cv2.adaptiveThreshold(v, 255,
cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 13)
# binary 이미지 통합
thresh = cv2.bitwise_or(thresh0, thresh1)
cv2.imshow('Image-thresh', thresh)
cv2.imshow('Image-thresh0', thresh0)
cv2.imshow('Image-thresh1', thresh1)
kernel = np.ones((3, 3), np.uint8)
img = cv2.dilate(thresh, kernel, iterations=1)
cv2.imshow('1', img)
kernel = np.ones((3, 3), np.uint8)
img = cv2.erode(img, kernel, iterations=2)
cv2.imshow('2', img)
kernel = np.ones((5, 5), np.uint8)
img = cv2.dilate(img, kernel, iterations=3)
cv2.imshow('3', img)
img = cv2.erode(img, kernel, iterations=3)
cv2.imshow('4', img)
cv2.imwrite('img/box_seg.png',img)
cv2.waitKey(0)
cv2.destroyAllWindows()