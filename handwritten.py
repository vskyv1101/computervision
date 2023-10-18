import cv2
import numpy as np

image = cv2.imread('C:/Users/hneul/OneDrive/바탕 화면/handwritten.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
cv2.imshow('binary', binary_image)

kernel_erode = np.ones((3,3), np.uint8)
dilated_image = cv2.dilate(binary_image, kernel_erode, iterations=1)
cv2.imshow('dilated image', dilated_image)

eroded_image = cv2.erode(dilated_image, kernel_erode, iterations=6)
cv2.imshow('eroded image', eroded_image)

inverted_img = cv2.bitwise_not(eroded_image)
cv2.imshow('inverted image', inverted_img)

cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_img)
dst = image

for i in range(1, cnt):
    x,y,w,h,area = stats[i]
    cv2.rectangle(dst(x,y), (x+w, y+h), (0,255,255))
cv2.imshow('connected component', dst)

cv2.imwrite('img/handwritten_connectedcomponent.png', dst)
cv2.imwrite('img/handwritten_seg.peg', eroded_image)

cv2.waitkey(0)
cv2.destroyAllWindows()