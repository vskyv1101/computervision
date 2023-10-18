import cv2
import numpy as np
# 이미지를 읽기
image = cv2.imread('C:/Users/hneul/Downloads/img2/face.jpg')
image = cv2.blur(image, (3,3))
# 이미지를 그레이스케일로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 이미지 이진화를 수행
_, binary = cv2.threshold(gray, 160, 255,
cv2.THRESH_BINARY)
cv2.imshow('Processed binary Image', binary)

red = np.full((image.shape[0],image.shape[1],3), 255, dtype='uint8')
red[binary==0] = (0,0,255)
cv2.imshow('red',red)
blue = np.full((image.shape[0],image.shape[1],3), 255, dtype='uint8')
blue[binary==0] = (255,0,0)
cv2.imshow('blue',blue)
green = np.full((image.shape[0],image.shape[1],3), 255, dtype='uint8')
green[binary==0] = (0,255,0)
cv2.imshow('green',green)
black = np.full((image.shape[0],image.shape[1],3), 255, dtype='uint8')
black[binary==0] = (0,0,0)
c1 = cv2.hconcat([black, red])
c2 = cv2.hconcat([blue, green])
c3 = cv2.vconcat([c1,c2])

cv2.waitKey(0)
cv2.destroyAllWindows()