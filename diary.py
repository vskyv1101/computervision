import cv2
import numpy as np
# 이미지 불러오기
img = cv2.imread('C:/Users/hneul/Downloads/img2/diary.png')
img_contour = img.copy()
# 그레이스케일로 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 이진화
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
thresh = cv2.bitwise_not(thresh)
binary_img_line = np.zeros_like(thresh)
# 캐니 에지 검출
edges = cv2.Canny(gray, 0, 150, apertureSize=3)
cv2.imshow('edges', edges)
# 허프 라인 변환
lines = cv2.HoughLines(edges, 1, np.pi/180, 230)
for line in lines:
    rho, theta = line[0]
a = np.cos(theta)
b = np.sin(theta)
x0 = a * rho
y0 = b * rho
x1 = int(x0 + 1000*(-b))
y1 = int(y0 + 1000*(a))
x2 = int(x0 - 1000*(-b))
y2 = int(y0 - 1000*(a))
cv2.line(binary_img_line, (x1, y1), (x2, y2), (255), 1)
cv2.imshow('hough line', binary_img_line)
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
binary_img = np.zeros_like(thresh)
cv2.drawContours(binary_img, contours, -1, (255), 2) # 마지막 인자는 선 두께cv2.drawContours(img_contour, contours, -1, (0, 255, 0), 3)
cv2.imshow('Contours', binary_img)
empty_note_bw = cv2.bitwise_and(binary_img, binary_img_line)
cv2.imshow('empty_note_bw', empty_note_bw)
empty_note_color = img.copy()
empty_note_color[empty_note_bw==0] = (255,255,255)
cv2.imshow('empty_note_color',empty_note_color)
cv2.waitKey(0)
cv2.destroyAllWindows()