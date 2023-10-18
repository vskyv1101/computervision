import cv2
import numpy as np
# 이미지를 읽기
image = cv2.imread("C:/Users/hneul/Downloads/img/carnum.jpg")
image = cv2.blur(image, (7,7))
# 이미지를 그레이스케일로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 이진 이미지 획득
_, binary_image = cv2.threshold(gray, 0, 255,
cv2.THRESH_OTSU)
cv2.imshow('binary', binary_image)
# 팽창(반전영상의 침식) 연산을 통해, 추가적인 노이즈 제거
kernel_erode = np.ones((3, 3), np.uint8)
dilated_image = cv2.dilate(binary_image, kernel_erode,
iterations=2)
cv2.imshow('dilated image', dilated_image)
# 글자 Blob을 찾기 위해, 글자를 하얗게 변경 (반전)
inverted_img = cv2.bitwise_not(dilated_image)
cv2.imshow('inverted image', inverted_img)
cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_img)
dst = image
# 임의의 threshold를 통해 글자에 해당되는 bbox만 남김
for i in range(1, cnt):
    x, y, w, h, area = stats[i]
    if w>20 and w<100:
        cv2.rectangle(dst, (x, y), (x+w, y+h), (0, 255, 255))
cv2.imshow('connected component', dst)
# 이미지 처리 결과를 저장
cv2.imwrite('img/carnum_connectedcomponent.png',dst)
cv2.imwrite('img/carnum_seg.png',dilated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()