import cv2
import numpy as np
# 이미지 불러오기
img = cv2.imread('C:/Users/hneul/Downloads/img2/box.jpg')
# 초기 마스크 생성
mask = np.zeros(img.shape[:2], np.uint8)
# 배경과 전경 모델 생성
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)
# 전경이 될 영역을 직사각형으로 지정
rect = (50, 50, 450, 290) # 임의 조정된 영역
# GrabCut 알고리즘 실행
cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5,
cv2.GC_INIT_WITH_RECT)
# 마스크를 2D 이미지로 변환
mask_2d = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# 마스크를 3D 이미지로 확장하고 원본 이미지에 적용
mask_color = mask_2d[:, :, np.newaxis]
img_result = img * mask_color
# 결과 출력
cv2.imshow('Original', img)
cv2.imshow('GrabCut', img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()