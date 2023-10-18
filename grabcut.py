import cv2
import numpy as np

drawing = False
ix, iy = -1, -1
rect_over = False
x, y = -1, -1
# 마우스 콜백 함수
def draw_rectangle(event, _x, _y, flags, param):
    global ix, iy, drawing, img, img2, rect_over, x, y # x, y를 전역 변수로 추가if event == cv2.EVENT_LBUTTONDOWN:
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = _x, _y # _x, _y 사용
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (_x, _y), (0, 255, 0), 2)
        rect_over = True
        x, y = _x, _y # 마우스 뗄 때 위치 전역 변수저장
        
img = cv2.imread('C:/Users/hneul/Downloads/img/box.jpg')
img2 = img.copy()

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rectangle)

while(1):
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27: # ESC 키 누르면 종료
        break
    elif k == ord('r') and rect_over: # r 키 누르면 GrabCut 적용
        mask = np.zeros(img.shape[:2],np.uint8)
        bgd_model = np.zeros((1,65),np.float64)
        fgd_model = np.zeros((1,65),np.float64)
        rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
        cv2.grabCut(img2,mask,rect,bgd_model,fgd_model,5,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img2*mask2[:,:,np.newaxis]
        rect_over = False
        
cv2.destroyAllWindows()
