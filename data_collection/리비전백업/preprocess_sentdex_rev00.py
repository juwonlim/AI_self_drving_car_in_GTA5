# === preprocess.py ===

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #이건 현재 파일 기준으로 상위 폴더(=루트) 를 자동으로 찾아주는 방식이야
import cv2
import numpy as np
from PIL import ImageGrab




""" 
def region_of_interest(screen, vertices):
    mask = np.zeros_like(screen)
    if len(screen.shape) > 2:
        channel_count = screen.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    masked_image = cv2.bitwise_and(screen), mask)
    return masked_image
"""





def grab_screen():
    screen = np.array(ImageGrab.grab(bbox=(0, 40, 1280, 720)))
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
    return screen





def region_of_interest(screen, vertices):
    mask = np.zeros_like(screen)
    if len(screen.shape) > 2:
        channel_count = screen.shape[2]
        ignore_mask_color = (0, 255, 0)  # 연두색 표시
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    masked = cv2.bitwise_and(screen, mask)
    masked_image = cv2.addWeighted(masked, 1, mask, 0.3, 0)  # 반투명 연두색 마스크 시각화
    return masked



def draw_roi_polygon(img, vertices, color=(0, 255, 0), thickness=5):
    """
    ROI 영역을 사다리꼴로 시각화하여 그려주는 함수
    - img: 원본 이미지 (컬러)
    - vertices: ROI 꼭지점들
    - color: 선 색상
    - thickness: 선 두께
    """
    # 폴리라인은 꼭지점을 하나의 배열로 받아야 함
    cv2.polylines(img, [vertices], isClosed=True, color=color, thickness=thickness)
    return img


"""

def overlay_roi_mask(img, vertices, color=(0, 255, 0), alpha=0.3):
    
    #img: 원본 이미지 (BGR)
    #vertices: ROI 꼭짓점 (np.array)
    #color: 마스크 색상 (기본 연두색)
    #alpha: 투명도 (0~1, 낮을수록 더 투명)
    
    mask = np.zeros_like(img)  # 이미지와 동일한 크기의 빈 mask 생성
    cv2.fillPoly(mask, [vertices], color)  # ROI 영역만 색칠
    overlayed = cv2.addWeighted(img, 1, mask, alpha, 0)  # 원본 + mask를 합성
    return overlayed

 """


def preprocess_img(masked):
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_img = cv2.Canny(blur, 100, 200)

    # Canny 디버깅
    resized = cv2.resize(canny_img, (426, 240))
    cv2.imshow("Canny", resized)
    cv2.waitKey(1)

    # ROI 설정
    roi_vertices = np.array([[(0, 560), (0, 350), (320, 200), (960, 200), (1280, 350), (1280, 560)]], dtype=np.int32)
    print("ROI vertices:", roi_vertices)

    #ROI 폴리라인을 원본 이미지 위에 그림
    roi_debug = masked.copy()  # 컬러 원본

    #draw_roi_polygon호출전에 roi_debug가 3채널 BGR인지 확인 또는 변환
    #cv2.polylines()는 BGR(3채널) 이미지에서만 색을 제대로 표현 가능하기 때문
    if len(roi_debug.shape) == 2 or roi_debug.shape[2] == 1:
        roi_debug = cv2.cvtColor(roi_debug, cv2.COLOR_GRAY2BGR)

    #roi_debug = draw_roi_polygon(roi_debug, roi_vertices[0], color=(0, 255, 0), thickness=5)
    for point in roi_vertices[0]:
        #cv2.circle(roi_debug, tuple(point), 10, (255, 0, 0), -1)  # 파란 점 찍기
        cv2.circle(roi_debug, tuple(point), 12, (0, 0, 255), -1)  # 굵은 빨간 점


    roi_debug = draw_roi_polygon(roi_debug, roi_vertices[0], color=(0, 0, 255), thickness=10)
    cv2.imwrite("roi_debug_output.jpg", roi_debug)


    roi_debug_resized = cv2.resize(roi_debug, (426, 240))
    cv2.imshow("ROI Debug Polygon", roi_debug_resized)
    cv2.waitKey(1)

    # ROI 마스크 적용 → 흑백 canny에 마스크
    roi_img = region_of_interest(canny_img, roi_vertices)
    return roi_img









""" 

def preprocess_img(masked):
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_img = cv2.Canny(blur, 100, 200) 

    #Canny 디버깅 창 추가
    resized = cv2.resize(canny_img, (426, 240))
    cv2.imshow("Canny", resized)
    cv2.waitKey(1)



    #roi_vertices = np.array([[10, 500], [10, 300], [300, 200],[1000, 200], [1270, 300], [1270, 500]], dtype=np.int32)
    roi_vertices = np.array([[(0, 560), (0, 350), (320, 200), (960, 200), (1280, 350), (1280, 560)]], dtype=np.int32)
    print("ROI vertices:", roi_vertices)
    #gaussian_blur = cv2.GaussianBlur(canny_img,(5,5),0)


    # ROI 라인 시각화 (원본 이미지 위에 덧그림)
    #debug_img = original_img.copy()
    #debug_img = resized.copy()
    debug_img = masked.copy() # region_of_interest 함수의 리턴값 masked
    debug_img = draw_roi_polygon(debug_img, roi_vertices[0])  # 꼭지점 배열에서 [0]만 전달해야 함
    debug_img_resized = cv2.resize(debug_img, (426, 240))
    cv2.imshow("ROI Debug Polygon", debug_img_resized)
    cv2.waitKey(1)
    


    roi_img = region_of_interest(canny_img, roi_vertices) #roi함수에 canny결과와 roi좌표값을 집어넣음
    #cv2.imshow("ROI after Masking", roi_img)
    #cv2.waitKey(1)
    return roi_img
    """

#grab screen에서 스크린 영역 캡쳐 --> ROI함수로 연두색 영역 표시 --> preprocess_img함수로 regiion_of_interest의 vertices값과 canny함수처리된 것을 넘김 --> 가우시안불러--> roi