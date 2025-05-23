
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #이건 현재 파일 기준으로 상위 폴더(=루트) 를 자동으로 찾아주는 방식

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), 'resources')

digits = np.load(os.path.join(RESOURCE_PATH, 'digits.npy'))
digits_labels = np.load(os.path.join(RESOURCE_PATH, 'digits_labels.npy'))
arrows = np.load(os.path.join(RESOURCE_PATH, 'arrows.npy'))
arrows_labels = np.load(os.path.join(RESOURCE_PATH, 'arrows_labels.npy'))

import win32gui
import win32ui
import cv2
import win32con
import pyautogui


def capture_screen():
    screen = np.array(pyautogui.screenshot())
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
    return screen

''' 
chatgpt가 만들어준 함수
def crop(image):
    # 바이크 기준으로 하늘/차량 제외: Y 150~600 사용
    return image[150:600, :, :]
'''


#lane_detect.py에서 이동해온 함수
def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    화면 상단의 하늘과 하단의 자동차 전면부를 제거하여
    관심 영역만 잘라냄 (약 280~590픽셀 높이만 사용)
    """
    #화면 상단과 하단을 적당히 자르는 함수.
    #위쪽은 하늘 제거, 아래쪽은 차량 대시보드 제거였는데 나는 바이크로 훈련시킬 예정이라 대시보드 무의미, 그냥 바이크 쉴드 밑은 안쓴다 정도
    image = image[150:600, :, :] 
    print("Image shape after crop:", image.shape)  # crop 직후
    return image # 예: 높이 720 기준으로 중간 부분 400픽셀 확보, 순서는 Y축,X축,채널


def resize(image, size=(1280, 450)):
    return cv2.resize(image, size)


def hsv_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(mask_white, mask_yellow)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image, mask

''' 
chatgpt가 만들어준 함수
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
'''

#lane_detect.py에서 이동해온 함수
def grayscale(img):
    """
    Applies the Grayscale transform
    This will return an image with only one color channel
    컬러 이미지를 흑백 이미지로 변환
    → Canny edge detection 등 전처리에 사용됨
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

''' 
chatgpt가 만들어준 함수
def gaussian_blur(image, kernel_size=7):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

'''


#lane_detect.py파일에서 이동됨
def gaussian_blur(img, kernel_size):
    """
    Applies a Gaussian Noise kernel
    가우시안 블러 필터 적용
    → 노이즈 제거 및 가장자리 부드럽게 처리
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=30, sigmaY=30)


''' 
chatgpt가 만들어준 함수
def canny(image, low_threshold=50, high_threshold=100):
    return cv2.Canny(image, low_threshold, high_threshold)
'''



#lane_detect.py에서 이동
def canny(img, low_threshold, high_threshold):
    """
    Applies the Canny transform
      Canny 엣지 검출 적용
    → 경계선을 뚜렷하게 검출하는 데 사용
    """
    return cv2.Canny(img, 30, 100)

''' 
chatgpt가 만들어준 함수
def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
'''

#lane_detect.py에서 이동한 함수
def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.

    이미지 마스킹 함수
    - 다각형 `vertices`로 정의된 영역만 남기고 나머지 부분은 제거
    - 보통 도로 영역만 남기기 위해 사용됨
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
     # 채널 수에 따라 마스크 컬러 정의 (RGB or 흑백)
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
     # 정의된 다각형 영역만 마스크에 채움
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    # 마스크 영역과 원본 이미지 AND 연산 → ROI 추출
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image




#get_preprocessed()는 실제 사용 시 screen 인자를 받고 있으니, 호출부에 명확하게 screen을 넣고, 그 내용도 preprocess.py에서 반영하면 가독성 개선, ( ) 로 남겨두는 것보다 나음
def get_preprocessed(screen):
    #screen = capture_screen()
    cropped = crop(screen)
    resized = resize(cropped)
    masked_image, mask = hsv_mask(resized)
    gray = grayscale(masked_image)
    blurred = gaussian_blur(gray, kernel_size=7)
    canny_edges = canny(blurred,low_threshold=50, high_threshold=100 )
    roi_vertices = np.array([[(0, 0), (0, 450), (1280, 450), (1280, 0)]], np.int32)
    roi = region_of_interest(canny_edges, roi_vertices)

    return {
        "screen": screen,
        "cropped": cropped,
        "resized": resized,
        "masked": masked_image,
        "mask": mask,
        "gray": gray,
        "blurred": blurred,
        "roi": roi,
        "canny_edge_lines": canny_edges
        
    }

