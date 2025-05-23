
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


""" 
#스크린 전체 캡쳐라고 해서 일단 주석처리
def capture_screen():
    screen = np.array(pyautogui.screenshot())
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
    return screen
"""
#navigation_img_proces.py에서 가져온 함수
# Done by Frannecklp
# GTA5 창을 캡처해서 numpy 이미지로 반환
# data_collect.py실행하여 k눌러서 차량주행시에 Saving data and closing the program으로 바로 종료되는 경우
# chatgpt : img_process.py의 grab_screen() 함수가 캡처 실패 시 None을 반환하고, 이후에 screen을 사용하는 부분에서 문제가 발생해 cv2.imshow() 같은 함수가 빈 프레임(None) 을 받아 종료되는 가능성
#이 함수의 용도 : 화면캡쳐 (mss)사용,공통 이미지 전처리 용도가 아님-->그러나 GTA5의 게임창 크기가 무엇이던지 자동캡쳐하는 기능이기에 PREPROCESS.PY로 이동이 맞다 판단. 

def grab_screen(winName: str = "Grand Theft Auto V"):
    desktop = win32gui.GetDesktopWindow()

    # get area by a window name
    gtawin = win32gui.FindWindow(None, winName)
    # get the bounding box of the window
    left, top, x2, y2 = win32gui.GetWindowRect(gtawin)
   
    # cut window boarders
    # 윈도우 프레임 보정 (타이틀바 + 테두리)
    top += 32
    left += 3
    y2 -= 4
    x2 -= 4
    width = x2 - left + 1
    height = y2 - top + 1

    # the device context(DC) for the entire window (title bar, menus, scroll bars, etc.)
    hwindc = win32gui.GetWindowDC(desktop)
    # Create a DC object from an integer handle
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    # Create a memory device context that is compatible with the source DC
    memdc = srcdc.CreateCompatibleDC()
    # Create a bitmap object
    bmp = win32ui.CreateBitmap()
    # Create a bitmap compatible with the specified device context
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    # Select an object into the device context.
    memdc.SelectObject(bmp)
    # Copy a bitmap from the source device context to this device context
    # parameters: destPos, size, dc, srcPos, rop(the raster operation))
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)



    # the bitmap bits
     # 비트맵 → numpy 이미지
    signedIntsArray = bmp.GetBitmapBits(True)
    # form a 1-D array initialized from text data in a string.
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)



    # Delete all resources associated with the device context
    # 리소스 해제
    srcdc.DeleteDC()
    memdc.DeleteDC()
    # Releases the device context
    win32gui.ReleaseDC(desktop, hwindc)
    # Delete the bitmap and freeing all system resources associated with the object.
    # After the object is deleted, the specified handle is no longer valid.
    win32gui.DeleteObject(bmp.GetHandle())

    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    #grab이 잘되는지 확인용도
    #cv2.imshow("GRAB_SCREEN_DEBUG", img)
    #print("grab_screen captured shape:", img.shape)
    #cv2.waitKey(1)

    return img # RGB로 변환하여 리턴


#lane_detect.py에서 이동해온 함수

""" 
#sentdex처럼 하기 위해서 주석처리 되는 부분들
def crop(image):
    
    #Crop the image (removing the sky at the top and the car front at the bottom)
    #화면 상단의 하늘과 하단의 자동차 전면부를 제거하여
    #관심 영역만 잘라냄 (약 280~590픽셀 높이만 사용)
    
    #화면 상단과 하단을 적당히 자르는 함수.
    #위쪽은 하늘 제거, 아래쪽은 차량 대시보드 제거였는데 나는 바이크로 훈련시킬 예정이라 대시보드 무의미, 그냥 바이크 쉴드 밑은 안쓴다 정도
    image = image[150:600, :, :] 
    print("Image shape after crop:", image.shape)  # crop 직후
    return image # 예: 높이 720 기준으로 중간 부분 400픽셀 확보, 순서는 Y축,X축,채널


def resize(image, size=(1280, 450)):
    return cv2.resize(image, size)
"""



def hsv_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #lower_white = np.array([0, 0, 200])
    lower_white = np.array([0, 0, 130]) #night
    #upper_white = np.array([180, 40, 255])
    upper_white = np.array([180, 70, 255]) #night
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(mask_white, mask_yellow)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    img = masked_img
    #resized = cv2.resize(img, (640, 360))
    #cv2.imshow("hsv_mask", resized)
    return img, mask #tuple로 리턴


#lane_detect.py에서 이동해온 함수
def grayscale(img):
    """
    Applies the Grayscale transform
    This will return an image with only one color channel
    컬러 이미지를 흑백 이미지로 변환
    → Canny edge detection 등 전처리에 사용됨
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #hsv_mask함수의 tuple로 들어오는 img값은 cv2가 처리할 수 없는 에러를 낸다, TypeError: src is not a numerical tuple
    #resized = cv2.resize(img, (640, 360))
    #cv2.imshow("grayscale", resized)

    return img




#lane_detect.py파일에서 이동됨
def gaussian_blur(img, kernel_size):
    """
    Applies a Gaussian Noise kernel
    가우시안 블러 필터 적용
    → 노이즈 제거 및 가장자리 부드럽게 처리
    """
    #return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=30, sigmaY=30)
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=0)
    #resized = cv2.resize(img, (640, 360))
    #cv2.imshow("gaussian_blur", resized)
    return img





#lane_detect.py에서 이동
def canny(img):
    """
    Applies the Canny transform
      Canny 엣지 검출 적용
    → 경계선을 뚜렷하게 검출하는 데 사용
    """
    low_threshold=100
    high_threshold=200
    img = cv2.Canny(img, low_threshold, high_threshold)
    resized = cv2.resize(img, (320, 180))
    cv2.imshow("canny", resized)
    cv2.waitKey(1)

    return img
 


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
    #ROI 점들은 순서대로 시계방향 또는 반시계방향으로 도형을 그릴 수 있는 순서로 좌표를 주어야 함
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
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img






def get_preprocessed():
    
    original_img = grab_screen()
    if original_img is None or original_img.size == 0:
        print("[ERROR] get_preprocessed(): screen is None or empty.")
        print("[ERROR] grab_screen failed.")
        return None

    
     #함수가 받은 screen을 모두 grab_screen이 덮어쓰기 떄문에 주석처리 해야된다는 chatgpt의 말이었는데 ㅣlane_detect.py쪽에서 같은 코드를 주석처리했음
    #cropped = crop(screen)
    #resized = resize(cropped)
    #masked_img, mask = hsv_mask(original_img)

    #masked_img = hsv_mask(original_img) #hsv_mask함수는 return img, mask  이렇게 튜플로 리턴인데 여기는 original_img하나만 있음.
    masked_img, _ = hsv_mask(original_img)  # mask는 지금 필요 없으니 버림, tuple 언패킹
    gray = grayscale(masked_img)
    
    blurred = gaussian_blur(gray, kernel_size=5)
    canny_edges = canny(blurred)
    #roi_vertices = np.array([[(0, 560),(0, 350),(400, 200),(900, 200),(1280, 350),(1280, 560)]], dtype=np.int32)
    roi_vertices = np.array([[(0, 560), (0, 350), (320, 200), (960, 200), (1280, 350), (1280, 560)]], dtype=np.int32)

    roi = region_of_interest(canny_edges, roi_vertices)

    return roi, original_img #칼라 GTA5이미지인 original_img도 같이 리턴함

""" 
#너무 리턴값이 많아서 주석처리
    return {
        "screen": original_img, #grab screen으로 가져온 결과
        "masked": masked_img, #hsv_mask함수에서 grab_screen결과를 받아서 처리
        #"mask": mask, #hsvmask가 반환하는 참조인자(참고용)
        "gray": gray, #hsvmask가 변환하는 결과를 grayscale함수에서 받아서 처리
        "blurred": blurred, #가우시안 블러함수에서 grayscale함수값을 받아서 처리후 반환
        "canny_edge_lines": canny_edges, #canny함수는 가우시안함수의 반환값을 받아서 처리
        "roi": roi #roi함수는 canny_edges함수의 반환값과 np.array로 정의된 roi좌표값을 받아서 반환
      
        
    }
"""