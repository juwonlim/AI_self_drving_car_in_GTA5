"""
Module for preprocessing screen captures
"""


"""
Module for preprocessing screen captures from GTA5.
- 화면을 캡처하고
- 속도 숫자 (세자리) + 방향 화살표 인식
- 이미지 리사이즈 후 반환
"""



import win32gui
import win32ui
import cv2
import numpy as np
import win32con


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #이건 현재 파일 기준으로 상위 폴더(=루트) 를 자동으로 찾아주는 방식이야


# ⬇️ KNN 모델 로딩 함수
def initKNN(data, labels, shape):
    knn = cv2.ml.KNearest_create()
    train = np.load(data).reshape(-1, shape).astype(np.float32)
    train_labels = np.load(labels)
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    return knn

# 숫자와 방향 화살표 인식용 KNN 모델 (미리 학습된 .npy 사용)
knnDigits = initKNN('..\data_collection\\resources\digits.npy',
                    '..\data_collection\\resources\digits_labels.npy', 40)
knnArrows = initKNN('..\data_collection\\resources\\arrows.npy',
                    '..\data_collection\\resources\\arrows_labels.npy', 90)


# Done by Frannecklp
# GTA5 창을 캡처해서 numpy 이미지로 반환
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

    return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB) # RGB로 변환하여 리턴


# KNN 예측 호출 함수
def predict(img, knn):
    ret, result, neighbours, dist = knn.findNearest(img, k=1)
    return result

# 이미지 전처리: 흑백 + 이진화
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, -5)
    return thr

# 세 자리 숫자 → 실제 속도 정수로 변환
def convert_speed(num1, num2, num3):
    hundreds = 1
    tens = 1
    speed = 0

    if num3[0][0] != 10:
        hundreds = 10
        tens = 10
        speed += int(num3[0][0])
    if num2[0][0] != 10:
        speed += tens * int(num2[0][0])
        hundreds = tens * 10
    if num1[0][0] != 10:
        speed += hundreds * int(num1[0][0])

    return speed

# 전체 전처리 루틴
def img_process(winName: str = "Grand Theft Auto V"):
    screen = grab_screen(winName)

    # Ji Hyun's computer
    # 👇 여기가 "속도 숫자 영역" 추출 좌표 (3자리 숫자)
    # 해상도 1280x720 기준이고, GTA5가 모니터 좌측 상단에 위치할 경우
    numbers = preprocess(screen[567:575, 683:702, :])
    # Rustam's computer
    # numbers = preprocess(screen[573:581, 683:702, :])

    # three fields for numbers
    # 숫자 세자리 분리 예측
    num1 = predict(numbers[:, :5].reshape(-1, 40).astype(np.float32), knnDigits)
    num2 = predict(numbers[:, 7:12].reshape(-1, 40).astype(np.float32), knnDigits)
    num3 = predict(numbers[:, -5:].reshape(-1, 40).astype(np.float32), knnDigits)

    # one field for direction arrows
    # Ji Hyun's computer
      # 👇 방향 화살표 인식 영역 (좌하단 또는 센터 왼쪽)
    direct = preprocess(screen[561:570, 18:28, :]).reshape(-1, 90).astype(np.float32)  #screen[567:575, 683:702, :] ← 이건 이미지에서의 좌표 슬라이싱이 맞고, 이 좌표가 GTA5 화면 상에서 속도계 숫자 영역에 정확히 대응하는지는, 실제 해보기 전까진 확실하게 모른다.


    # Rustam's computer
    # direct = preprocess(screen[567:576, 18:28, :]).reshape(-1, 90).astype(np.float32)
    direct = int(predict(direct, knnArrows)[0][0])

    speed = convert_speed(num1, num2, num3)
    resized = cv2.resize(screen, (320, 240)) # 학습용 이미지 크기로 변환

    return screen, resized, speed, direct


"""
왜 해보기 전까지 알 수 없냐?
이건 실제로 운영체제의 창 위치, GTA5의 HUD 위치, 게임 해상도,
그리고 모니터에서 화면이 정확히 어디에 떠 있는지에 따라 픽셀 좌표가 바뀔 수 있기 때문이야.

특히 GTA5는 창 모드라도 Windows의 창 테두리, 타이틀바(32px), 그림자 등이 추가되기 때문에,
좌표는 항상 1:1 정해진 게 아냐.

그래서 가장 현실적인 방법은?
캡처된 이미지에서 숫자가 제대로 잡히는지 직접 시각화해서 확인하는 거야.

 예시 코드 (테스트용):
python
복사
편집
screen, resized, speed, direction = img_process()

# 속도 숫자 영역만 보기
cv2.imshow("Speed Area", screen[567:575, 683:702])
cv2.imshow("Full Frame", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
→ 이렇게 하면 그 좌표에 실제 숫자가 있는지 육안으로 바로 확인 가능해.
만약 검은 화면이거나 숫자 아닌 부분이 뜬다면 → 좌표 수정 필요.

좌표 튜닝 팁
OpenCV에서 캡처한 전체 이미지 사이즈가 1280x720과 같다고 해도
게임창 내부 콘텐츠가 그보다 작을 수 있어.

따라서 offset 조정은 ±10~30px 단위로 수동 미세조정이 필요해.

결론:

저 좌표가 실제 속도계에 대응돼? 직접 해보기 전까지 절대 모름
어떻게 확인해?  시각화해서 숫자가 잘리는지 여부로 확인
맞지 않으면? 좌표 screen[y1:y2, x1:x2] 범위를 조정하면 됨
"""

