import math

import cv2
import numpy as np

# GTA5 게임 화면을 캡처하는 함수 (img_process에서 가져옴)
from data_collection.img_process import grab_screen

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #이건 현재 파일 기준으로 상위 폴더(=루트) 를 자동으로 찾아주는 방식이야


# 이전 프레임에서 검출한 차선 정보를 저장하는 전역 변수
# [left_line, right_line, stop_line]
prev_lines = [[], [], []]


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    화면 상단의 하늘과 하단의 자동차 전면부를 제거하여
    관심 영역만 잘라냄 (약 280~590픽셀 높이만 사용)
    """
    return image[280:-130, :, :]


def grayscale(img):
    """
    Applies the Grayscale transform
    This will return an image with only one color channel
    컬러 이미지를 흑백 이미지로 변환
    → Canny edge detection 등 전처리에 사용됨
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold=100, high_threshold=300):
    """
    Applies the Canny transform
      Canny 엣지 검출 적용
    → 경계선을 뚜렷하게 검출하는 데 사용
    """
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """
    Applies a Gaussian Noise kernel
    가우시안 블러 필터 적용
    → 노이즈 제거 및 가장자리 부드럽게 처리
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=30, sigmaY=30)


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


def construct_lane(lines):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the add_images() function below


    허프라인 검출 결과(lines)를 기반으로:
    - 왼쪽/오른쪽 차선 구분
    - 정지선 검출
    - 평균 선 생성 및 이전 프레임(prev_lines)과의 비교를 위한 준비

    반환값:
    - lane = [left_line, right_line]
    - stop_line = [첫 번째 점, 두 번째 점]
    """
      # 왼쪽 차선 후보점들
    left_line_x = []#
    left_line_y = []

    # 오른쪽 차선 후보점들
    right_line_x = []
    right_line_y = []

    # 정지선 후보점들 (가로선)
    stop_line_x_first = []
    stop_line_y_first = []
    stop_line_x_second = []
    stop_line_y_second = []

     # 최종 반환할 차선, 정지선
    lane = [[], []]
    stop_line = []

       # 차선의 y 좌표 범위 설정
    min_y = 0
    max_y = 190

#여기까지 	기본 import, 이전 프레임 상태 변수, 영상 전처리 함수 (crop, grayscale, canny 등),관심영역 마스킹 (region_of_interest),construct_lane 함수 초기 설정 (차선 후보점 수집용 리스트 등)

########################## 여기까지 #1
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:

                 # 기울기 계산 (x1 == x2일 경우 분모 0 방지)
                slope = (y2 - y1) / (x2 - x1) if x1 != x2 else 0  # <-- Calculating the slope.
                
                 # 기울기가 너무 완만한 선은 무시 (정지선도 아님)
                if 0.05 < math.fabs(slope) < 0.3:  # not interested
                    continue

                # 거의 수평에 가까운 선 = 정지선 후보
                if math.fabs(slope) <= 0.05:  # stop line
                    if (y1 > 20) and (y2 > 20):
                         # 상단 정지선, 하단 정지선으로 나눠 저장
                        # we need to detect two stop lines (top and bottom)
                        if not stop_line_x_first or abs(stop_line_y_first[0] - y1) < 15:
                            stop_line_x_first.extend([x1, x2])
                            stop_line_y_first.extend([y1, y2])
                        else:
                            stop_line_x_second.extend([x1, x2])
                            stop_line_y_second.extend([y1, y2])
                
                # 음의 기울기 → 왼쪽 차선
                elif slope <= 0:  # <-- If the slope is negative, left group.
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                
                  # 양의 기울기 → 오른쪽 차선
                else:  # <-- Otherwise, right group.
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])

        offset = 7 # 이전 프레임과의 차이 허용 범위 (스무딩 효과)

        # ================= LEFT LANE =================
        if left_line_x:
            # 왼쪽 차선 추정 선형 모델 (1차 회귀)
            poly_left = np.poly1d(np.polyfit(
                left_line_y,
                left_line_x,
                deg=1
            ))

            x1 = int(poly_left(max_y)) # 하단 점
            x2 = int(poly_left(min_y))  # 상단 점
            
             # 이전 프레임과 비교하여 변화량 제한
            if prev_lines[0]:
                # recalculate x1
                if abs(x1 - prev_lines[0][0]) > offset:
                    x1 = prev_lines[0][0] - offset if prev_lines[0][0] > x1 else prev_lines[0][0] + offset
                # recalculate x2
                if abs(x2 - prev_lines[0][1]) > offset:
                    x2 = prev_lines[0][1] - offset if prev_lines[0][1] > x2 else prev_lines[0][1] + offset

            prev_lines[0] = [x1, x2]
            lane[0] = [x1, max_y, x2, min_y]  # 좌측 차선 저장
        elif prev_lines[0]:
            lane[0] = [prev_lines[0][0], max_y, prev_lines[0][1], min_y]
            prev_lines[0] = []


         # ================= RIGHT LANE =================
        if right_line_x:
            poly_right = np.poly1d(np.polyfit(
                right_line_y,
                right_line_x,
                deg=1
            ))

            x1 = int(poly_right(max_y))
            x2 = int(poly_right(min_y))
            if prev_lines[1]:
                # recalculate x1
                if abs(x1 - prev_lines[1][0]) > offset:
                    x1 = prev_lines[1][0] - offset if prev_lines[1][0] > x1 else prev_lines[1][0] + offset
                # recalculate x2
                if abs(x2 - prev_lines[1][1]) > offset:
                    x2 = prev_lines[1][1] - offset if prev_lines[1][1] > x2 else prev_lines[1][1] + offset

            prev_lines[1] = [x1, x2]
            lane[1] = [x1, max_y, x2, min_y]  # 우측 차선 저장
        elif prev_lines[1]:
            lane[1] = [prev_lines[1][0], max_y, prev_lines[1][1], min_y]
            prev_lines[1] = []


        # ================= STOP LINE =================
        if stop_line_x_second:
             # 수평선 (정지선) 추정 모델
            poly_stop = np.poly1d(np.polyfit(
                stop_line_x_first,
                stop_line_y_first,
                deg=1
            ))

            y1 = int(poly_stop(50))
            y2 = int(poly_stop(750))
            if prev_lines[2]:
                # recalculate y1
                if abs(y1 - prev_lines[2][0]) > offset:
                    y1 = prev_lines[2][0] - offset if prev_lines[2][0] > y1 else prev_lines[2][0] + offset
                # recalculate y2
                if abs(y2 - prev_lines[2][1]) > offset:
                    y2 = prev_lines[2][1] - offset if prev_lines[2][1] > y2 else prev_lines[2][1] + offset

            prev_lines[2] = [y1, y2]
            stop_line.append([50, y1, 750, y2]) # 정지선 좌표 저장
        elif prev_lines[2]:
            stop_line.append([50, prev_lines[2][0], 750, prev_lines[2][1]])
            prev_lines[2] = []

    return lane, stop_line


def hough_lines(img, rho=6, theta=np.pi / 120, threshold=160, min_line_len=60, max_line_gap=10):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    허프 선 변환 함수 (cv2.HoughLinesP)
    - 입력 이미지는 Canny 엣지 결과여야 함
    - 직선 후보들을 검출해서 반환 (선분 집합)

    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return lines


#여기까지 slope 값 기반 좌/우/정지선 분류, polyfit + poly1d 로 선형 회귀, prev_lines 활용해 결과 안정화 (스무딩), 허프라인 검출 (hough_lines)	

#######################여기까지 #2



# Python 3 has support for cool math symbols.
# 이미지 두 장을 합성해서 반환 (weighted sum)
def add_images(img, initial_img):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    #  img: 차선만 그려진 블랙 이미지 (출력용)
    # initial_img: 원본 이미지
    # 두 이미지를 가중합으로 합쳐서 최종 출력 이미지 생성
    return cv2.add(initial_img, img)


# 차선/정지선을 실제 이미지 위에 그려주는 함수
def draw_lane(original_img, lane, stop_line, left_color, right_color, thickness=5):
    # 빈 이미지 생성 (차선 그리기 용도)
    img = np.zeros((original_img.shape[0], original_img.shape[1], 3), dtype=np.uint8)
    polygon_points = None
    offset_from_lane_edge = 8 # 시각화용 살짝 오프셋

    # draw lane lines
    # 좌측 차선 그리기
    if lane[0]:
        for x1, y1, x2, y2 in [lane[0]]:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), left_color, thickness)

      # 우측 차선 그리기        
    if lane[1]:
        for x1, y1, x2, y2 in [lane[1]]:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), right_color, thickness)

    # color the lane
      # 차선 내부를 색칠
    if lane[0] and lane[1]:
        lane_color = [40, 60, 0]  # 어두운 녹색 음영
        for x1, y1, x2, y2 in [lane[0]]:
            p1 = (x1 + offset_from_lane_edge, y1)
            p2 = (x2 + offset_from_lane_edge, y2)

        for x1, y1, x2, y2 in [lane[1]]:
            p3 = (x2 - offset_from_lane_edge, y2)
            p4 = (x1 - offset_from_lane_edge, y1)

        polygon_points = np.array([[p1, p2, p3, p4]], np.int32)
        cv2.fillPoly(img, polygon_points, lane_color)

    # draw stop line
      # 정지선 표시
    if stop_line:
        for x1, y1, x2, y2 in stop_line:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), [0, 0, 255], thickness * 3)
            
            # 정지선이 차선 안에 위치한다면 다시 마스킹 처리
            if polygon_points is not None:
                for px1, py1, px2, py2 in [lane[0]]:
                    p1 = (px1 - offset_from_lane_edge, py1)
                    p2 = (px2 - offset_from_lane_edge, py2)

                for px1, py1, px2, py2 in [lane[1]]:
                    p3 = (px2 + offset_from_lane_edge, py2)
                    p4 = (px1 + offset_from_lane_edge, py1)

                polygon_points = np.array([[p1, p2, p3, p4]], np.int32)

                img = region_of_interest(img, polygon_points)

    return add_images(img, original_img)


# 차선 검출 전체 파이프라인을 하나로 묶은 함수
def detect_lane(screen):
    # 0. Crop the image
    image = crop(screen)
    # 1. convert to gray
    image = grayscale(image)
    # 2. apply gaussian filter
    image = gaussian_blur(image, 7)
    # 3. canny
    image = canny(image, 50, 100)
    # 4. ROI
    image = region_of_interest(image, np.array([[(0, 190), (0, 70), (187, 0),
                                                 (613, 0), (800, 70), (800, 190)]], np.int32))
    # 5. Hough lines
    lines = hough_lines(image)
    # 6. construct lane
    return construct_lane(lines)


# 디버깅용 main() 함수 (실제 실행 시 사용되지 않음)
def main():
    while True:
        original_img = grab_screen()
        # 1. convert to gray
        image = grayscale(crop(original_img))
        # 2. apply gaussian filter
        image = gaussian_blur(image, 7)
        # 3. canny
        image = canny(image, 50, 100)
        # 4. ROI
        image = region_of_interest(image, np.array([[(0, 190), (0, 70), (187, 0),
                                                     (613, 0), (800, 70), (800, 190)]], np.int32))
        # 5. Hough lines
        lines = hough_lines(image)
        # 6. construct lane
        lane, stop_line = construct_lane(lines)
        # 7. Place lane detection output on the original image
        original_img[280:-130, :, :] = draw_lane(original_img[280:-130, :, :], lane, stop_line, [0, 255, 0],
                                                 [0, 255, 0])

        cv2.imshow("Frame", original_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()

#draw_lane()	차선, 정지선, 차선 내부 색상 시각화
#detect_lane()	전체 파이프라인 묶음 (전처리 → 허프라인 → 차선 추출)
#main()	테스트용, 실제 프로젝트에선 미사용
#draw_lane()는 실제 시각적 피드백 제공 → drive.py에서 사용됨
#############여기까지 #3