
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #이건 현재 파일 기준으로 상위 폴더(=루트) 를 자동으로 찾아주는 방식이야
import math
import cv2
import numpy as np

# GTA5 게임 화면을 캡처하는 함수 (navigation_img_process.py에서 가져옴)
from data_collection.preprocess import get_preprocessed
from data_collection.preprocess import region_of_interest
 #hough_lines함수는 lane_detect.py에서만 쓰이므로 preprocess.py에 포함시키지 않음





# 이전 프레임에서 검출한 차선 정보를 저장하는 전역 변수
# [left_line, right_line, stop_line]
prev_lines = [[], [], []]





def hough_lines(roi):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    허프 선 변환 함수 (cv2.HoughLinesP)
    - 입력 이미지는 Canny 엣지 결과여야 함
    - 직선 후보들을 검출해서 반환 (선분 집합)

    """
    rho = 6
    theta = np.pi / 120
    threshold = 160
    min_line_len = 60
    max_line_gap = 10


    lines = cv2.HoughLinesP(roi, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return lines #construct_lane함수로 넘겨줌





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


    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:

                 # 기울기 계산 (x1 == x2일 경우 분모 0 방지)
                slope = (y2 - y1) / (x2 - x1) if x1 != x2 else 0  # <-- Calculating the slope.
                
                 # 기울기가 너무 완만한 선은 무시 (정지선도 아님)
                #if 0.05 < math.fabs(slope) < 0.3:  # not interested
                    #continue

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
    
    #TypeError: 'int' object is not subscriptable , 이 줄에서 문제가 발생한 이유는 left_lane 또는 right_lane 중 하나가 **리스트가 아니라 int**인 경우
    print("Left lane (full):", lane[0]) #방어코드
    print("Right lane (full):", lane[1])

    return lane, stop_line #이걸 detect_lane함수에서 construct_lane 함수를 호출해서 내부에서 받음. 




"""
def detect_lane():
   
   
    processed = get_preprocessed() #preprocess.py파일의 get_preprocessed함수를 processed 변수로 할당

    if processed is None or 'roi' not in processed or processed['roi'] is None or processed['roi'].size == 0: #방어코드
        print("[ERROR] preprocess failed or returned incomplete data.")
        return [[], []], []

    if processed is None or 'roi' not in processed:
        print("[ERROR] preprocess failed or returned incomplete data.")
        return [[], []], []

    roi = processed['roi'] #processed변수에 roi를 호출하여 roi에 저장

    print("ROI pixel sum:", np.sum(roi))  


    # 시각화: MASK
    if 'mask' in processed:
        cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("mask", 480, 270)
        cv2.imshow("mask", processed['mask'])
        print("Image shape mask:", processed['mask'].shape)
        cv2.waitKey(1)

    # 시각화: CANNY
    if 'canny_edge_lines' in processed:
        cv2.namedWindow("canny_edge_lines", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("canny_edge_lines", 480, 270)
        cv2.imshow("canny_edge_lines", processed['canny_edge_lines'])
        print("canny_edge:", processed['canny_edge_lines'].shape)
        cv2.waitKey(1)

    # 시각화: ROI
    if roi is not None:
        cv2.namedWindow("roi", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("roi", 480, 270)
        cv2.imshow("roi", roi)
        print("Image shape after roi:", roi.shape)
        cv2.waitKey(1)

    if roi is None or roi.size == 0:
        print("[ERROR] ROI image is empty.")
        return [[], []], []

    # 차선 검출
    lines = hough_lines(roi) #함수내부에서 hough line함수호출
    lanes, stop_line = construct_lane(lines) #함수내부에서 construct_lane함수호출

    #return (lanes, stop_line), image, processed["screen"] #이건 뭐꼬?
    #return (lanes, stop_line), image #세번째 인자는 data_collect.py에서 안씀. 이걸 MAIN으로 넘기나?
    return lanes, stop_line #내 생각에 세번쨰 인자 image는 안쓰이는듯해서 삭제


 """



def detect_lane():
    roi = get_preprocessed()  # preprocess.py의 get_preprocessed는 이제 roi만 반환함

    if roi is None or roi.size == 0:
        print("[ERROR] ROI image is empty.")
        return [[], []], None

    print("ROI pixel sum:", np.sum(roi))

    # 시각화: ROI만 유지 (mask, canny는 get_preprocessed에서 제거되었으므로 생략 가능)
    cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI", 480, 270)
    cv2.imshow("ROI", roi)
    print("Image shape after ROI:", roi.shape)
    cv2.waitKey(1)

    # 차선 검출
    lines = hough_lines(roi)
    lanes, stop_line = construct_lane(lines)

    return lanes, stop_line







# 차선/정지선을 실제 이미지 위에 그려주는 함수
def draw_lane(*args,**kwargs):
       
    cropped_img = kwargs.get("cropped_img")
    if cropped_img is None:
        print("[ERROR] draw_lane(): cropped_img is None")
        return None
    lane = kwargs.get("lane", [[], []])
    stop_line = kwargs.get("stop_line")
    left_color = kwargs.get("left_color", [0, 255, 0])
    right_color = kwargs.get("right_color", [0, 255, 0])
    thickness = kwargs.get("thickness", 5)

    print("draw_lane() called")
    print("Left lane points:", lane[0])
    print("Right lane points:", lane[1])
    
    
    
    # 빈 이미지 생성 (차선 그리기 용도)
    img = np.zeros((cropped_img.shape[0], cropped_img.shape[1], 3), dtype=np.uint8)
    
    if img is None or img.shape[0] == 0 or img.shape[1] == 0:
        print("[ERROR] Drawn image is invalid.")
        return cropped_img

    
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
    
    print("img:", img.shape if img is not None else None)
    print("cropped_img:", cropped_img.shape if cropped_img is not None else None)
    blended = add_images(img, cropped_img) #여기서 add_image함수 호출하여 작업후 blended에 담음

    return blended


# Python 3 has support for cool math symbols.
# 이미지 두 장을 합성해서 반환 (weighted sum)
def add_images(*args, **kwargs):
    img = kwargs.get("img")
    initial_img = kwargs.get("cropped_img")
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

    # 예외 방지: None 체크 -->Nonetype에러방지 (AttributeError: 'NoneType' object has no attribute 'shape')
    if img is None:
        print("[ERROR] add_images: img is None")
        return initial_img
    if initial_img is None:
        print("[ERROR] add_images: initial_img is None")
        return img


     # 이미지 크기 맞추기
     #cv2.error: OpenCV(3.4.2) ... error: (-209:Sizes of input arguments do not match) The operation is neither 'array op array' (where arrays have the same size and channels)에러해결목적
    if initial_img.shape != img.shape:
        img = cv2.resize(img, (initial_img.shape[1], initial_img.shape[0]))

    # 채널 수 맞추기 (ex: gray -> BGR)
    if len(initial_img.shape) == 3 and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
    print("initial_img shape:", initial_img.shape)
    print("img shape:", img.shape)
    img = cv2.add(initial_img, img)
    return img #이게 다시 draw_lane이 받아감



"""

def main():
    crop_top = 200 #차선이 있는 Y축 좌표 위쪽
    crop_bottom = 550 #마찬가지로 차선이 들어오는 y축 좌표 아래
    
    
    while True:
        lane_result, roi_image, screen = detect_lane()
        if roi_image is None:
            continue

        cropped = screen[crop_top:crop_bottom, :, :]

        blended = draw_lane(
            original_img=cropped,
            lane=lane_result["lanes"],
            stop_line=lane_result["stop_line"],
            left_color=[0, 255, 0],
            right_color=[0, 255, 0],
            thickness = 5 #누락시켰던 값 추가
        )
        screen[crop_top:crop_bottom, :, :] = blended

        cv2.imshow("LANE_VIEW", screen)
        cv2.imshow("ROI", roi_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


""" 



def main():
    crop_top = 200 #차선이 있는 Y축 좌표 위쪽
    crop_bottom = 550 #마찬가지로 차선이 들어오는 y축 좌표 아래
    
    
    while True:
        #lane_result, roi_image, screen = detect_lane()
        lane_result, roi_image = detect_lane() #detect_lane함수가 2개만 리턴하므로 main에서도 2개 받아야함
        if roi_image is None:
            continue

        cropped = screen[crop_top:crop_bottom, :, :]

        blended = draw_lane(
            cropped_img=cropped,
            lane=lane_result["lanes"],
            stop_line=lane_result["stop_line"],
            left_color=[0, 255, 0],
            right_color=[0, 255, 0],
            thickness = 5 #누락시켰던 값 추가
        )
        

        cv2.imshow("LANE_VIEW", blended)
        cv2.imshow("ROI", roi_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()

"""
draw_lane() 함수 안에서 내부적으로 add_images()를 한 번 호출

add_images()는 blended 이미지 1장을 만들어서 return

그 결과값을 draw_lane()이 그대로 return함

main()에서 그 리턴값을 받아 screen[200:250]에 덮어씀
"""