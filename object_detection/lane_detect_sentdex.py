
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #이건 현재 파일 기준으로 상위 폴더(=루트) 를 자동으로 찾아주는 방식이야
import cv2
import numpy as np
from numpy import ones, vstack, mean
from numpy.linalg import lstsq
from PIL import ImageGrab
from data_collection.preprocess_sentdex import preprocess_img,grab_screen


def hough_lines(roi_img):
    #lines = cv2.HoughLinesP(roi_img, 1, np.pi / 180, 180, np.array([]), minLineLength=40, maxLineGap=25)
    lines = cv2.HoughLinesP(roi_img, 1, np.pi/180, threshold=50, minLineLength=40, maxLineGap=30)

    return lines




#def draw_lanes(roi_img, lines, color=[0, 255, 255], thickness=3):
def draw_lanes(original_img, roi_img, color=[0, 255, 255], thickness=3):
    lines = hough_lines(roi_img)  # 이 줄을 반드시 추가해야 작동함
    if lines is None:
        print("[WARN] No lane lines found.")
        #return screen, None
        return original_img, None

    # if this fails, go with some default line
    try:

        # finds the maximum y value for a lane marker 
        # (since we cannot assume the horizon will always be at the same point.)

        ys = []  
        for i in lines:
            for ii in i:
                ys += [ii[1],ii[3]]
        min_y = min(ys)
        max_y = 600
        new_lines = []
        line_dict = {}

        for idx,i in enumerate(lines):
            for xyxy in i:
                # These four lines:
                # modified from http://stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points
                # Used to calculate the definition of a line, given two sets of coords.
                x_coords = (xyxy[0],xyxy[2])
                y_coords = (xyxy[1],xyxy[3])
                A = vstack([x_coords,ones(len(x_coords))]).T
                m, b = lstsq(A, y_coords)[0]

                # Calculating our new, and improved, xs
                x1 = (min_y-b) / m
                x2 = (max_y-b) / m

                line_dict[idx] = [m,b,[int(x1), min_y, int(x2), max_y]]
                new_lines.append([int(x1), min_y, int(x2), max_y])

        final_lanes = {}

        for idx in line_dict:
            final_lanes_copy = final_lanes.copy()
            m = line_dict[idx][0]
            b = line_dict[idx][1]
            line = line_dict[idx][2]
            
            if len(final_lanes) == 0:
                final_lanes[m] = [ [m,b,line] ]
                
            else:
                found_copy = False

                for other_ms in final_lanes_copy:

                    if not found_copy:
                        if abs(other_ms*1.2) > abs(m) > abs(other_ms*0.8):
                            if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8):
                                final_lanes[other_ms].append([m,b,line])
                                found_copy = True
                                break
                        else:
                            final_lanes[m] = [ [m,b,line] ]

        line_counter = {}

        for lanes in final_lanes:
            line_counter[lanes] = len(final_lanes[lanes])

        top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2] 
        #여기서 top_lanes 리스트 자체가 비어있는 경우  (즉, 검출된 선분이 하나도 없을 때), top_lanes[0][0]는 list index out of range 예외를 발생시킴

        #위이 문제 방어코드
        if len(top_lanes) < 2:
            print("[WARN] Not enough lane lines for draw_lanes.")
            return None, None 


        lane1_id = top_lanes[0][0]
        lane2_id = top_lanes[1][0]

        def average_lane(lane_data):
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            for data in lane_data:
                x1s.append(data[2][0])
                y1s.append(data[2][1])
                x2s.append(data[2][2])
                y2s.append(data[2][3])
            return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s)) 

        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

        print("[DEBUG] number of valid lane groups:", len(top_lanes)) #차선이 안그려질떄 로그찍기
        #return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2] #리턴값으로 좌/우 차선 좌표(만) 제공, 그래서 lane_detect윈도우에 차선안그려진다는
        lane_img = original_img.copy()
        cv2.line(lane_img, (l1_x1, l1_y1), (l1_x2, l1_y2), color, thickness)
        cv2.line(lane_img, (l2_x1, l2_y1), (l2_x2, l2_y2), color, thickness)
        return lane_img, [l1_x1, l1_x2, l2_x1, l2_x2]

        
    #except Exception as e:
     #   print("[ERROR in draw_lanes]:", e)
    

     #return None, None #none으로 리턴시 data_collect파일에서  TypeError: src is not a numpy array, neither a scalar 이렇게 오류나옴. lane_img가 none으로 받아지기에 발생
    #return screen.copy(), None #screen 변수가 draw_lanes() 함수에 정의되어 있지 않기 때문에, 현재 함수 시그니처 (original_img, roi_img, ...)에 맞춰수정한다
    except Exception as e:
        print("[ERROR in draw_lanes]:", e)
        return original_img.copy(), None
    
   


    #except Exception as e:
       # print(str(e))
    #return original_img.copy(), None





def main():
    last_time = time.time()
    while True:
        bbox = np.array([[(0, 560), (0, 350), (320, 200), (960, 200), (1280, 350), (1280, 560)]], dtype=np.int32)
        screen =  np.array(ImageGrab.grab(bbox))
        print('Frame took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        new_screen,original_image = preprocess_img(screen)
        cv2.imshow('window', new_screen)
        cv2.imshow('window2',cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break