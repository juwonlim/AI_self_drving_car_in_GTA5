"""
Car driving module.
학습된 모델(model-xxx.h5)을 불러와서
YOLO + 차선 + 속도 + 방향 정보 기반으로 자율주행을 수행
"""

# reading and writing files
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #이건 현재 파일 기준으로 상위 폴더(=루트) 를 자동으로 찾아주는 방식
import cv2
import numpy as np
# load our saved model
from keras.models import load_model # 학습된 keras 모델 로딩용

# helper classes
# 입력 이미지 및 센서 정보 처리
from data_collection.img_process import img_process
from data_collection.key_cap import key_check

# gamepad axes limits and gamepad module
# (게임패드 관련 모듈은 주석 처리됨)
#from driving.gamepad import AXIS_MIN, AXIS_MAX, TRIGGER_MAX, XInputDevice
#from object_detection.direction import Direct



# YOLO algorithm
# YOLO + 차선 인식 + 전처리
from object_detection.object_detect import yolo_detection
# lane detection algorithm
from object_detection.lane_detect import detect_lane, draw_lane
from training.utils import preprocess



import pyautogui
import time

#키보드로 게임을 조종하는 함수
def apply_keyboard_controls(steering, throttle):
    """
    AI가 예측한 스티어링과 스로틀 값을 바탕으로 키보드 입력 전달
    - throttle: -1 ~ 1 (음수면 후진)
    - steering: -1 ~ 1 (음수면 좌회전, 양수면 우회전)
    """

    # =====================
    # 1. 전진 (W) / 후진 (S)
    # =====================
    if throttle > 0.2:
        pyautogui.keyDown('w')
        pyautogui.keyUp('s')
    elif throttle < -0.2:
        pyautogui.keyDown('s')
        pyautogui.keyUp('w')
    else:
        pyautogui.keyUp('w')
        pyautogui.keyUp('s')

    # =====================
    # 2. 좌/우 회전 (A/D)
    # =====================
    if steering < -0.2:
        pyautogui.keyDown('a')
        pyautogui.keyUp('d')
    elif steering > 0.2:
        pyautogui.keyDown('d')
        pyautogui.keyUp('a')
    else:
        pyautogui.keyUp('a')
        pyautogui.keyUp('d')
    print("Applying control - Steering:", steering, "Throttle:", throttle) #이렇게 하면 매 프레임마다 AI가 결정한 조향(steering) 과 가속(throttle) 값을 콘솔에 바로 출력,"AI가 어떤 명령을 내리고 있는지" 실시간으로 볼 수 있게 돼.






# 모델 경로 설정
model_path = "..\\training"
gamepad = None  # 원래는 XInputDevice를 여기 연결함



# 실제 게임패드에 throttle/steering 값을 반영
# 현재는 사용되지 않음 (gamepad 기능 비활성화 상태)
def set_gamepad(controls):
    # trigger value
    trigger = int(round(controls[0][1] * TRIGGER_MAX))
    if trigger >= 0:
        # set left trigger to zero
        gamepad.SetTrigger('L', 0)
        gamepad.SetTrigger('R', trigger)
    else:
        # inverse value
        trigger = -trigger
        # set right trigger to zero
        gamepad.SetTrigger('L', trigger)
        gamepad.SetTrigger('R', 0)

    # axis value
    axis = 0
    if controls[0][0] >= 0:
        axis = int(round(controls[0][0] * AXIS_MAX))
    else:
        axis = int(round(controls[0][0] * (-AXIS_MIN)))
    gamepad.SetAxis('X', axis)


# 자율주행 루프
def drive(model):
    global gamepad
    
    #gamepad = XInputDevice(1)
    #gamepad.PlugIn()

    # last_time = time.time()  # to measure the number of frames
    close = False  # to exit execution ,# 종료 여부
    pause = True  # to pause execution ,# 일시정지 상태
    stop = False    # to stop the car , # 목적지 도달 후 정지 여부
    throttle = 0    # 속도 제어값
    left_line_max = 75
    right_line_max = 670

    print("Press T to start driving")

    while not close:
        # 화면, 전처리 이미지, 속도, 방향 추출
        yolo_screen, resized, speed, direct = img_process("Grand Theft Auto V")
        cv2.imshow("Driving-mode", yolo_screen)
        cv2.waitKey(1)

        while not pause:
            # apply the preprocessing
            screen, resized, speed, direct = img_process("Grand Theft Auto V")

              # 작은 레이더 센서 이미지 추출 (전면 화면 일부 영역)
            radar = cv2.cvtColor(resized[206:226, 25:45, :], cv2.COLOR_RGB2BGR)[:, :, 2:3]
            resized = preprocess(resized)
            
              # 조향 선 색상 초기화
            left_line_color = [0, 255, 0]
            right_line_color = [0, 255, 0]

            # predict steering angle for the image
            # original + radar (small) + speed
             # 모델 추론 (조향 angle 예측)
            controls = model.predict([np.array([resized]), np.array([radar]), np.array([speed])], batch_size=1)
            
            # check that the car is following lane
             # 차선 및 정지선 인식
            lane, stop_line = detect_lane(screen)

            # detect objects
             # 객체 인식 (YOLO)
            yolo_screen, color_detected, obj_distance = yolo_detection(screen, direct)

             # 속도 제어
            if not stop:
                # adjusting speed
                if speed < 45:
                    throttle = 0.4
                elif speed > 50:
                    throttle = 0.0

                if 0 <= obj_distance <= 0.6:
                    if speed < 5:
                        throttle = 0
                    else:
                        throttle = -0.7 if obj_distance <= 0.4 else -0.3

                elif color_detected == "Red":
                    if stop_line:
                        if speed < 5:
                            throttle = 0
                        elif 0 <= stop_line[0][1] <= 50:
                            throttle = -0.5
                        elif 50 < stop_line[0][1] <= 120:
                            throttle = -1
                    # else:
                    #     throttle = -0.5
            elif speed > 5:
                throttle = -1
            else:
                throttle = 0
                cv2.destroyAllWindows()
                pause = True

            # adjusting steering angle
            # 조향 보정
            if lane[0] and lane[0][0] > left_line_max:
                if abs(controls[0][0]) < 0.27:
                    controls[0][0] = 0.27
                    left_line_color = [0, 0, 255]
            elif lane[1] and lane[1][0] < right_line_max:
                if abs(controls[0][0]) < 0.27:
                    controls[0][0] = -0.27
                    right_line_color = [0, 0, 255]

            # set the gamepad values
            # 여기서 실제 gamepad에 값 적용 (현재는 비활성화된 상태)
            #set_gamepad([[controls[0][0], throttle]])
            apply_keyboard_controls(controls[0][0], throttle) #키보드로 조작하는 코드 삽입

            # print('Main loop took {} seconds'.format(time.time() - last_time))
            # last_time = time.time()

             # 차선 시각화
            screen[280:-130, :, :] = draw_lane(screen[280:-130, :, :], lane, stop_line,
                                               left_line_color, right_line_color)
            cv2.imshow("Driving-mode", yolo_screen)
            cv2.waitKey(1)


             # 목적지 도착 처리
            if direct == 6:
                print("Arrived at destination.")
                stop = True

            # print('Main loop took {} seconds'.format(time.time() - last_time))
            # last_time = time.time()

            keys = key_check()
            #print("Pressed keys:", keys) #key_check()함수가 정말 키를 감지하고 있는지 로그 찍어보기
            if 'T' in keys:
                cv2.destroyAllWindows()
                pause = True  #아래 코드에 따라, 시작 시 루프는 pause = True이므로 drive() 루프에서 실제 자율주행이 대기 상태
                # release gamepad keys
                #set_gamepad([[0, 0]]) #완전 키보드 기반으로 간다면 이것도 주석처리
                print('Paused. To exit the program press Z.')
                time.sleep(0.5)

        keys = key_check()
        print("Pressed keys:", keys) #key_check()함수가 정말 키를 감지하고 있는지 로그 찍어보기,T나 Z키를 눌렀을때만 확인하려는 목적이면 여기에 PRINT추가
        if 'T' in keys:
            pause = False
            stop = False
            print('Unpaused')
            time.sleep(1)
        elif 'Z' in keys:
            cv2.destroyAllWindows()
            close = True
            print('Closing the program.')
            #gamepad.UnPlug() #비활성화 처리

            #종료시 안정 정리 코드(키가 계속 눌리는 것 방지) : 'Z' 키 눌려 종료되는 블록 안--AI가 눌렀던 키를 안전하게 해제하려면 종료 직전에 실행돼야 함
            pyautogui.keyUp('w')
            pyautogui.keyUp('a')
            pyautogui.keyUp('d')
            pyautogui.keyUp('s')


# 메인 실행 함수
def main():
    # load model
    #location = os.path.join(model_path, 'base_model.h5')
    #location = os.path.join(model_path, 'model-010.h5')
    location = r"D:\gta5_project\AI_GTA5\training\model-039.h5"
    model = load_model(location) # 학습된 모델 불러오기
    # control a car
    drive(model)  # 자율주행 시작


if __name__ == '__main__':
    main()
