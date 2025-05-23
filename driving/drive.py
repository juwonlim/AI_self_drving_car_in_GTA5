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
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #gpu사용을 멈추고 cpu로 강제 작동시켜서 cpu로 자율주행시키는 것,gpu가 6기가 밖에 안되는 경우라서
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



#import pyautogui
import pydirectinput as pyautogui #다이렉트 인풋
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

    # gamepad = XInputDevice(1)
    # gamepad.PlugIn()

    close = False  # 종료 여부
    pause = True  # 일시정지 여부
    stop = False  # 목적지 도착 여부
    throttle = 0.3   # 가속도 제어
    left_line_max = 75
    right_line_max = 670

    print("Press T to start driving")

    while not close:
        # ========== 여기부터 업데이트 ==========
        yolo_screen, resized, speed, direct = img_process("Grand Theft Auto V")
        cv2.imshow("Driving-mode", yolo_screen)
        cv2.waitKey(1)
        # ========== 여기까지 업데이트 ==========

        while not pause:
            screen, resized, speed, direct = img_process("Grand Theft Auto V")

            radar = cv2.cvtColor(resized[206:226, 25:45, :], cv2.COLOR_RGB2BGR)[:, :, 2:3]
            resized = preprocess(resized)

            left_line_color = [0, 255, 0]
            right_line_color = [0, 255, 0]

            controls = model.predict([np.array([resized]), np.array([radar]), np.array([speed])], batch_size=1)

            lane, stop_line = detect_lane(screen)
            yolo_screen, color_detected, obj_distance = yolo_detection(screen, direct)

            if not stop:
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
            elif speed > 5:
                throttle = -1
            else:
                throttle = 0
                cv2.destroyAllWindows()
                pause = True

            if lane[0] and lane[0][0] > left_line_max:
                if abs(controls[0][0]) < 0.27:
                    controls[0][0] = 0.27
                    left_line_color = [0, 0, 255]
            elif lane[1] and lane[1][0] < right_line_max:
                if abs(controls[0][0]) < 0.27:
                    controls[0][0] = -0.27
                    right_line_color = [0, 0, 255]

            #set_gamepad([[controls[0][0], throttle]])  # 비활성화
            apply_keyboard_controls(controls[0][0], throttle)  # 새코드: 키보드로 조작

            screen[280:-130, :, :] = draw_lane(screen[280:-130, :, :], lane, stop_line, left_line_color, right_line_color)
            cv2.imshow("Driving-mode", yolo_screen) #객체감지한 yolo를 병행해서 띄워주는 새창이네!
            cv2.waitKey(1)

            #if direct == 6: #미션없이 주행할때, 작동을 멈추게 하므로 주석처리
                #print("Arrived at destination.")
                #stop = True

            keys = key_check()
            # print("Pressed keys:", keys)  # 생략 가능 (디버깅용)
            if 'T' in keys:
                cv2.destroyAllWindows()
                pause = True
                # set_gamepad([[0, 0]])  # 비활성화
                print('Paused. To exit the program press Z.')
                time.sleep(0.5)

        keys = key_check()
        # print("Pressed keys:", keys)  # 생략 가능
        if 'T' in keys:
            pause = False
            stop = False
            print('Unpaused')
            time.sleep(1)
        elif 'Z' in keys:
            cv2.destroyAllWindows()
            close = True
            print('Closing the program.')
            # gamepad.UnPlug()  # 비활성화

            # 새코드: 키보드 안전하게 해제
            pyautogui.keyUp('w')
            pyautogui.keyUp('a')
            pyautogui.keyUp('d')
            pyautogui.keyUp('s')


# 메인 실행 함수
def main():
    # load model
    location = r"D:\gta5_project\AI_GTA5\training\model-016.h5"
    model = load_model(location) # 학습된 모델 불러오기
    # control a car
    drive(model)  # 자율주행 시작


if __name__ == '__main__':
    main()
