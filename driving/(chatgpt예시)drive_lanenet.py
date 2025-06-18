import numpy as np
import cv2
import time
import pyautogui
from tensorflow.keras.models import load_model
from data_collection.navigation_img_process import img_process
from lanenet_predict import predict_lane
from object_detection.yolo_detect import yolo_detection

# LANENet 가중치 경로
weights_path = "weights/tusimple_lanenet.pb"

# 자율주행 루프
def drive(model):
    close = False  # 종료 여부
    auto_mode = False  # 자율주행 활성화 여부
    throttle = 0.3   # 초기 가속도

    print("[INFO] Press 'D' to activate self-driving mode")
    print("[INFO] Press 'F' to deactivate self-driving mode")
    print("[INFO] Press 'Z' to quit")

    while not close:
        # 기본 화면 업데이트
        screen, resized, speed, _ = img_process("Grand Theft Auto V")
        yolo_screen, _, _ = yolo_detection(screen, 0)
        cv2.imshow("Driving-mode", yolo_screen)
        cv2.waitKey(1)

        # 키 입력 확인
        keys = pyautogui.keyDown

        if pyautogui.keyDown('d'):
            auto_mode = True
            print("[INFO] Self-driving mode activated")
            time.sleep(0.5)
        elif pyautogui.keyDown('f'):
            auto_mode = False
            pyautogui.keyUp('w')
            pyautogui.keyUp('a')
            pyautogui.keyUp('d')
            pyautogui.keyUp('s')
            print("[INFO] Self-driving mode deactivated")
            time.sleep(0.5)
        elif pyautogui.keyDown('z'):
            close = True
            pyautogui.keyUp('w')
            pyautogui.keyUp('a')
            pyautogui.keyUp('d')
            pyautogui.keyUp('s')
            print("[INFO] Program terminated")
            break

        if auto_mode:
            screen, resized, speed, _ = img_process("Grand Theft Auto V")
            binary_mask, _ = predict_lane(screen, weights_path)

            # LANENet 기반 조향 계산
            lane_indices = np.where(binary_mask[200:250, :] == 255)
            if lane_indices[1].size > 0:
                lane_center_x = np.mean(lane_indices[1])
                frame_center_x = binary_mask.shape[1] / 2
                error = (lane_center_x - frame_center_x) / frame_center_x
                steering = error * 0.5
            else:
                steering = 0

            # 시각화
            lane_viz = cv2.addWeighted(screen, 0.8, cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR), 0.5, 0)
            cv2.imshow("LANENet Visualization", lane_viz)

            # keras 모델로 throttle 계산
            radar = cv2.cvtColor(resized[206:226, 25:45, :], cv2.COLOR_RGB2BGR)[:, :, 2:3]
            resized_input = cv2.resize(screen, (200, 66))
            controls = model.predict([np.array([resized_input]), np.array([radar]), np.array([speed])], batch_size=1)

            apply_keyboard_controls(steering, controls[0][1])

        cv2.waitKey(1)

# 키보드 제어 함수 예시
def apply_keyboard_controls(steering, throttle):
    pyautogui.keyUp('a')
    pyautogui.keyUp('d')
    if steering < -0.2:
        pyautogui.keyDown('a')
    elif steering > 0.2:
        pyautogui.keyDown('d')

    if throttle > 0.1:
        pyautogui.keyDown('w')
    elif throttle < -0.1:
        pyautogui.keyDown('s')
    else:
        pyautogui.keyUp('w')
        pyautogui.keyUp('s')

if __name__ == '__main__':
    model_path = r"D:\gta5_project\AI_GTA5\training\model-016.h5"
    model = load_model(model_path)
    drive(model)
