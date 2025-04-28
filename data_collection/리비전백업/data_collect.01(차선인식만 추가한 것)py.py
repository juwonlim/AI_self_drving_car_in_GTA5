import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import threading
import time
import winsound

import h5py

from data_collection.gamepad_cap import Gamepad
from data_collection.img_process import img_process
from data_collection.key_cap import key_check

### [추가]
from object_detection.lane_detect import detect_lane  # 차선 인식 추가

lock = threading.Lock()

path = "training/training_data_by_user_drive.h5"
os.makedirs(os.path.dirname(path), exist_ok=True)

data_file = None
if os.path.isfile(path):
    data_file = h5py.File(path, 'a')
else:
    data_file = h5py.File(path, 'w')
    data_file.create_dataset('img', (0, 240, 320, 3), dtype='u1',
                             maxshape=(None, 240, 320, 3), chunks=(30, 240, 320, 3))
    data_file.create_dataset('controls', (0, 2), dtype='i1', maxshape=(None, 2), chunks=(30, 2))
    data_file.create_dataset('metrics', (0, 2), dtype='u1', maxshape=(None, 2), chunks=(30, 2))
    
    ### [추가]
    data_file.create_dataset('lanes', (0, 4), dtype='i2', maxshape=(None, 4), chunks=(30, 4))
    # lanes: 왼쪽 x1,x2, 오른쪽 x1,x2 (총 4개 좌표)

### [수정]
# save 함수 수정
def save(data_img, controls, metrics, lanes):
    with lock:
        if data_img:
            data_file["img"].resize((data_file["img"].shape[0] + len(data_img)), axis=0)
            data_file["img"][-len(data_img):] = data_img
            data_file["controls"].resize((data_file["controls"].shape[0] + len(controls)), axis=0)
            data_file["controls"][-len(controls):] = controls
            data_file["metrics"].resize((data_file["metrics"].shape[0] + len(metrics)), axis=0)
            data_file["metrics"][-len(metrics):] = metrics

            ### [추가]
            data_file["lanes"].resize((data_file["lanes"].shape[0] + len(lanes)), axis=0)
            data_file["lanes"][-len(lanes):] = lanes

def delete(session):
    frames = session if session < 500 else 500
    data_file["img"].resize((data_file["img"].shape[0] - frames), axis=0)
    data_file["controls"].resize((data_file["controls"].shape[0] - frames), axis=0)
    data_file["metrics"].resize((data_file["metrics"].shape[0] - frames), axis=0)

    ### [추가]
    data_file["lanes"].resize((data_file["lanes"].shape[0] - frames), axis=0)

def main():
    gamepad = Gamepad()
    gamepad.open()

    alert_time = time.time()
    close = False
    pause = True
    session = 0
    training_img = []
    controls = []
    metrics = []

    ### [추가]
    lanes = []

    print("Press RB on your gamepad or keyboard 'K' to start recording")
    print("🔥 프로그램 시작됨. K를 눌러 녹화를 시작하세요.")

    while not close:
        while not pause:
            throttle, steering = gamepad.get_state()
            ignore, screen, speed, direction = img_process("Grand Theft Auto V")

            ### [추가]
            lane, stop_line = detect_lane(screen)

            # 차선이 검출됐을 때만 저장, 없으면 0으로 초기화
            left_lane = lane[0] if lane[0] else [0, 0, 0, 0]
            right_lane = lane[1] if lane[1] else [0, 0, 0, 0]

            # 왼쪽 차선 x1,x2, 오른쪽 차선 x1,x2만 추출
            lanes.append([left_lane[0], left_lane[2], right_lane[0], right_lane[2]])

            training_img.append(screen)
            controls.append([throttle, steering])
            metrics.append([speed, direction])
            session += 1

            if speed > 60 and time.time() - alert_time > 1:
                winsound.PlaySound('.\\resources\\alert.wav', winsound.SND_ASYNC)
                alert_time = time.time()

            if len(training_img) % 30 == 0:
                threading.Thread(target=save, args=(training_img, controls, metrics, lanes)).start()
                training_img = []
                controls = []
                metrics = []
                ### [추가]
                lanes = []

            time.sleep(0.015)

            if gamepad.get_RB() or 'P' in key_check():
                pause = True
                print('Paused. Save the last 15 seconds?')

                keys = key_check()
                while ('Y' not in keys) and ('N' not in keys):
                    keys = key_check()

                if 'N' in keys:
                    delete(session)
                    training_img = []
                    controls = []
                    metrics = []
                    lanes = []  ### [추가]
                    print('Deleted.')
                else:
                    print('Saved.')

                print('To exit the program press LB or keyboard L.')
                session = 0
                time.sleep(0.5)

        keys = key_check()
        if gamepad.get_RB() or 'K' in keys:
            pause = False
            print('Unpaused by keyboard or gamepad')
            time.sleep(1)

        elif gamepad.get_LB() or 'L' in keys:
            gamepad.close()
            close = True
            print('Saving data and closing the program.')
            save(training_img, controls, metrics, lanes)  # 종료할 때도 lanes 저장
            data_file.close()

if __name__ == '__main__':
    print("✅ data_collect.py 실행됨.")
    main()
