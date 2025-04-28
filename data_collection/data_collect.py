#차선, 앞차 인식 데이터 수집까지 완성된 버전
#차간거리 유지는 아직임

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 상위 폴더 접근 가능하게

import threading
import time
import winsound

import h5py  # HDF5 파일 입출력용

from data_collection.gamepad_cap import Gamepad  # 게임패드/키보드 입력 감지
from data_collection.img_process import img_process  # GTA5 화면 캡처 및 처리
from data_collection.key_cap import key_check  # 키보드 입력 감지

### [추가] 차선 인식, YOLO 객체 인식 모듈
from object_detection.lane_detect import detect_lane
from object_detection.object_detect import yolo_detection

### [추가] TensorFlow GPU 메모리 4GB 제한 설정
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # 4GB까지만 할당
            )
    except RuntimeError as e:
        print(e)

lock = threading.Lock()  # 저장 중 데이터 충돌 방지용

# 저장할 파일 경로
path = "training/training_data_by_user_drive.h5"
os.makedirs(os.path.dirname(path), exist_ok=True)

# HDF5 파일 열기
data_file = None
if os.path.isfile(path):
    data_file = h5py.File(path, 'a')  # 있으면 이어쓰기
else:
    data_file = h5py.File(path, 'w')  # 없으면 새로 생성
    # 기본 데이터셋 생성
    data_file.create_dataset('img', (0, 240, 320, 3), dtype='u1', maxshape=(None, 240, 320, 3), chunks=(30, 240, 320, 3))
    data_file.create_dataset('controls', (0, 2), dtype='i1', maxshape=(None, 2), chunks=(30, 2))
    data_file.create_dataset('metrics', (0, 2), dtype='u1', maxshape=(None, 2), chunks=(30, 2))
    ### [추가]
    data_file.create_dataset('lanes', (0, 4), dtype='i2', maxshape=(None, 4), chunks=(30, 4))  # 왼쪽/오른쪽 차선 좌표
    data_file.create_dataset('obj_distance', (0, 1), dtype='f2', maxshape=(None, 1), chunks=(30, 1))  # 앞차 거리

# 데이터를 저장하는 함수
def save(data_img, controls, metrics, lanes, obj_distances):
    with lock:
        if data_img:
            data_file["img"].resize((data_file["img"].shape[0] + len(data_img)), axis=0)
            data_file["img"][-len(data_img):] = data_img
            data_file["controls"].resize((data_file["controls"].shape[0] + len(controls)), axis=0)
            data_file["controls"][-len(controls):] = controls
            data_file["metrics"].resize((data_file["metrics"].shape[0] + len(metrics)), axis=0)
            data_file["metrics"][-len(metrics):] = metrics
            data_file["lanes"].resize((data_file["lanes"].shape[0] + len(lanes)), axis=0)
            data_file["lanes"][-len(lanes):] = lanes
            data_file["obj_distance"].resize((data_file["obj_distance"].shape[0] + len(obj_distances)), axis=0)
            data_file["obj_distance"][-len(obj_distances):] = obj_distances

# 최근 데이터 삭제 함수
def delete(session):
    frames = session if session < 500 else 500
    data_file["img"].resize((data_file["img"].shape[0] - frames), axis=0)
    data_file["controls"].resize((data_file["controls"].shape[0] - frames), axis=0)
    data_file["metrics"].resize((data_file["metrics"].shape[0] - frames), axis=0)
    data_file["lanes"].resize((data_file["lanes"].shape[0] - frames), axis=0)
    data_file["obj_distance"].resize((data_file["obj_distance"].shape[0] - frames), axis=0)

# 메인 루프
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
    lanes = []
    obj_distances = []

    print("Press RB on your gamepad or keyboard 'K' to start recording")
    print("🔥 프로그램 시작됨. K를 눌러 녹화를 시작하세요.")

    while not close:
        while not pause:
            throttle, steering = gamepad.get_state()  # 게임패드로부터 throttle, steering 읽기
            ignore, screen, speed, direction = img_process("Grand Theft Auto V")  # 화면 캡처 및 차량 속도

            ### [추가] 차선 검출
            lane, stop_line = detect_lane(screen)
            left_lane = lane[0] if lane[0] else [0, 0, 0, 0]
            right_lane = lane[1] if lane[1] else [0, 0, 0, 0]
            lanes.append([left_lane[0], left_lane[2], right_lane[0], right_lane[2]])

            ### [추가] YOLO로 앞차 감지
            _, _, obj_distance = yolo_detection(screen, direct=0)
            if obj_distance is None:
                obj_distance = 1.0  # 기본값: 앞차 없음
            obj_distances.append([obj_distance])

            # 데이터 누적
            training_img.append(screen)
            controls.append([throttle, steering])
            metrics.append([speed, direction])
            session += 1

            # 속도 60km/h 초과시 경고음
            if speed > 60 and time.time() - alert_time > 1:
                winsound.PlaySound('.\\resources\\alert.wav', winsound.SND_ASYNC)
                alert_time = time.time()

            # 30프레임마다 비동기로 저장
            if len(training_img) % 30 == 0:
                threading.Thread(target=save, args=(training_img, controls, metrics, lanes, obj_distances)).start()
                training_img = []
                controls = []
                metrics = []
                lanes = []
                obj_distances = []

            time.sleep(0.015)  # CPU 부하 줄이기

            # 일시정지/저장/삭제 핸들링
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
                    lanes = []
                    obj_distances = []
                    print('Deleted.')
                else:
                    print('Saved.')

                print('To exit the program press LB or keyboard L.')
                session = 0
                time.sleep(0.5)

        # 녹화 재개
        keys = key_check()
        if gamepad.get_RB() or 'K' in keys:
            pause = False
            print('Unpaused by keyboard or gamepad')
            time.sleep(1)

        # 프로그램 종료
        elif gamepad.get_LB() or 'L' in keys:
            gamepad.close()
            close = True
            print('Saving data and closing the program.')
            save(training_img, controls, metrics, lanes, obj_distances)
            data_file.close()

# 프로그램 시작
if __name__ == '__main__':
    print("✅ data_collect.py 실행됨.")
    main()
