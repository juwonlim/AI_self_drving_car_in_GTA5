#차선, 앞차 인식 데이터 수집까지 완성된 버전
#차간거리 유지는 아직임

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 상위 폴더 접근 가능하게

import threading #여러작업동시처리, 즉 ,주행 끊김없이 데이터 저장 목적
import time #프로그램 속도를 제어하거나, 시간 경과를 측정할 때 , ex) sleep(0.015), time()
import winsound #속도 초과 경고 같은 걸 소리로 알려주기 위해

import h5py  # HDF5 파일 입출력용

from data_collection.gamepad_cap import Gamepad  # 게임패드/키보드 입력 감지
from data_collection.img_process import img_process  # GTA5 화면 캡처 및 처리
from data_collection.key_cap import key_check  # 키보드 입력 감지

### [추가] 차선 인식, YOLO 객체 인식 모듈
from object_detection.lane_detect import detect_lane
from object_detection.lane_detect import draw_lane
from object_detection.object_detect import yolo_detection
import cv2

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
    data_file.create_dataset('img', (0, 180, 320, 3), dtype='u1', maxshape=(None, 180, 320, 3), chunks=(30, 180, 320, 3))
    data_file.create_dataset('controls', (0, 2), dtype='i1', maxshape=(None, 2), chunks=(30, 2))
    data_file.create_dataset('metrics', (0, 2), dtype='u1', maxshape=(None, 2), chunks=(30, 2))
    ### [추가]
    data_file.create_dataset('lanes', (0, 4), dtype='i2', maxshape=(None, 4), chunks=(30, 4))  # 왼쪽/오른쪽 차선 좌표
    data_file.create_dataset('obj_distance', (0, 1), dtype='f2', maxshape=(None, 1), chunks=(30, 1))  # 앞차 거리

# 데이터를 저장하는 함수
def save(data_img, controls, metrics, lanes, obj_distances):
    with lock:
        # 각 데이터셋 존재 여부 확인 후 없으면 생성
        if 'img' not in data_file:
            data_file.create_dataset('img', (0, 180, 320, 3), maxshape=(None, 180, 320, 3),
                                     dtype='u1', chunks=(30, 180, 320, 3))
        if 'controls' not in data_file:
            data_file.create_dataset('controls', (0, 2), maxshape=(None, 2),
                                     dtype='f', chunks=(30, 2))
        if 'metrics' not in data_file:
            data_file.create_dataset('metrics', (0, 2), maxshape=(None, 2),
                                     dtype='f', chunks=(30, 2))
        if 'lanes' not in data_file:
            data_file.create_dataset('lanes', (0, 4), maxshape=(None, 4),
                                     dtype='i', chunks=(30, 4))
        if 'obj_distance' not in data_file:
            data_file.create_dataset('obj_distance', (0,), maxshape=(None,),
                                     dtype='f', chunks=(30,))

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
#P로 일시정지하고 → N 누르면 최근 최대 500프레임(약 15초) 분량 데이터를 삭제 --> 잘못운행한 구간 삭제하는 기능
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
    print("프로그램 시작됨. K를 눌러 녹화를 시작하세요.")

    while not close:
        while not pause:
            throttle, steering = gamepad.get_state()  # 게임패드로부터 throttle, steering 읽기
            ignore, screen, speed, direction = img_process("Grand Theft Auto V")  # 화면 캡처 및 차량 속도
            

            #lane_detect.py파일의 73,366줄에서 발생에러, attributeError : 'NoneType' object has no attribute 'shape'를 예방하기 위한 if else문
            #화면캡쳐 실패시 skip
            if screen is None:
                continue # 화면이 없으면 그냥 넘어가고 정상화면이면 차선,객체 검출진행
                
            #else:
                #continue  # 화면 캡처 실패 시 skip

            #이 아래는 screen이 정상일 때만 실행 (차선,객체 검출) 
            #lane, stop_line = detect_lane(screen) #lane_detect.py파일에서 함수 호출하여 
            (lane, stop_line), lane_img = detect_lane(screen) #여기서 lane은 ㅣanes이고 stop_line은 stop_line임. 즉 lane_detect.py에서 construct_lane()의 두값을 그대로 data_collect.py에서 **(lane,stope_line)**으로 받아서 분해해 저장

            
            # draw_lane 하기 전에 이미지 복사해서 확인 --lane view라는 작은 창이 열림, 내가 운전할 떄 debug lane view외에도 작은 창이 따로 열리는것
            #view = screen.copy()
            #view[280:-130, :, :] = draw_lane(view[280:-130, :, :], lane, stop_line, [0, 255, 0], [0, 255, 0])
            #cv2.imshow("Lane View", view)



            # 시각화용 차선 그리기
            #screen[280:-130, :, :] = draw_lane(screen[280:-130, :, :], lane, stop_line, [0, 255, 0], [0, 255, 0])
            #cv2.imshow("Raw GTA5 Screen", screen)  # 원본 사이즈 화면 확인


            # draw_lane 하기 전에 이미지 복사해서 디버깅용 확인용 화면 구성
            debug_view = cv2.resize(screen.copy(), (640, 360))  # 16:9 비율 유지하며 디버깅용 축소
            debug_view = draw_lane(debug_view, lane, stop_line, [0, 255, 0], [0, 255, 0])
            #cv2.imshow("Debug Lane View", debug_view) #창이 너무 많아서 띄우지 않기로

            # 시각화용 차선 그리기 (훈련용)
            screen[280:-130, :, :] = draw_lane(screen[280:-130, :, :], lane, stop_line, [0, 255, 0], [0, 255, 0])
            cv2.imshow("Lane View", screen)  # 실시간 확인용 ,size는 1280x720
            cv2.waitKey(1) #여기까지 시각화용 차선 그리기

            #아래3줄, 차선 좌표 포맷가공
            left_lane = lane[0] if lane[0] else [0, 0, 0, 0]
            right_lane = lane[1] if lane[1] else [0, 0, 0, 0]
            lanes.append([left_lane[0], left_lane[2], right_lane[0], right_lane[2]])

            ### [추가] YOLO로 앞차 감지
            _, _, obj_distance = yolo_detection(screen, direct=0)
            if obj_distance is None:
                obj_distance = 1.0  # 기본값: 앞차 없음
            obj_distances.append([obj_distance])

            # 데이터 누적
            #resized = cv2.resize(screen, (320, 180)) #이거 때문에 ROI가 잡히지 않아서 주석처리함. 
            #training_img.append(resized) #320,180으로 리사이즈 된 이미지를 학습에 넘겨줌
            #training_img.append(screen) #학습용 screen은 training_img.append(screen)으로 그대로 저장됨 (리사이즈된 건 아님) ,실제 학습에 사용되는 이미지임!!!!!screen은 원래 크기 (1280x720) 그대로 저장됨
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
        #print(f"[DEBUG] Key input: {keys}")  # <- 추가 , 입력값 출력하는 코드,  디버깅 용도, 너무 디버그 줄이 계속 나와서 주석처리
        if 'K' in keys:
            print(f"[DEBUG] Key input: {keys}")
        if gamepad.get_RB() or 'K' in keys:
            pause = False
            print('Unpaused by keyboard or gamepad')
            time.sleep(1)

        # 정확히 'L'만 눌렸을 때만 종료
        elif gamepad.get_LB() or ('L' in keys and len(keys) == 1): # 이 elif문 아래에 5개의 코드줄이 없으면 아래에 if --name__문에서 에러발생
                                                                   #이 조건이 빈 코드 블럭(아무 동작도 없음)이라, 이후의 실제 종료처리 코드가 주석 처리되어 있어도 main() 루프가 빠져나와 종료될 수 있음
                                                                   # close = True 설정이 없으면 while 루프는 계속 돌아야 하는데, 이 부분도 빠져 있어 예상치 못한 종료가 일어나는 겁니다
               print("Saving data and closing the program.")
               save(training_img, controls, metrics, lanes, obj_distances) #이미지,차선,거리,조작 정보 저장
               data_file.close()
               gamepad.close()
               close = True
               

        
        # 프로그램 종료
        #elif gamepad.get_LB() or 'L' in keys:
         #   gamepad.close()
          #  close = True
           # print('Saving data and closing the program.')
            #save(training_img, controls, metrics, lanes, obj_distances)
            #data_file.close()
        



# 프로그램 시작
if __name__ == '__main__':
    print("data_collect.py 실행됨.")
    main()
