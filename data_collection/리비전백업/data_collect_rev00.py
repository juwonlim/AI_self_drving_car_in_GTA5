
"""
Data collection module (saves data in H5 format).
Saves screen captures and pressed keys into a file
for further trainings of NN.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #이건 현재 파일 기준으로 상위 폴더(=루트) 를 자동으로 찾아주는 방식
#밑에 from data_collection.img_process보다 먼저 나와야 함

import threading
import time
import winsound

import h5py

from data_collection.gamepad_cap import Gamepad   # 조이스틱 입력을 감지하는 클래스 ,gamepad_cap.py는 "키보드를 게임패드처럼 사용하는 대체 모듈"이기 때문에 필수
from data_collection.img_process import img_process # 화면 캡처 및 정보 추출
from data_collection.key_cap import key_check # 키보드 입력 감지



# 루트 폴더 경로를 sys.path에 추가 (상위 모듈 접근 가능하게)


lock = threading.Lock()  # 저장 중 데이터 일관성 유지를 위한 쓰레드 락


# open the data file
# HDF5 파일 저장 경로 (경로는 변경 권장)
#path = "F:\Graduation_Project\\training_data.h5"
path = "training/training_data_by_user_drive.h5"
os.makedirs(os.path.dirname(path), exist_ok=True)  # 이 줄 추가해주면 폴더 없을시에 만들어줌

"""
질문:path = "training/training_data_by_user_drive.h5" 그러면 이렇게 파일명을 지으면, 이거 다른 파일에서 사용되지 않나? 아까 train.py에는 base_model.h5만 호출되던데

ChatGPT의 말:
정확하게 짚었어
path = "training/training_data_by_user_drive.h5" 이렇게 파일명을 정하면
train.py나 다른 파일과 충돌이 나는지 확인해야 하잖아?

결론부터 말하면:
파일명 충돌 안 나. 문제 없어.
이유: data_collect.py에서 지정한 path는 해당 파일 안에서만 사용되는 지역 경로 변수야
train.py는 .h5 파일 경로를 data_collect.py에서 import한 path 변수를 사용하지만,
그걸 네가 직접 바꾸면 train.py도 자동으로 새 파일명을 참조하게 돼

예를 들어
#data_collect.py: 
                   path = "training/training_data_by_user_drive.h5"
#train.py: 
           from data_collection.data_collect import path  # 여기서 불러옴
          data = h5py.File(path, 'r')

따라서 네가 data_collect.py의 path만 바꾸면,
train.py는 아무 것도 수정하지 않아도 새로운 .h5 파일을 사용하게 됨.

"""





# HDF5 파일 열기 (있으면 append 모드, 없으면 새로 생성)
data_file = None
if os.path.isfile(path): #이게  False일 때 h5py.File(..., 'w')로 h5파일 생성
    data_file = h5py.File(path, 'a')
else:
    data_file = h5py.File(path, 'w')
    # Write data in chunks for faster writing and reading by NN
     # 세 개의 데이터셋 생성: 이미지, 조작값, 주행 상태
     # 3개 데이터셋을 만듦
     #.h5 파일은 이 시점에서 생성되는 거야.
    #만약 해당 경로(training/training_data.h5)에 파일이 없다면,
    #h5py.File(..., 'w')를 통해 새 파일을 생성
    #그리고 그 안에 img, controls, metrics 라는 세 개의 데이터셋을 만듬
    data_file.create_dataset('img', (0, 240, 320, 3), dtype='u1',
                             maxshape=(None, 240, 320, 3), chunks=(30, 240, 320, 3))
    data_file.create_dataset('controls', (0, 2), dtype='i1', maxshape=(None, 2), chunks=(30, 2))
    data_file.create_dataset('metrics', (0, 2), dtype='u1', maxshape=(None, 2), chunks=(30, 2)) 


#def save함수
# 데이터를 HDF5 파일에 저장하는 함수
#data_file 은 전역 변수로 이미 열려 있는 .h5 파일 객체야
#따라서 save() 함수에서 다시 파일명을 지정하지 않아도 → 이미 열려 있는 data_file 에 데이터를 저장하고 있음
def save(data_img, controls, metrics):
    with lock:  # make sure that data is consistent ,  # 쓰레드 안전 보장
        if data_img:  # if the list is not empty  ,# 저장할 이미지가 있다면
            # last_time = time.time()
            data_file["img"].resize((data_file["img"].shape[0] + len(data_img)), axis=0)  # 내부 데이터셋에 덧붙임
            data_file["img"][-len(data_img):] = data_img
            data_file["controls"].resize((data_file["controls"].shape[0] + len(controls)), axis=0)
            data_file["controls"][-len(controls):] = controls
            data_file["metrics"].resize((data_file["metrics"].shape[0] + len(metrics)), axis=0)
            data_file["metrics"][-len(metrics):] = metrics
            # print('Saving took {} seconds'.format(time.time() - last_time))

# 최근 세션에서 저장된 데이터를 삭제하는 함수 (최대 500 프레임)
def delete(session):
    frames = session if session < 500 else 500
    data_file["img"].resize((data_file["img"].shape[0] - frames), axis=0)
    data_file["controls"].resize((data_file["controls"].shape[0] - frames), axis=0)
    data_file["metrics"].resize((data_file["metrics"].shape[0] - frames), axis=0)




#키보드 사용가능하게 개선된 main함수
#녹화 시작 	'K' or RB
#녹화 일시정지	'P' or RB
#마지막 15초 저장 여부	'Y', 'N'
#프로그램 종료	'L' or LB
def main():
    # 게임패드 초기화
    gamepad = Gamepad()
    gamepad.open()

    alert_time = time.time()
    close = False
    pause = True
    session = 0
    training_img = []
    controls = []
    metrics = []

    print("Press RB on your gamepad or keyboard 'K' to start recording")
    print("🔥 프로그램 시작됨. K를 눌러 녹화를 시작하세요.")


    while not close: #프로그램 전체 실행 루프
        while not pause:   # 데이터 수집 루프 (녹화 상태)
            throttle, steering = gamepad.get_state()
            ignore, screen, speed, direction = img_process("Grand Theft Auto V") #img_process가 호출되는 구간

            training_img.append(screen)
            controls.append([throttle, steering])
            metrics.append([speed, direction])
            session += 1

            if speed > 60 and time.time() - alert_time > 1:
                winsound.PlaySound('.\\resources\\alert.wav', winsound.SND_ASYNC)
                alert_time = time.time()

            if len(training_img) % 30 == 0:
                threading.Thread(target=save, args=(training_img, controls, metrics)).start()
                training_img = []
                controls = []
                metrics = []

            time.sleep(0.015)

            # 일시정지: 게임패드 RB 또는 키보드 'P'
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
                    print('Deleted.')
                else:
                    print('Saved.')

                print('To exit the program press LB or keyboard L.')
                session = 0
                time.sleep(0.5)

        # 녹화 재시작: 게임패드 RB 또는 키보드 'K'
        #Puase상태일떄 녹화멈춤
        keys = key_check()
        if gamepad.get_RB() or 'K' in keys:
            pause = False
            print('Unpaused by keyboard or gamepad')
            time.sleep(1)

        # 종료: 게임패드 LB 또는 키보드 'L'
        # 즉,프로그램 종료  
        elif gamepad.get_LB() or 'L' in keys:
            gamepad.close()
            close = True
            print('Saving data and closing the program.')
            save(training_img, controls, metrics)

    data_file.close()


"""
# 메인 루프
def main():
    # initialize gamepad
    # 게임패드 초기화
    gamepad = Gamepad()
    gamepad.open()

    # last_time = time.time()   # to measure the number of frames
    alert_time = time.time()  # to signal about exceeding speed limit ,# 속도 알림 타이머
    close = False  # to exit execution , # 종료 플래그
    pause = True  # to pause execution, # 일시정지 상태
    session = 0  # number of frames recorded in one session, # 현재 세션 프레임 수
    training_img = []  # lists for storing training data, # 이미지 저장 리스트
    controls = []  # 조작값 저장 리스트
    metrics = []  # 주행정보 저장 리스트

    print("Press RB on your gamepad to start recording")
    while not close:
        while not pause:
            # 게임패드 입력 읽기
            # read throttle and steering values from the gamepad
            throttle, steering = gamepad.get_state()
            # get screen, speed and direction
             # 화면 캡처 및 속도/방향 추출
            ignore, screen, speed, direction = img_process("Grand Theft Auto V")

            training_img.append(screen)
            controls.append([throttle, steering])
            metrics.append([speed, direction])
            session += 1


              # 속도가 60 이상일 때 알림음 재생 (1초 간격)
            if speed > 60 and time.time() - alert_time > 1:
                winsound.PlaySound('.\\resources\\alert.wav', winsound.SND_ASYNC)
                alert_time = time.time()

            # save the data every 30 iterations
              # 30프레임마다 쓰레드로 저장
            if len(training_img) % 30 == 0:
                # print("-" * 30 + "Saving" + "-" * 30)
                threading.Thread(target=save, args=(training_img, controls, metrics)).start()
                training_img = []
                controls = []
                metrics = []

            time.sleep(0.015)  # in order to slow down fps 
                               # 프레임 속도 조절 (약 60 FPS)
            # print('Main loop took {} seconds'.format(time.time() - last_time))
            # last_time = time.time()

            # RB 버튼을 누르면 일시정지 + 저장 여부 선택
            if gamepad.get_RB():
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
                    print('Deleted.')
                else:
                    print('Saved.')

                print('To exit the program press LB.')
                session = 0
                time.sleep(0.5)
        # RB 누르면 다시 시작, LB 누르면 종료
        if gamepad.get_RB():
            pause = False
            print('Unpaused')
            time.sleep(1)
        elif gamepad.get_LB():
            gamepad.close()
            close = True
            print('Saving data and closing the program.')
            save(training_img, controls, metrics)

    data_file.close()

     """




# 실행 시작
if __name__ == '__main__':
    print("✅ data_collect.py 실행됨.")

    main()



"""
1. 시작: data_collect.py 실행
   └─ FOLDER: training/
   └─ FILE: training_data.h5 (없으면 새로 생성)

2. 생성 시 구조:
   └─ img       (240x320 RGB 이미지들)
   └─ controls  (throttle + steering)
   └─ metrics   (speed + direction)

3. 이후
   └─ 30프레임마다 save() 호출 → 위 3개 데이터셋에 내용 추가


"""



"""
[STEP 1] 직접 운전하며 h5 생성 → data_collect.py
    └─ 결과: training_data.h5 또는 dataset.h5 같은 이름의 HDF5 파일

[STEP 2] 모델 학습 → train.py
    └─ 위에서 만든 h5 파일을 기반으로 모델을 학습

"""


'''
RB 누르면 주행 데이터 수집 시작

화면 캡처 + 조작값 + 속도/방향 정보 저장

30프레임마다 h5에 저장

다시 RB 누르면 일시정지 → Y/N로 최근 데이터 저장 여부 결정

LB 누르면 종료

'''