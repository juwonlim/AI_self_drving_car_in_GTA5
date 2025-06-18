#!/usr/bin/env python3
# lanenet_predict.py

import sys
import os
# 현재 파일 위치 기준으로 lanenet_model 폴더 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__)))
import tensorflow as tf
import numpy as np
import cv2
import yaml

#from lanenet_model import lanenet  # GitHub lanenet-lane-detection의 lanenet_model 사용
from lanenet_inference.lanenet_model import lanenet

from config import global_config

# Config 로드
CFG = global_config.cfg
CFG.BACKEND = 'tensorflow'  # 혹은 'tensorflow' 그대로

def predict_lane(image_path, weights_path):
    """
    이미지 1장을 받아서 LANENet으로 차선 추론을 수행하고, 결과를 리턴한다.
    :param image_path: 테스트 이미지 경로
    :param weights_path: checkpoint 파일명 (확장자 없는 경로, 예: './model/tusimple_lanenet/tusimple_lanenet.ckpt')
    :return: binary_seg_image, instance_seg_image (numpy 배열)
    """

    # TensorFlow 세션 설정
    tf.reset_default_graph()
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    phase_tensor = tf.constant('test', dtype=tf.string)

    # LANENet 모델 로드
    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor, phase_tensor)

    # Saver 준비
    saver = tf.train.Saver()

    # 이미지 로드 & 전처리
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image / 127.5 - 1.0
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가

    with tf.Session() as sess:
        # weight 로드
        saver.restore(sess, weights_path)

        binary_seg_image, instance_seg_image = sess.run(
            [binary_seg_ret, instance_seg_ret],
            feed_dict={input_tensor: image}
        )

        binary_seg_image = binary_seg_image[0]  # (256, 512, 1)
        instance_seg_image = instance_seg_image[0]  # (256, 512, 4)

        # 결과 후처리 (binary_seg_image를 0~255로 변환)
        binary_seg_image = (binary_seg_image * 255).astype(np.uint8)

        return binary_seg_image, instance_seg_image


def main():
    # 테스트 예제
    test_image = './data/test_image.jpg'
    weights_path = './model/tusimple_lanenet/tusimple_lanenet.ckpt'

    binary_mask, instance_mask = predict_lane(test_image, weights_path)

    # 결과 저장
    cv2.imwrite('./output/binary_mask.png', binary_mask)

    # 결과 시각화 (옵션)
    cv2.imshow('binary_mask', binary_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
