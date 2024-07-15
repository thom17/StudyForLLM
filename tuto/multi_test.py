"""
진짜 선형 함수 자체는 학습이 쉽고 결과도 잘나올것
먼저 예측하기 어려운 (분기같은) 임의의 가상의 함수를 정의해서
해당 모델을 예측할 수 있는지 태스트
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda, Dropout, BatchNormalization, LSTM

import tensorflow as tf
from MultiTestSet import MultiTestSet

def test_simple(x):
    return 2 * x[0] + - x[2]

def fake_simple(x):
    """
    3개의 파라미터중 하나만 유효한 경우
    (페이크 파라미터 태스트)
    :param x:
    :return:
    """
    x = x[0]
    return 3 * (x - 2) * (x-2) -5

def test_fun_369(x):
    """
    소수는 버림. 학습때는 일부러 넣어서 태스트
    :param x:
    :return:
    """
    count = 0
    num = x[0]
    div_num = x[1]
    add_num = x[2]

    if div_num is 0:
        div_num = 1

    while 0 < num:
        d = num % 10
        if 0 < d and d % div_num is 0:
            count += 1
        num = int(num/10)

    return count * add_num + x[0]

def test_draw(x):
    if x < 80:
        return 99.8 ** x
    else:
        return 0

def make_model():
    # 딥러닝 모델 생성
    model = Sequential([
        Dense(32, activation='relu', input_shape=(3,)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='relu')
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
    import tensorflow as tf

    print(tf.__version__)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    data_set = MultiTestSet(range_size= 100, start = -10, end=10, func=test_fun_369, rand_seed=2)
    # data_set.make_graph()



    # x= data_set.shuffled_x[:10, 0]
    # y = data_set.shuffled_x[:10, 1]
    # z = data_set.shuffled_x[:10, 2]
    # ans = np.array(data_set.sorted_y[:10])
    # predict_y = ans * 2
    # data_set.make_3d_xyz_graph(x1=x, x2=z, y=ans, predict_y=predict_y, x1_label="2x", x2_label="-z", y_label="y")

    model = make_model()

    test_rate = 0.001

    # data_set.set_data_size(10000)
    data_set.set_train_model(model)
    predict_y = data_set.train_and_test(epochs = 2, test_rate=test_rate)

    data_set.make_all_3d_graphs(test_rate=test_rate, predict_y=predict_y)

    data_set.make_pair_plot(test_rate=test_rate, predict_y=predict_y)