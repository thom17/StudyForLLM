"""
진짜 선형 함수 자체는 학습이 쉽고 결과도 잘나올것
먼저 예측하기 어려운 (분기같은) 임의의 가상의 함수를 정의해서
해당 모델을 예측할 수 있는지 태스트
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda, Dropout, BatchNormalization, LSTM

import tensorflow as tf
from simpleTestSet import SimpleTestSet

def test_simple(x):
    return 3 * (x - 2) * (x-2) -5

def test_fun_369(x):
    """
    소수는 버림. 학습때는 일부러 넣어서 태스트
    :param x: 
    :return: 
    """
    count = 0
    x_abs = abs(x)
    num = int(x_abs)
    while 0 < num:
        d = num%10
        if 0 < d and d%2 is 0:
            count += 1
        num = int(num/10)
    return count * 100 + x

def test_draw(x):
    if x < 80:
        return 99.8 ** x
    else:
        return 0

def lstm_model(data_set):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Input

    train_x, train_y, test_x, test_y = data_set.get_train_set()

    # LSTM 모델 생성
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(train_x.shape[1], 1)))
    model.add(Dense(64, activation='relu'))
    # model.add(LSTM(64, activation='relu'))

    model.add(Dense(1))

    # 모델 컴파일
    model.compile(optimizer='adam', loss='mean_squared_error')

    data_set.set_train_model(model)
    data_set.train_and_test(epochs = 200)

def basic_model(data_set):
    train_x, train_y, test_x, test_y = data_set.get_train_set()

    # 딥러닝 모델 생성
    model = Sequential([
        Dense(32, activation='relu', input_shape=(1,)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='relu')
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    data_set.set_train_model(model)
    data_set.train_and_test(epochs = 200)

    # # 모델 학습
    # model.fit(train_x, train_y, epochs=200, verbose=1)
    #
    # predict_y = model.predict(test_x)
    # data_set.make_graph(predict_y=predict_y)

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    data_set = SimpleTestSet(size = 100, start = 0, end=100, func=test_draw)
    # data_set.make_graph()

    basic_model(data_set)
