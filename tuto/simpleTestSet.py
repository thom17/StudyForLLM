"""
20204-07-12
1차 적으로 선형 회귀 모델부터 차근 차근 학습
그를 위한 데이터셋 제조 클래스
"""
import numpy
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
class SimpleTestSet:
    """
    범위와 입출력 함수를 받아 데이터 셋을 생성
    일단 f(x) 형태만 고려
    """
    def __init__(self, size=1000, end=1000, start=0, rand_seed=0, func=None):
        n = (end - start)/size
        self.sorted_x = np.arange(start, end, n).reshape(-1, 1)
        self.data_size = size
        self.start = start
        self.end = end

        self.__rand_seed = rand_seed
        np.random.seed(rand_seed)
        tf.random.set_seed(rand_seed)

        self.func = func
        if func is None:
            self.func = lambda x: x

        self.sorted_y = np.zeros(size)
        for i in range(self.data_size):
            self.sorted_y[i]= self.func(self.sorted_x[i])

        self.shuffled_x = np.random.permutation(self.sorted_x)
        self.shuffled_y = np.zeros(size)
        for i in range(self.data_size):
            self.shuffled_y[i]= self.func(self.shuffled_x[i])

        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        self.shuffled_xn = self.scaler_x.fit_transform(self.shuffled_x)
        self.shuffled_yn = self.scaler_y.fit_transform(self.shuffled_y.reshape(-1, 1)).flatten()

    def set_train_model(self, model: Sequential):
        self.model = model

    def train_and_test(self, test_rate=0.10, epochs=200):
        test_idx = int(self.data_size * test_rate)
        train_x = self.shuffled_xn[:-test_idx]
        train_y = self.shuffled_yn[:-test_idx]

        test_x = self.shuffled_xn[-test_idx:]
        test_y = self.shuffled_y[-test_idx:]

        self.model.fit(train_x, train_y, epochs=epochs, verbose=1)
        predict_yn = self.model.predict(test_x)
        predict_y = self.scaler_y.inverse_transform(predict_yn.reshape(-1, 1)).flatten()

        self.make_graph(predict_y=predict_y)

    def get_train_set(self, test_rate = 0.10):
        test_idx = int(self.data_size * test_rate)

        shuffled_x = np.copy(self.shuffled_x)
        shuffled_y = np.copy(self.shuffled_y)

        #train x, y test x, y
        return shuffled_x[:-test_idx], shuffled_y[:-test_idx], shuffled_x[-test_idx:], shuffled_y[-test_idx:]

    def make_graph(self, test_rate=0.10, predict_y = None):

        train_x, train_y, test_x, test_y = self.get_train_set(test_rate)

        plt.figure(figsize=(10, 6))
        plt.plot(self.sorted_x, self.sorted_y, label='Original Data', color='lightgray')
        plt.scatter(train_x, train_y, label='Train Set', color='blue', alpha=0.3)
        if predict_y is not None:
            plt.scatter(test_x, predict_y, label='Predict', color='green', alpha=0.8)

        plt.scatter(test_x, test_y, label='Test Set', color='red', alpha=0.3)


        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.title('Train and Test Sets')
        plt.show()

def fun1(x):
    return 3*x - (x*x)


if __name__ == "__main__":


    test_set = SimpleTestSet(start=-10, end=10, size=100 ,func=fun1)
    train_x, train_y, test_x, test_y = test_set.get_train_set(0.01)

    print(len(train_x))
    print(len(train_y))

    print(len(test_x))
    print(len(test_y))

    test_set.make_graph()