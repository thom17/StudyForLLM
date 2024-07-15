import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler


class SimpleTestSet:
    """
    범위와 입출력 함수를 받아 데이터 셋을 생성
    일단 f(x) 형태만 고려
    """

    def __init__(self, size=1000, end=1000, start=0, rand_seed=0, func=None):
        n = (end - start) / size
        self.sorted_x = np.arange(start, end, n).reshape(-1, 1)
        self.data_size = size
        self.start = start
        self.end = end

        self.__rand_seed = rand_seed
        np.random.seed(rand_seed)

        self.func = func
        if func is None:
            self.func = lambda x: x

        self.sorted_y = np.zeros(size)
        for i in range(self.data_size):
            self.sorted_y[i] = self.func(self.sorted_x[i])

        self.shuffled_x = np.random.permutation(self.sorted_x)
        self.shuffled_y = np.zeros(size)
        for i in range(self.data_size):
            self.shuffled_y[i] = self.func(self.shuffled_x[i])

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

        # 정렬된 X 값으로 그래프 그리기
        sorted_idx = np.argsort(test_x.flatten())
        test_x_sorted = test_x[sorted_idx]
        test_y_sorted = test_y[sorted_idx]
        predict_y_sorted = predict_y[sorted_idx]

        self.make_graph(test_x_sorted, test_y_sorted, predict_y=predict_y_sorted)

    def get_train_set(self, test_rate=0.10):
        test_idx = int(self.data_size * test_rate)
        shuffled_x = np.copy(self.shuffled_x)
        shuffled_y = np.copy(self.shuffled_y)
        return shuffled_x[:-test_idx], shuffled_y[:-test_idx], shuffled_x[-test_idx:], shuffled_y[-test_idx:]

    def make_graph(self, test_x, test_y, predict_y=None):
        train_x, train_y, _, _ = self.get_train_set(test_rate=0.10)
        plt.figure(figsize=(10, 6))
        plt.plot(self.sorted_x, self.sorted_y, label='Original Data', color='lightgray')
        plt.scatter(train_x, train_y, label='Train Set', color='blue', alpha=0.6)
        if predict_y is not None:
            plt.plot(self.scaler_x.inverse_transform(test_x), predict_y, label='Predict', color='green')
        plt.scatter(self.scaler_x.inverse_transform(test_x), test_y, label='Test Set', color='red', alpha=0.6)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.title('Train and Test Sets')
        plt.show()


def fun1(x):
    return 3 * (x - 2) ** 2 + 5


if __name__ == "__main__":
    test_set = SimpleTestSet(start=-10, end=10, size=100, func=fun1)

    # 모델 정의
    model = Sequential()
    model.add(Dense(64, input_dim=1, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    test_set.set_train_model(model)
    test_set.train_and_test(test_rate=0.10, epochs=200)
