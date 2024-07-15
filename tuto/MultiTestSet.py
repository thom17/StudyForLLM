"""
20204-07-15
SimpleTestSet 에서 파라미터를 추가하여 학습하기 위해 생성
여러개의 파라미터 역시 내가 임의로 직접 생성하여 태스트하는게 좋을듯
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

import itertools

from datetime import datetime
import tensorflow as tf
class MultiTestSet:
    """
    범위와 입출력 함수를 받아 데이터 셋을 생성
    일단 파라미터는 x 하나에 대하여 조합 형태만 고려

    가능하면 numpy 배열을 기본 배열로 사용. (메모리 효율적으로 이득)
    """
    def __init__(self, range_size=100, end=100, start=0, rand_seed=0, param_size=3, combination_rate=1.0, func=None):
        n = (end - start) / range_size
        self.sorted_x = np.arange(start, end, n).reshape(-1, 1)
        self.data_range = range_size
        self.start = start
        self.end = end

        self.__rand_seed = rand_seed
        np.random.seed(rand_seed)
        tf.random.set_seed(rand_seed)

        #여기까지는 기본적으로 동일

        #여기 부터 조합 관련하여 변경
        self.param_size = param_size
        self.combination_rate = combination_rate
        start = time.time()

        #x에 대하여 조합으로 파라미터 생성
        self.pram_comb_list = list(itertools.combinations_with_replacement(self.sorted_x, param_size))
        #무작위 순서로 입력 값 생성
        self.shuffled_x = np.random.permutation(self.pram_comb_list).reshape(-1, param_size)
        self.comb_size = len(self.shuffled_x)
        self.data_size = int(self.comb_size * self.combination_rate)
        end = time.time()
        print(f"Make Combination time {end-start} sec, len {self.data_size}")


        self.func = func
        if func is None:
            self.func = lambda x: x[0] + x[1] + x[2]

        start = time.time()
        self.sorted_y = np.zeros(self.comb_size)
        for i in range(self.comb_size):
            self.sorted_y[i] = self.func(self.pram_comb_list[i])




        self.shuffled_y = np.zeros(self.comb_size)
        for i in range(self.comb_size):
            self.shuffled_y[i]= self.func(self.shuffled_x[i])

        end = time.time()
        print(f"Make Y time {end - start} sec")

        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        self.shuffled_xn = self.scaler_x.fit_transform(self.shuffled_x)
        self.shuffled_yn = self.scaler_y.fit_transform(self.shuffled_y.reshape(-1, 1)).flatten()



    def set_data_size(self, size_num: int):
        self.data_size = size_num

    def set_train_model(self, model: Sequential):
        self.model = model

    def train_and_test(self, test_rate=0.10, epochs=200):
        test_start_idx = self.data_size - int(self.data_size * test_rate)

        train_x = self.shuffled_xn[:test_start_idx]
        train_y = self.shuffled_yn[:test_start_idx]

        test_x = self.shuffled_xn[test_start_idx:self.data_size]
        test_y = self.shuffled_y[test_start_idx:self.data_size]

        self.model.fit(train_x, train_y, epochs=epochs, verbose=1)
        predict_yn = self.model.predict(test_x)
        predict_y = self.scaler_y.inverse_transform(predict_yn.reshape(-1, 1)).flatten()


        # self.make_all_3d_graphs(test_rate=test_rate, predict_y=predict_y)

        return predict_y

    def get_train_set(self, test_rate = 0.10):
        test_idx = int(self.data_size * test_rate)

        shuffled_x = np.copy(self.shuffled_x)
        shuffled_y = np.copy(self.shuffled_y)

        #train x, y test x, y
        return shuffled_x[:-test_idx], shuffled_y[:-test_idx], shuffled_x[-test_idx:], shuffled_y[-test_idx:]

    def make_pair_plot(self, test_rate, predict_y):
        import numpy as np
        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt


        train_x, train_y, test_x, test_y = self.get_train_set(test_rate)

        # 데이터프레임으로 변환
        df = pd.DataFrame(test_x, columns=['x1', 'x2', 'x3'])
        df['y_actual'] = test_y
        df['y_predicted'] = predict_y


        #합치기 태스트
        # DataFrame에 'Type' 컬럼 추가하여 실측값과 예측값 구분
        df_actual = df[['x1', 'x2', 'x3', 'y_actual']]
        df_actual['Type'] = 'Actual'

        df_predicted = df[['x1', 'x2', 'x3', 'y_predicted']]
        df_predicted.columns = ['x1', 'x2', 'x3', 'y_actual']  # y_actual 컬럼으로 이름 통일
        df_predicted['Type'] = 'Predicted'

        # 두 DataFrame 합치기
        df_combined = pd.concat([df_actual, df_predicted])

        # Pairplot 생성
        sns.pairplot(df_combined, hue='Type', vars=['x1', 'x2', 'x3', 'y_actual'],  palette='bright')
        plt.show()






        # #두개의 값 동시에 비교
        # df_melted = pd.melt(df, id_vars=['x1', 'x2', 'x3', 'ans_y'], value_vars=['predict_y'], var_name='type',
        #                     value_name='value')

        # sns.pairplot(df, vars=['x1', 'x2', 'x3', 'ans_y'], hue='predict_y', palette='viridis')
        # plt.suptitle('Pair Plot for True and Predicted Values')#, y=1.02)
        # plt.show()


        # # Seaborn의 pairplot을 사용하여 데이터 시각화
        # sns.pairplot(df)
        # plt.suptitle('Pair Plot for All Variables', y=1.02)
        # plt.show()

    def make_3d_xyz_graph(self, x1, x2, y, predict_y, x1_label, x2_label, y_label):
        # 정확한 값 빨간색으로

        # 3D 산점도
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel(x1_label)
        ax.set_ylabel(x2_label)
        ax.set_zlabel(y_label)
        ax.set_title(f'f({x1_label}, {x2_label}) = {y_label}')

        print(x1_label, x1.shape)
        print(y_label, y.shape)
        print(x2_label, x2.shape)
        print("predict",predict_y.shape)


        ax.scatter(x1,x2,y, c='r')
        ax.scatter(x1, x2, predict_y, c='g')

        # 실제 값과 예측 값 연결하는 라인
        for i in range(len(y)):
            ax.plot([x1[i], x1[i]], [x2[i], x2[i]], [predict_y[i], y[i]], color='gray')

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"../result/multi_3d_plot_{current_time}({x1_label}{x2_label}={y_label}).png"
        plt.savefig(filename)
        plt.show()
    def make_all_3d_graphs(self, test_rate=0.10, predict_y = None):

        train_x, train_y, test_x, test_y = self.get_train_set(test_rate)

        x = test_x[:, 0]
        y = test_x[:, 1]
        z = test_x[:, 2]
        #
        #
        # # 3D 산점도
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x, y , z, c=colors)
        #
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # ax.set_title('3D Scatter Plot')
        #
        # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # filename = f"../result/multi_3d_plot_{current_time}.png"
        # plt.savefig(filename)
        # plt.show()

        self.make_3d_xyz_graph(x1=x, x2=z, y=test_y, predict_y=predict_y, x1_label="num", x2_label="add", y_label="y")
        self.make_3d_xyz_graph(x1=x, x2=y, y=test_y, predict_y=predict_y, x1_label="num", x2_label="div", y_label="y")
        self.make_3d_xyz_graph(x1=y, x2=z, y=test_y, predict_y=predict_y, x1_label="div", x2_label="add", y_label="y")


    def make_2d_xy_graph(self, test_rate=0.10, predict_y = None):

        train_x, train_y, test_x, test_y = self.get_train_set(test_rate)

        train_x = [d[0] for d in train_x ]
        test_x = [d[0] for d in test_x]

        sorted_x = [d[0] for d in self.pram_comb_list]


        plt.figure(figsize=(10, 6))
        plt.plot(sorted_x, self.sorted_y, label='Original Data', color='lightgray')
        plt.scatter(train_x, train_y, label='Train Set', color='blue', alpha=0.3)
        if predict_y is not None:
            plt.scatter(test_x, predict_y, label='Predict', color='green', alpha=0.8)

        plt.scatter(test_x, test_y, label='Test Set', color='red', alpha=0.3)


        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.title('multi_2d_plot (Only x)')


        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"../result/multi_2d_plot_{current_time}.png"
        plt.savefig(filename)
        plt.show()



def fun1(x):
    return 3*x - (x*x)


if __name__ == "__main__":
    MultiTestSet()