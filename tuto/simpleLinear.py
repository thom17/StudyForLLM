# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import numpy as np
# import matplotlib.pyplot as plt
#
#
#
# # 데이터 생성
# np.random.seed(0)
# rands = np.random.rand(500, 1)
#
# X = 2 * rands
#
# i = 0
# y = np.zeros(500)
# for r in X:
#     if r < 0.5 or (1.5 < r and r<2):
#         d = 1
#     else:
#         d = -1
#     y[i] = d + 4 + 3 * r
#     i+=1
#
# activations = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'selu']
#
# # 딥러닝 모델 생성
# model = Sequential([
#     Dense(64, input_dim=1, activation='relu'),  # 은닉층
#     Dense(64, activation='relu'),  # 은닉층
#     Dense(64, activation='relu'),  # 은닉층
#     Dense(64, activation='relu'),  # 은닉층
#     Dense(64, activation='relu'),  # 은닉층
#     Dense(1)  # 출력층
# ])
#
# # 모델 컴파일
# model.compile(optimizer='adam', loss='mean_squared_error')
#
# # 모델 학습
# model.fit(X, y, epochs=300, verbose=0)
#
# # 예측
# X_new = np.array([[0], [2]])
# y_predict = model.predict(X_new)
#
# print(y_predict)
#
# # 결과 시각화
# plt.scatter(X, y)
# plt.plot(X_new, y_predict, "r-", linewidth=2)
# plt.xlabel("X")
# plt.ylabel("y")
# plt.show()
#
#
# test_x = np.arange(0.0, 10, 0.1)
# test_y = model.predict(test_x)
#
#
# # 결과 시각화
# plt.scatter(test_x, test_y)
# # plt.plot(X_new, y_predict, "r-", linewidth=2)
# plt.xlabel("X")
# plt.ylabel("y")
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda

# 데이터 생성
np.random.seed(0)
rands = np.random.rand(500, 1)
X = 2 * rands

y = np.zeros(500)
for i, r in enumerate(X):
    if r < 0.5 or (1.5 < r and r < 2):
        d = 2
    else:
        d = -2
    y[i] = d + r * 0.5

# 주기적인 특성을 반영한 입력 데이터 변환
X_sin = np.sin(X * np.pi)



print(X.shape)
print(X_sin.shape)


X = np.hstack((X, X_sin))
print(X.shape)
# 딥러닝 모델 생성
model = Sequential([
    Dense(64, input_dim=2, activation='relu'),  # 첫 번째 은닉층
    Dense(64, activation='relu'),               # 두 번째 은닉층
    Dense(64, activation='relu'),               # 세 번째 은닉층
    Dense(1)                                    # 출력층
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
model.fit(X, y, epochs=200, verbose=1)


test_x = np.arange(0.0, 10, 0.1).reshape(-1, 1)
test_x_sin = np.sin(test_x * np.pi)

print(test_x.shape)
print(test_x_sin.shape)

test_x_set = np.hstack((test_x, test_x_sin))
test_y = model.predict(test_x_set)

ans = np.zeros(len(test_x))
for i, r in enumerate(test_x):
    if r < 0.5 or (1.5 < r and r < 2):
        d = 2
    else:
        d = -2
    ans[i] = d + r * 0.5




# 결과 시각화
plt.scatter(test_x, test_y)
# plt.plot(X_new, y_predict, "r-", linewidth=2)
plt.xlabel("X")
plt.ylabel("y")
plt.show()


plt.scatter(test_x, ans)
# plt.plot(X_new, y_predict, "r-", linewidth=2)
plt.xlabel("X")
plt.ylabel("y")
plt.show()
