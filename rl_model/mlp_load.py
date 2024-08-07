import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from SimpleRender import SimpleRandChartState



"""
2024-08-05
PPO로 간단히 구현한 강화 학습.

지금은 total_timesteps를
100000 번 정도 수행하면 만족스러운 결과가 나온다.

2024-08-07
일단 간단하게 기록을 쌓아서 입력데이터 화 해볼까

[Sn-5, Sn-4 ... Sn] 와 같이. (패딩을 먼저 이걸로 연습?)
Sn-m ~ Sn-1 까지는 act에 대한 정보도 포함시키고  

이렇게 하면 학습 횟수를 줄여도 좀 잘 학습되지 않을까??


"""

# 사용자 정의 환경

model_list = []

for i in range(1):
    total_time_steps = 1000 * i * 100

    path = f'ppo_result/money_block_{i}k.h5'

    # 모델 로드
    model = PPO.load(path)

    model_list.append(model)

time_step = 100


total_rate = 0
for ep in range(30):
    env = DummyVecEnv([lambda: SimpleRandChartState()])
    n_action = np.array([1.0]).reshape(1, 1)
    for i in range(time_step):
        env.step(n_action)
    total_money = env.envs[0].get_total_money()
    start_money = env.envs[0].start_money
    rate = (total_money - start_money) / start_money
    print(f"{total_money} ({rate:.2f}%)")
    total_rate += rate

print(f"all in total rate {total_rate:.3f}")

min_rate = 1.0
max_rate = -1.0

min_env = None
max_env = None

for m in range(len(model_list)):
    print(m, end=" : ")
    model = model_list[m]
    total_rate = 0
    for ep in range(30):
        env = DummyVecEnv([lambda: SimpleRandChartState()])
        obs = env.reset()
        for i in range(time_step):
            action, _ = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
        total_money = env.envs[0].get_total_money()
        start_money = env.envs[0].start_money
        rate = (total_money - start_money) / start_money
        print(f"({rate:.2f}%)", end=" ")

        if rate < min_rate:
            min_env = env.envs[0]
        elif max_rate < rate:
            max_env = env.envs[0]

        total_rate += rate
    print(f"\ntotal rate {total_rate:.3f}")

env.envs[0]