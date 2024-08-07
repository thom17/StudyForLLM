import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from SimpleRender_stack_ver import SimpleRandChartState

# 사용자 정의 환경


#학습 시간별 모델 저장
for i in range(1, 12):
    env = DummyVecEnv([lambda: SimpleRandChartState()])

    total_time_steps = 1000 * i
    if i == 11:
        total_time_steps *= 10

    # 모델 초기화
    model = PPO('MlpPolicy', env, verbose=0)

    # 모델 학습
    model.learn(total_timesteps=total_time_steps)

    model.save(f'ppo_result/stack/{int(total_time_steps/1000)}k.h5')


time_step = 10000

#동일한 환경에서 결과를 보기위헤
import random as rand

total_rate = 0
rand.seed(0)
for ep in range(10):
    env = DummyVecEnv([lambda: SimpleRandChartState()])
    obs = env.reset()
    n_action = np.array([1.0]).reshape(1, 1)
    for i in range(time_step):
        env.step(n_action)
    total_money = env.envs[0].get_total_money()
    start_money = env.envs[0].start_money
    rate = (total_money - start_money) / start_money
    print(f"{total_money} ({rate:.2f}%)")
    total_rate += rate

print(f"all in total rate {total_rate:.3f}")

total_rate = 0
rand.seed(0)
for ep in range(10):
    env = DummyVecEnv([lambda: SimpleRandChartState()])
    obs = env.reset()
    for i in range(time_step):
        action, _ = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
    total_money = env.envs[0].get_total_money()
    start_money = env.envs[0].start_money
    rate = (total_money - start_money) / start_money
    print(f"{total_money} ({rate:.2f}%)")
    total_rate += rate
print(f"model total rate {total_rate:.3f}")
