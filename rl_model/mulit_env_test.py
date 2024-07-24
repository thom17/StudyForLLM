import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from mulit_rand_chart import RandChartEnv

# 사용자 정의 환경
env = DummyVecEnv([lambda: RandChartEnv()])

# 모델 초기화
model = PPO('MlpPolicy', env, verbose=0)

total_timesteps= 1000000 #100000

# 모델 학습
model.learn(total_timesteps=total_timesteps)
print("learn done")


time_step = 10000


total_rate = 0
for ep in range(10):
    env = DummyVecEnv([lambda: RandChartEnv()])
    n_action= np.zeros((1,12))
    for a in range(env.envs[0].chart_num):
        n_action[0][a] = 1.0/(1+env.envs[0].chart_num)

    for i in range(time_step):
        env.step(n_action)
    totla_money = env.envs[0].get_total_money()
    start_money = env.envs[0].start_money
    rate = (totla_money-start_money)/start_money
    print(f"{totla_money} ({rate:.2f}%)" )
    total_rate+=rate

print(f"all in total rate {total_rate:.3f}")


total_rate = 0
for ep in range(10):
    env = DummyVecEnv([lambda: RandChartEnv()])
    obs = env.reset()
    for i in range(time_step):
        action, _ = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
    totla_money = env.envs[0].get_total_money()
    start_money = env.envs[0].start_money
    rate = (totla_money-start_money)/start_money
    print(f"{totla_money} ({rate:.2f}%)" )
    total_rate += rate
print(f"model total rate {total_rate:.3f}")

