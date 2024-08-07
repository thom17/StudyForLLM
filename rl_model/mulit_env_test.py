import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from mulit_rand_chart import RandChartEnv
from CustomPolicy import CustomPolicy

import time

# 사용자 정의 환경
num_envs = 4000
env = DummyVecEnv([lambda: RandChartEnv() for _ in range(num_envs)])

# 모델 초기화 (GPU 사용)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("use device ", device)
# model = PPO('MlpPolicy', env, verbose=0)
model = PPO(CustomPolicy, env, verbose=0, device=device, policy_kwargs={'env': env})

total_timesteps = 1000000
print("start learn")

# 모델 학습
start = time.time()
model.learn(total_timesteps=total_timesteps)
end = time.time()

print("learn done")
print(f"{total_timesteps} time {(end-start):.2f}(s)  stemp/time = {(total_timesteps/(end-start)):.2f}")
for e in range(len(env.envs)):
    print(f"{e} env : {len(env.envs[e].records)} ep, {len(env.envs[0].records[1])} size")



print(len(env.envs))

time_step = 100

total_rate = 0
for ep in range(10):
    env = DummyVecEnv([lambda: RandChartEnv()])
    n_action = np.zeros((1, 12))
    for a in range(env.envs[0].chart_num):
        n_action[0][a] = 1.0 / (1 + env.envs[0].chart_num)

    for i in range(time_step):
        env.step(n_action)
    totla_money = env.envs[0].get_total_money()
    start_money = env.envs[0].start_money
    rate = (totla_money - start_money) / start_money
    print(f"{totla_money} ({rate:.2f}%)" )
    total_rate += rate

print(f"all in total rate {total_rate:.3f}")

total_rate = 0
# 평균 act 계산

for ep in range(10):
    env = DummyVecEnv([lambda: RandChartEnv()])
    avg_act = [0.0 for i in range(env.envs[0].chart_num)]
    obs = env.reset()


    for i in range(time_step):
        action, _ = model.predict(obs)
        obs, rewards, dones, info = env.step(action)


    totla_money = env.envs[0].get_total_money()
    start_money = env.envs[0].start_money
    rate = (totla_money - start_money) / start_money
    total = np.zeros(env.envs[0].MAX_CHART_NUM)
    for record_state in env.envs[0].records[1]:
        data = record_state.get_data('act')
        total = total +  np.array(data).reshape(env.envs[0].MAX_CHART_NUM)

    print(f"{totla_money} ({rate:.2f}%) total act {total}" )
    total_rate += rate
print(f"model total rate {total_rate:.3f}")
