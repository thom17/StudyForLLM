import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from mulit_rand_chart import RandChartEnv
import time

# 사용자 정의 환경
env = DummyVecEnv([lambda: RandChartEnv()])

# 모델 초기화
model = PPO('MlpPolicy', env, verbose=0)

start = time.time()
total_timesteps= 1000000 #100000

# 모델 학습
model.learn(total_timesteps=total_timesteps)
end = time.time()

print("learn done")
print(f"{total_timesteps} time {(end-start):.2f}(s)  stemp/time = {(total_timesteps/(end-start)):.2f}")

time_step = 100


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

    #평균 act 계산
    avg_act = [0.0 for i in range(env.envs[0].chart_num)]
    for record in env.envs[0].records[0]:
        for i in range(env.envs[0].chart_num):
            avg_act[i] += record.get_data('act')[i]
    avg_act = np.array(avg_act)/env.envs[0].chart_num

    print(f"{totla_money} ({rate:.2f}%) total act {avg_act}" )
    total_rate += rate
print(f"model total rate {total_rate:.3f}")

