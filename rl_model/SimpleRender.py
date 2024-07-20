from typing import Optional, Union, List

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.core import RenderFrame

from RandChart import RandChart

class SimpleRandChartState(gym.Env):
    def __init__(self, chart=None):
        super(SimpleRandChartState, self).__init__()
        self.record_num = 0
        self.records = {}
        self.chart = chart if chart is not None else RandChart()
        self.seed_value = 0.0
        self.start_money = 100000
        self.money = self.start_money
        self.act_count = 0
        self.action_list = []

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.get_state_size(),), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def get_state_size(self) -> int:
        chart_state_list = self.get_state_normalize()
        return len(chart_state_list)

    def get_state_normalize(self):
        block_list = self.chart.get_normalize_block()
        return np.array(block_list)  # 일단 블록만 해보자

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)
        self.chart = RandChart()
        self.money = self.start_money
        self.seed_value = 0.0
        self.act_count = 0
        self.action_list = []
        return self.get_state_normalize(), {}

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        sum_act = 0.0
        for record_act in self.action_list:
            sum_act += record_act
        avg_act = float(sum_act / len(self.action_list))
        seed_money = self.seed_value * self.chart.price

        print(f"price: {self.chart.price} x {self.seed_value} = {seed_money}")
        print(f"cash : {self.money} total {seed_money + self.money}")
        print(f"rate : {(seed_money + self.money - self.start_money) /self.start_money }")

    def get_total_money(self):
        return (self.seed_value * self.chart.price) + self.money

    def step(self, n_action: float, update=True):
        action = 0

        self.act_count += 1
        self.action_list.append(n_action)

        # 소수점 제외
        buy_money = n_action * self.money
        if 0 < n_action:  # 구입
            can_buy = self.chart.buy(money=buy_money)
            action = int(can_buy * n_action)
        else:
            action = int(self.seed_value * n_action)

        self.seed_value += action
        self.money = self.money - (action * self.chart.price)  # action 이 음수라면 주식을 판것

        before_total_money = self.get_total_money()

        if update:
            self.chart.update()

        state = self.get_state_normalize()
        total_money = self.get_total_money()
        reward = (total_money - before_total_money) / before_total_money
        print(f"{self.act_count}({action}): {total_money}", end="\t\r")

        done = False
        truncated = False

        # # 정상적으로 동작중인지 체크를 위한 시각화
        # self.chart.print_blocks()
        # print(f"cash(seed) {int(self.money)}({float(self.seed_value):.2f}), act : {float(n_action):.2f} ({float(action):.2f})")
        # print("total ",total_money)
        # print(f"{self.act_count} reward {float(reward):.2f}, total reward {((total_money-self.start_money)/self.start_money):.2f}")
        # #

        return state, float(reward), done, truncated, {}

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
