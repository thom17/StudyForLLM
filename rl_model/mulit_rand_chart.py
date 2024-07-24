from typing import Optional, Union, List

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.core import RenderFrame

from RandChart import RandChart

import random as rand
from env_state import EnvState

"""
다중 차트 테스트
"""

class RandChartState(gym.Env):
    def __init__(self, chart_list: Optional[List[RandChart]] = None, max_chart_num=12):
        super(RandChartState, self).__init__()

        self.MAX_CHART_NUM = max_chart_num

        if chart_list is None:
            chart_list = self.__make_rand_charts()

        self.chart_list: List[RandChart] = chart_list
        self.chart_num = len(chart_list)
        self.records = {}
        self.seed_value = [0.0 for _ in range(self.chart_num)]
        self.start_money = 100000
        self.cash = self.start_money
        self.act_count = 0

        self.state_list = []
        self.action_list = []

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.get_state_size(),), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.get_state_size(),), dtype=np.float32)

    def __make_rand_charts(self):
        chart_list = []

        r = rand.randint(3, self.MAX_CHART_NUM)
        rand_idx = list(range(r))
        rand.shuffle(rand_idx)
        for i in range(r):
            rotate = None
            if 1 < i:
                rotate = i ** 2
            min_val = rand.randint(-4, -1)
            max_val = rand.randint(1, 4)

            price = rand_idx[i] * 100 + 200
            chart = RandChart(price=price, min=min_val, max=max_val, rotate=rotate)
            chart_list.append(chart)
        return chart_list

    def make_state(self) -> EnvState:
        state_dict = {}
        this_env_normalize_order = []  # ['pos_encode']
        chart_state_size = 0
        for i in range(self.chart_num):
            chart = self.chart_list[i]
            chart_state = chart.make_state()

            seed = self.seed_value[i]
            chart_state.update_data('seed', (seed, [seed]))  # To do: 추가적인 정규화 처리

            price_rate = chart.price / self.cash
            if 1 < price_rate:
                price_rate = -1.0

            chart_state.update_data('price', (chart.price, [price_rate]))

            chart_state_normalize_order = ['seed', 'price'] + chart_state.normalize_order
            chart_state.set_normalize_order(chart_state_normalize_order)

            state_dict[f'chart_{i}'] = (chart_state, chart_state.normalize_data)
            this_env_normalize_order.append(f'chart_{i}')
        chart_state_size = len(chart_state.normalize_data)
        for i in range(self.chart_num, self.MAX_CHART_NUM):
            n_padding = [0.0 for _ in range(chart_state_size)]
            state_dict[f'chart_{i}'] = (None, n_padding)
            this_env_normalize_order.append(f'chart_{i}')
        return EnvState(state_dict, this_env_normalize_order)

    def get_state_size(self) -> int:
        state = self.make_state()
        return len(state.normalize_data) * self.MAX_CHART_NUM

    def get_state_normalize(self):
        state = self.make_state()
        return state.normalize_data

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.chart = RandChart()
        self.money = self.start_money
        self.seed_value = [0.0 for _ in range(self.chart_num)]
        self.act_count = 0
        self.action_list = []
        state = self.get_state_normalize()
        return state, {}

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        sum_act = sum(self.action_list)
        avg_act = float(sum_act / len(self.action_list)) if self.action_list else 0
        seed_money = self.seed_value[0] * self.chart.price

        print(f"price: {self.chart.price} x {self.seed_value[0]} = {seed_money}")
        print(f"cash : {self.money} total {seed_money + self.money}")
        print(f"rate : {(seed_money + self.money - self.start_money) / self.start_money }")

    def get_total_money(self):
        return (self.seed_value[0] * self.chart.price) + self.money

    def step(self, n_action: float, update=True):
        action = 0

        self.act_count += 1
        self.action_list.append(n_action)

        buy_money = n_action * self.money
        if 0 < n_action:  # 구입
            can_buy = self.chart.buy(money=buy_money)
            action = int(can_buy * n_action)
        else:
            action = int(self.seed_value[0] * n_action)

        self.seed_value[0] += action
        self.money -= (action * self.chart.price)

        before_total_money = self.get_total_money()

        if update:
            self.chart.update()

        state = self.get_state_normalize()
        total_money = self.get_total_money()
        reward = (total_money - before_total_money) / before_total_money if before_total_money != 0 else 0
        print(f"{self.act_count}({action}): {total_money}", end="\t\r")

        done = False
        truncated = False

        return state, float(reward), done, truncated, {}

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    chartState = RandChartState()
    colors = plt.cm.get_cmap('viridis', chartState.chart_num)
    c_idx = 0
    for i in range(200):
        for chart in chartState.chart_list:
            chart.update()
    for chart in chartState.chart_list:
        x = list(range(len(chart.records)))
        y = [record['price'] for record in chart.records]

        plt.plot(x, y, c=colors(c_idx))
        c_idx += 1

    plt.xlabel("time")
    plt.ylabel("price")
    plt.show()

    print(chartState.chart_num, " num :")
    i = 0
    n_data = chartState.get_state_normalize()
    div = chartState.get_state_size() / chartState.MAX_CHART_NUM

    for data in n_data:
        print(data, end="\t")
        i += 1
        if i % div == 0:
            print()
