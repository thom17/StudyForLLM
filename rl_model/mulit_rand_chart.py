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

class RandChartEnv(gym.Env):
    def __init__(self, chart_list: Optional[List[RandChart]] = None, max_chart_num=12):
        super(RandChartEnv, self).__init__()

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
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.MAX_CHART_NUM,), dtype=np.float32)

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

            price = rand_idx[i] * 200 + 500
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

            if self.cash:
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
        return len(state.normalize_data) # * self.MAX_CHART_NUM #패딩을 포함했으니 이것도 제거

    def get_state_normalize(self):
        state = self.make_state()
        return state.normalize_data

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.chart_list: [RandChart] = self.__make_rand_charts()
        self.chart_num = len(self.chart_list)
        self.records = {}
        self.seed_value = [0.0 for _ in range(self.chart_num)]

        self.cash = self.start_money
        self.act_count = 0

        self.state_list = []
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
        total_money = self.cash
        for i in range(self.chart_num):
            total_money += self.chart_list[i].price * self.seed_value[i]

        return total_money

    def __is_enable_action(self, n_action_lst: [float]) -> bool:
        buy_rate = 0
        for i in range(self.MAX_CHART_NUM):
            n_action = n_action_lst[i]
            if abs(n_action) and self.chart_num <= i:
                return False
            elif 0 < n_action:
                buy_rate += n_action
            elif n_action < 0 and self.seed_value[i] == 0.0:
                continue

        return buy_rate <= 1.0

    def update_charts(self):
        for chart in self.chart_list:
            chart.update()

    def step(self, n_action_lst: [float], update=True):
        before_total_money = self.get_total_money()

        done = False
        truncated = False

        self.act_count += 1
        self.action_list.append(n_action_lst)

        if not self.__is_enable_action(n_action_lst=n_action_lst):
            if update:
                self.update_charts()

            n_state = self.get_state_normalize()

            return n_state, float(-1), done, truncated, {}


        total_dif_cash = 0
        for i in range(self.chart_num):
            n_action = n_action_lst[i]
            chart:RandChart = self.chart_list[i]

            if 0 <= n_action: #구입 & 대기
                dif_seed = int(chart.buy(n_action * self.cash))
                dif_cash = chart.price * dif_seed * -1
            else:
                dif_seed = int(self.seed_value[i] * n_action)
                dif_cash = chart.price * dif_seed #action 이 -

            self.seed_value[i] = self.seed_value[i] + dif_seed
            total_dif_cash = total_dif_cash + dif_cash


        #action은 현제 cash 에서 분배 기준이므로 한번에 변경해야함
        self.cash = self.cash + total_dif_cash

        if update:
            self.update_charts()

        normal_state = self.get_state_normalize()
        total_money = self.get_total_money()
        reward = (total_money - before_total_money) / before_total_money if before_total_money != 0 else 0

        return normal_state, float(reward), done, truncated, {}

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    chartsEnv = RandChartEnv()
    colors = plt.cm.get_cmap('viridis', chartsEnv.chart_num)
    c_idx = 0
    for i in range(200):
        for chart in chartsEnv.chart_list:
            chart.update()
    for chart in chartsEnv.chart_list:
        x = list(range(len(chart.records)))
        y = [record['price'] for record in chart.records]

        plt.plot(x, y, c=colors(c_idx))
        c_idx += 1

    plt.xlabel("time")
    plt.ylabel("price")
    plt.show()

    print(chartsEnv.chart_num, " num :")
    i = 0

    state = chartsEnv.make_state()

    print("state Size = ", chartsEnv.get_state_size())
    div = chartsEnv.get_state_size() / chartsEnv.MAX_CHART_NUM
    print("size / max chart num = ", div)

    for key in state.normalize_order:
        print(f"{key} {state.data_dict[key]}")