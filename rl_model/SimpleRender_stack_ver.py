"""
입력 데이터가 커져서 그런지 학습이 잘 안되는 느낌인듯.
물론 학습 시간을 늘리니 결과는 잘나오는대.. 흠...

"""
from typing import Optional, Union, List

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.core import RenderFrame

from RandChart import RandChart
from env_state import EnvState

class SimpleRandChartState(gym.Env):
    def __init__(self, chart=None, STACK_SIZE = 5):
        super(SimpleRandChartState, self).__init__()
        self.records = []
        self.chart = chart if chart is not None else RandChart()
        self.seed_value = 0.0
        self.start_money = 100000
        self.money = self.start_money
        self.act_count = 0
        self.action_list = []

        self.STACK_SIZE = STACK_SIZE #이전 데이터들을 들여다보는 크기
        self.cur_step_state: EnvState = self.__make_cur_state()

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.get_state_size(),), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)





    def get_state_size(self) -> int:
        one_step_env_size = self.__get_cur_state_size()
        act_reward_size = 2

        #과거 이력은 STACK_SIZE 만큼 현제 이력에는 act, reward 없음
        return (one_step_env_size + act_reward_size) * self.STACK_SIZE + one_step_env_size


    def __get_cur_state_size(self) -> int:
        if self.cur_step_state:
            cur_state = self.cur_step_state
        else:
            cur_state = self.__make_cur_state()

        return len(cur_state.normalize_data)


    def __make_cur_state(self)->EnvState:
        chart_state_list = self.chart.make_state()
        seed_money = self.seed_value * self.chart.price
        total_money = seed_money + self.money
        seed_rate = seed_money/total_money

        total_dif = (total_money - self.start_money)/self.start_money

        chart_state_list.update_data('total_money', (total_money, [total_dif]) )

        chart_state_list.update_data('seed_value', (self.seed_value, [seed_rate]))

        normalize_order = ['total_money', 'seed_value','blocks' ]
        chart_state_list.set_normalize_order(normalize_order)
        return chart_state_list

    def get_state_normalize(self):
        # # 일단 블록만 해보자
        # state = self.chart.make_state()
        # return np.array(state.normalize_data)

        normalize_list:[float] = self.make_record_state() + self.cur_step_state.normalize_data
        # print(normalize_list)
        return np.array(normalize_list)



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

    #과거 기록(STACK_SIZE) 만큼의 정규화 데이터를 생산
    def make_record_state(self) -> list[float]:
        records_num = len(self.records)
        padding_size = self.STACK_SIZE - records_num

        #이전 정보가 포함된 정규화 데이터
        stack_normalize_data = []

        if 0 < padding_size:
            for i in range(padding_size):
                one_record_size = self.__get_cur_state_size() + 2 #state + action + reward
                stack_normalize_data = stack_normalize_data + [0 for _ in range(one_record_size)]


        #가장 최근 데이터 기준으로 이전 기록을 데이터화
        for record in self.records[-self.STACK_SIZE:]:
            record:EnvState

            # 하나의 스탭에 대한 정규화 데이터에 Action과 reward 정보를 추가한다.
            stack_order = record.normalize_order + ['action', 'reward']
            n_data = record.normalize(stack_order)
            # n_data = record.normalize_data
            # record_action = record.get_data('action', normalize=True)
            # record_reward = record.get_data('reward', normalize=True)
            # n_data = n_data + record_action + record_reward
            #
            # record.normalize()
            stack_normalize_data = stack_normalize_data + n_data

        return stack_normalize_data



    def step(self, n_action: [float], update=True):
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
        # print(f"{self.act_count}({action}): {total_money}", end="\t\r")

        next_state = self.__make_cur_state()

        self.cur_step_state.update_data('action', (action, n_action))
        self.cur_step_state.update_data('reward' , (reward, [reward]))
        self.cur_step_state.update_data('next_state', (next_state, None))
        self.records.append(self.cur_step_state)

        self.cur_step_state = next_state

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
