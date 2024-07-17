"""
Reinforcement Learning
강화 학습을 위한 태스트 코드
"""

from RandChart import RandChart

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


chart = RandChart()



# 1. 환경 정의
class SimpleRandChartState:
    def __init__(self, chart: RandChart = None):
        self.record_num = 0
        self.records = {}
        self.chart =chart
        if chart is None:
            self.chart = RandChart()
        self.seed = 0.0
        self.start_money = 100000
        self.money = self.start_money
        self.act_count = 0
        self.action_list=[]



    def get_state_size(self) ->int:
        chart_state_list = self.get_state_normalize()
        return len(chart_state_list)

    def get_state_normalize(self):
        block_list = self.chart.get_normalize_block()
        return np.array(block_list)   #일단 블록만 해보자
        



    def reset(self):
        self.chart = RandChart()
        self.money = self.start_money
        self.seed = 0.0
        self.act_count=0
        self.action_list = []
        return self.get_state_normalize()

    def get_total_money(self):
        return (self.seed * self.chart.price) + self.money


    def step(self, n_action: float, update= True):
        """
        n_action : 변동되는 소유 주량? (정규화 값)
        :param action:
        :return:
        """
        # 예측된 값 action에 따른 보상과 새로운 상태 정의
        action = 0

        self.act_count+=1
        self.action_list.append(n_action)

        # 소수점 제외
        buy_money = n_action * self.money
        if 0 < n_action: #구입
            can_buy = self.chart.buy(money=buy_money)
            action = int(can_buy * n_action)
        else:
            action = int(self.seed * n_action)

        #정상적으로 동작중인지 체크를 위한 시각화
        # self.chart.print_blocks()
        # print(f"cash(seed) {self.money}({self.seed}), act : {n_action:.2f} ({action})")
        #

        self.seed += action
        self.money = self.money - (action * self.chart.price) # action 이 음수라면 주식을 판것

        if update:
            self.chart.update()

        state = self.get_state_normalize()
        total_money = self.money + (self.seed * self.chart.price)
        reward = (total_money - self.start_money)/self.start_money

        print(f"{self.act_count}({action}): {total_money}", end="\t\r")

        return state, float(reward)


class TradeChartAgent:
    def __init__(self, env = None, model = None):
        if env is None:
            env = SimpleRandChartState()

        self.env = env
        self.state_size: int = env.get_state_size()
        self.model = model
        if model is None:
            self.model = self.build_model(self.state_size)
            self.model.compile(optimizer='adam', loss='mse')


    def build_model(self, input_dim):
        model = models.Sequential()
        model.add(layers.Dense(48, input_dim=input_dim, activation='relu'))
        model.add(layers.Dense(48, activation='relu'))
        model.add(layers.Dense(1, activation='linear'))
        return model



    def get_rand_n_act(self):
        """
        소유하고있을때만 판매 가능
        -1 전체 팔기 :1 전체 구입
        :return:
        """

        r = np.random.rand()
        if 1 < self.env.seed:
            return 2 * r - 1.0
        else:
            return r

    def predict(self, env = None):
        if env is None:
            env = self.env

        state = env.get_state_normalize().reshape(1, -1)
        return self.model.predict(state)[0]
    def train(self, episodes=1000, gamma=0.05, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, act_size = 10):
        """
        :param episodes: 에포치 (시나리오 수행 수)
        :param gamma: (1에 가까울수록 현재를 중시)
        :param epsilon: (점점 감소하는 값. 무작위로 선택할 확률)
        :param epsilon_decay: (epsilon를 감소시키는 비율)
        :param epsilon_min: (최소 epsilon 값. 이를 통해 최소한의 탐험을 수행)
        :return:
        """
        for e in range(episodes):
            self.env.reset()
            state = self.env.get_state_normalize().reshape(1, -1)
            done = False
            print(f"episodes : {e}", end="\t\r")
            while not done:
                if np.random.rand() <= epsilon:
                    n_action = self.get_rand_n_act()  # 임의의 예측 값
                else:
                    n_action = self.model.predict(state)[0]

                if n_action < 0 and self.env.seed < 1:
                    n_action = 0

                next_state, reward = self.env.step(n_action)
                next_state = np.reshape(next_state, [1, self.state_size])

                done = (act_size == self.env.act_count)

                target = reward
                if not done:
                    target = reward + gamma * self.model.predict(next_state)[0][0]

                target_f = self.model.predict(state)
                target_f[0][0] = target

                self.model.fit(state, target_f, epochs=1, verbose=0)

                state = next_state

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            sum_act = 0.0
            for record_act in self.env.action_list:
                sum_act += record_act
            avg_act = float(sum_act/20.0)
            seed_money = self.env.seed * self.env.chart.price
            print(f"Episode {e + 1}/{episodes} - Epsilon: {epsilon:.2f}, Reward: {reward:.2f} avg_act {avg_act:.2f}")
            print(f"price: {self.env.chart.price} x {self.env.seed} = {seed_money}")
            print(f"cash : {self.env.money} total {seed_money+self.env.money}")

        #모든 에피소드가 완료되면 모델을 저장
        self.model.save(f'trade_chart_model_ep{episodes}_act{act_size}.h5')

if __name__ == "__main__":
    from tensorflow.keras.models import load_model
    #
    # chart = RandChart()
    # my_env = SimpleRandChartState(chart= chart)
    # agent_env = SimpleRandChartState(chart= chart)
    #
    # loaded_model = load_model(r'trade_chart_model_ep10_act100.h5')
    # agent = TradeChartAgent(env=agent_env, model=loaded_model)
    #
    # while True:
    #     chart.print_blocks()
    #     try:
    #         n_act = float(input('your n_act (-1~1)\n->'))
    #     except:
    #         n_act = 0
    #
    #     advice = agent.predict(my_env)
    #     my_env.step(n_action= n_act, update=False)
    #
    #     agent_act = agent.predict()
    #
    #     agent_env.step(n_action = agent_act, update=False)
    #     print(f"agent act {agent_act}, advice {advice}")
    #
    #     chart.update()
    #
    #     my_total = my_env.get_total_money()
    #     agent_total = agent_env.get_total_money()
    #     start_money = my_env.start_money
    #
    #     print(f"seed {my_env.seed}/{agent_env.seed}({my_env.seed-agent_env.seed})")
    #     print(f"cash {my_env.money}/{agent_env.money}({my_env.money - agent_env.money})")
    #     print(f"total {my_total}/{agent_total}({my_total-agent_total})" )
    #     print(f"rate {(my_total-start_money)/start_money:.2f} {(agent_total-start_money)/start_money:.2f}")



    #4. 환경 및 에이전트 초기화
    agent = TradeChartAgent()

    #5. 모델 학습
    agent.train(episodes=1000, act_size=500)

