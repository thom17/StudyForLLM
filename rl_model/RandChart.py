"""
Reinforcement Learning
강화 학습을 위한 태스트 코드

by tuto.RandChart
여러개의 차트를 고려해서 파라미터에 따라 랜덤성이 좀 다르게 설정
"""
import matplotlib.pyplot as plt
from colorama import Fore, Style, init
init(autoreset=True)

import numpy as np
import random as rand

from env_state import EnvState


class RandChart:
    c_id = 0
    """
    간단한 모의 주식 차트??
    """
    def __init__(self, price=1000, min=-2, max=2, rotate=None):
        self.price = price
        self.blocks = {}
        self.min = min
        self.max = max
        self.rotate = rotate
        self.set_blocks()

        self.records = []
        self.id = RandChart.c_id
        RandChart.c_id += 1
        # rand.seed(0)

    def __block_to_list(self):
        return [self.blocks[key] for key in self.blocks]

    def to_list(self):
        block_list = self.__block_to_list()
        block_list.append(self.price)
        return block_list

    def __get_normalize_block(self):
        block_list = self.__block_to_list()
        return [(d+2)/(self.max-self.min) for d in block_list]

    def make_state(self):
        data_dict = {'price': (self.price, None),
                     'min': (self.min, None),
                     'max': (self.max, None),
                     'blocks': (self.blocks, self.__get_normalize_block()),
                     'rotate': (self.rotate, None),
                     'id': (self.id, None)
                     }
        normalize_order = ['blocks']
        return EnvState(data=data_dict, order=normalize_order)

    def buy(self, money) ->int:
        if money:
            return int(money/self.price)
        else:
            return 0

    def set_blocks(self):
        min = self.min
        if self.price + self.min <= 1:
            min = 0

        for i in range(2, 13):
            self.blocks[i] = rand.randint(min, self.max)

    def print_blocks(self):
        print("price : ", self.price)
        for key in self.blocks:
            print(key, end="\t")
        print()
        for key in self.blocks:
            if 0 <= self.blocks[key]:
                print(f"{Fore.BLUE}+{self.blocks[key]}", end="\t")
            else:
                print(f"{Fore.RED}{self.blocks[key]}", end="\t")
        print('\033[0m')

    def roll(self):
        d1 = rand.randint(1, 6)
        d2 = rand.randint(1, 6)
        d = d1+d2
        #print
        result = self.blocks[d]
        color = ''
        if 0 < result:
            color = Fore.BLUE
        elif result < 0:
            color = Fore.RED

        # print(f"[{d1}][{d2}] : {d} = {color}{result}")

        return result, (d1, d2)

    def update(self):
        record = {}
        record['block'] = self.blocks
        record['price'] = self.price

        r, dice = self.roll()
        self.price = self.price + r

        if self.rotate is None and dice[0] == dice[1]:
            self.set_blocks()
        elif self.rotate and len(self.records) % self.rotate == 0:
            self.set_blocks()

        record['roll'] = r
        record['next_block'] = self.blocks
        record['next_price'] = self.price

        self.records.append(record)

        if self.price <= 1:
            # print("Rand Chart price 0 !!!")
            self.price = 1

    def make_plot(self):
        x = [i for i in range(len(self.records))]
        y = [record['price'] for record in self.records]

        plt.plot(x, y)
        plt.xlabel("time")
        plt.ylabel("price")
        plt.show()


if __name__ == "__main__":
    chart = RandChart()

    for i in range(500):
        chart.update()
    chart.make_plot()

        # chart.print_blocks()
        # r, dice=chart.roll()
        # chart.price += r
        # print()
        # chart.print_blocks()