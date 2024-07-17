"""
Reinforcement Learning
강화 학습을 위한 태스트 코드
"""
from colorama import Fore, Style, init
init(autoreset=True)

import numpy as np
import random as rand
class RandChart:
    c_id = 0
    """
    간단한 모의 주식 차트??
    """
    def __init__(self):
        self.price = 1000
        self.blocks = {}
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

    def get_normalize_block(self):
        block_list = self.__block_to_list()
        return [(d+2)/4.0 for d in block_list]

    def buy(self, money):
        return money/self.price

    def set_blocks(self):
        for i in range(2, 13):
            self.blocks[i] = rand.randint(-2, 2)

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

        return result

    def update(self):
        record = {}
        record['block'] = self.blocks
        record['score'] = self.price

        r = self.roll()
        self.price = self.price + r
        self.set_blocks() # 새로운 블록 설정. (이거 주기도 커스텀 해보자)

        record['roll'] = r
        record['next_block'] = self.blocks
        record['next_score'] = self.price

        self.records.append(record)

if __name__ == "__main__":
    chart = RandChart()

    for i in range(30):
        chart.print_blocks()
        r=chart.roll()
        chart.price += r
        print()
        chart.print_blocks()