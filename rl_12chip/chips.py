"""
12chip 포커로 태스트
"""
import random as rand

from colorama import Fore, Style, init
init(autoreset=True)

class Chips:
    def __init__(self, player_num=3, order=None):
        if order is None:
            order = list(range(1, 13))
            rand.shuffle(order)

        self.order: [int] = order

        self.red_order: [int] = []
        self.blue_order: [int] = []
        self.player_num = player_num

        for num in order:
            if Chips.is_red(num):
                self.red_order.append(num)
            else:
                self.blue_order.append(num)

    def deal(self):
        hands = []
        for player_idx in range(self.player_num):
            idx = player_idx * 2
            hand = []
            for i in range(2):
                hand.append(self.red_order[i+idx])
                hand.append(self.blue_order[i+idx])
            hands.append(hand)
        return hands

    @staticmethod
    def get_red_chips(chips: [int])->list[tuple]:
        """
        :param chips: [num: int]
        :return: [(num: int, idx: int)]
        """
        return [(num, chip) for num, chip in enumerate(chips) if Chips.is_red(chip)]
    # def get_red_chips(cls, chips: [int]):
    #     reds = []
    #     for index, num in enumerate(chips):
    #         if Chips.is_red(num):
    #             reds.append((num, index))
    #     return reds

    @staticmethod
    def get_normalize_chips(chips: [int])->list:
        """
        :param chips: [num: int]
        :return: [n_num: int]
        """
        return [num/12 for num in chips]

    @classmethod
    def is_red(cls, num):
        return 3 < num and num < 10


    @classmethod
    def sort_chip(cls, chips):
        chips = [num for num in chips if num]
        reds = Chips.get_red_chips(chips)
        return [num for idx, num in reds] + [num for num in chips if not Chips.is_red(num)]

    @classmethod
    def print(cls, chips, only_color = False, sort = False, end=''):
        chips = [num for num in chips if num]

        if sort:
            chips = Chips.sort_chip(chips=chips)
        print('[', end='')
        for num in chips:
            if Chips.is_red(num):
                color = Fore.RED
            else:
                color = Fore.BLUE
            if only_color:
                num = 0
            print(f"{color}{num}", end=" ")
        print(']', end=end)