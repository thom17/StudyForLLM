from chips import Chips
from colorama import Fore, Style, init
init(autoreset=True)

import random as rand

from rl_model.env_state import EnvState

class Player:
    def __init__(self, hands: [int], game_env):
        self.hands: [int] = hands
        self.place_chip: [int] = []
        self.env =game_env

        self.rand_play = True
    def make_state(self)->EnvState:
        data_dict = {}
        data_dict['hand'] = (self.hands, Chips.get_normalize_chips(self.hands))
        data_dict['place_chip'] = (self.place_chip, Chips.get_normalize_chips(self.place_chip))
        return EnvState(data_dict)

    def print_hand(self, sort= False):
        if sort:
            for num in sorted(self.hands):
                if Chips.is_red(num):
                    print(f"{Fore.RED}{num}", end=" ")
                else:
                    print(f"{Fore.BLUE}{num}", end=" ")
            print()
        else:
            Chips.print(self.hands, end="\n")


    def place_hand(self, state:EnvState):
        """
          :param state: 학습시 사용될 state 정보
          :return: num : 그냥 숫자
          """
        if self.rand_play:
            rand_idx = rand.randint(0, len(self.hands)-1)
            state.update_data('act', (rand_idx , [rand_idx/3]))
            return self.hands.pop(rand_idx)
        else:
            self.env.print_state(show_idx=0, state=state)
            idx = self._input(self.hands)
            return self.hands.pop(idx)

    def _input(self, datas: [int]):
        idx = -1
        while idx is -1:
            try:
                ip =int(input(str(datas)+"->"))
                if ip in datas:
                    idx = datas.index(ip)
            except:
                print("err input")
        return idx

    def get_place_trick(self, state:EnvState)->int:
        """
        :param state: 학습시 사용될 state 정보
        :return: idx
        """
        if self.rand_play:
            placed_chip: [int] = state.get_data('placed_chip')
            red_chip: [int] = Chips.get_red_chips(placed_chip)
            if red_chip:
                rand_idx=rand.randint(0, len(red_chip)-1)
                idx = red_chip[rand_idx][0]
                state.update_data('act', (idx, [idx / 3]))
                return idx
            else:
                placed_chip = [(idx, num) for idx, num in enumerate(placed_chip) if num]
                rand_idx = rand.randint(0, len(placed_chip) - 1)
                idx = placed_chip[rand_idx][0]
                state.update_data('act', (idx, [idx / 3]))
                return idx
        else:
            placed_chip: [int] = state.get_data('placed_chip')
            self.env.print_state(show_idx=0, state=state)
            idx = self._input(placed_chip)
            return idx




if __name__ == "__main__":
    pass