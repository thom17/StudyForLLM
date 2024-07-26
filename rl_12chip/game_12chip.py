"""
12chip 포커로 태스트
"""
import random as rand
from collections import defaultdict

from chips import Chips
from player import Player

from rl_model.env_state import EnvState

class Game12Tricks:
    def __init__(self, player_num=3):
        self.player_num = 3
        self.chips = Chips(player_num=player_num)
        self.players: [Player] =[]
        self.trick = 0 #사실 상 차례 인덱스

        self.placed_chip = [0 for _ in range(player_num)]
        self.is_placed_step = True

        self.record_map = defaultdict(list)
        self.match_count = 0
        for hands in self.chips.deal():
            self.players.append(Player(hands, self))

        self.cur_state = None

    def make_state(self) ->EnvState:
        data_dict = {}
        data_dict['placed_chip'] = (self.placed_chip, Chips.get_normalize_chips(self.placed_chip))
        data_dict['is_placed_step'] = (self.is_placed_step, [self.is_placed_step])
        data_dict['trick'] = (self.trick, [self.trick/(self.player_num-1)])
        for idx, player in enumerate(self.players):
            data_dict[f'player_{idx}'] = (player.make_state(), None)

        return EnvState(data_dict)

    def print_state(self, state:EnvState = None, show_idx = 0):
        if state is None:
            state:EnvState = self.cur_state


        # print(end='hand: ')
        for idx in range(self.player_num):
            player_state:EnvState = state.get_data(f'player_{idx}')
            if show_idx is None or show_idx == idx:
                Chips.print(chips=player_state.get_data('hand'), sort=True, end=":")
            else:
                Chips.print(chips=player_state.get_data('hand'), sort=True, end=":", only_color=True)
            Chips.print(player_state.get_data('place_chip'), end="\t")

        print()
        # # print(end='place_chip: ')
        # for idx in range(self.player_num):
        #     player_state: EnvState = state.get_data(f'player_{idx}')
        #     Chips.print(player_state.get_data('place_chip'), end="\t\t")
        #
        #
        # print()


        # print(f"trick : {state.get_data('trick')}")
        # print(f"act : {state.get_data('act')}")
        # print(f"is_placed_step : {state.get_data('is_placed_step')}")
        # print(state.get_data('placed_chip'))

        print('placed_chip:',end="")
        Chips.print(state.get_data('placed_chip'), end="\n")

        print("=====" * 5)
        # print("=====" * 5)



    def check_placed_step(self):
        """
        플레이/회수 단계 체크
        함수 내에서 플레이<->회수 변경
        :return:
        """



        #회수 단계 진행 중 모든 칩을 회수 하지 못한 경우
        if not self.is_placed_step and self.placed_chip.count(0) < self.player_num:
            return False

        #회수 단계 진행 중이면 플레이로 전환 플레이면 속행
        elif 0 in self.placed_chip:
            return True

        #플레이 단계 중 모든 칩을 놓음 회수단계로 전환
        else:
            self.__set_trick_player()
            return False


        #행동 직후 저장
    def __record_step(self, state):
        """
        현제 state를 복사하여 저장.
        :param state:
        :return:
        """
        self.record_map[self.match_count].append(state)
        self.print_state(state)

    def __set_trick_player(self):
        # 값을 기준으로 정렬
        max_idx, max_value = max(enumerate(self.placed_chip[:self.player_num]), key=lambda x: x[1])
        assert max_value, f"assert __get_trick_player {self.placed_chip}"

        self.trick = max_idx




    def __step_place(self):
        act_player = self.players[self.trick]

        self.placed_chip[self.trick] = act_player.place_hand(self.cur_state)
    def __step_draw(self):
        act_player = self.players[self.trick]
        place_idx = act_player.get_place_trick(self.cur_state)

        # add to hand
        if 0 in self.placed_chip:
            act_player.hands.append(self.placed_chip[place_idx])
        else:
            act_player.place_chip.append(self.placed_chip[place_idx])

        self.placed_chip[place_idx] = 0


    def step(self):
        self.is_placed_step = self.check_placed_step()
        self.cur_state = self.make_state()

        if self.check_end():
            return self.cur_state, True

        if self.is_placed_step:
            self.__step_place()
        else:
            self.__step_draw()

        #행동까지 포함하여 저장
        self.__record_step(self.cur_state)
        self.trick = (self.trick + 1) % self.player_num

        #새로운 상태 생성
        self.cur_state = self.make_state()

        is_end = self.check_end()

        return self.cur_state, is_end


    def check_end(self):
        if self.is_placed_step:
            return False

        #진행 중이면
        if self.placed_chip.count(0) < self.player_num:
            return False

        for player in self.players:
            if len(player.hands):
                continue
            else:
                return True
        return False


if __name__ == "__main__":
    game = Game12Tricks()
    game.players[0].rand_play = False
    state, finish = game.step()
    while not finish:
        state, finish = game.step()

    print(len(game.record_map[0]))

    for state in game.record_map[0]:
        print(state)