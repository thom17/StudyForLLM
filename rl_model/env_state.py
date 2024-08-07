"""
"""
import numpy as np


class EnvState:
    def __init__(self, data: [str, (list, list) ], order: [str] = []):
        """
        환경 클래스에서 상태화할 변수를 입력받아 EnvState를 생성
        :data {str 변수명 , tuple(데이터 값, 정규화 값) }

        필요에 따라 파생 데이터를 입력. (라벨링, 알고리즘 계산 값 등?)
        이 파생데이터도 data_dict 에 넣을지 별도의 변수에 넣을지..

        order [ str 변수명]
        실질적으로 학습 등에 사용할 정규화 변수들의 순서.
        data_dict의 두번쨰 요소에서 값을 꺼내 조합하여 하나의 정규화 데이터를 만든다.

        :param data:
        """

        assert isinstance(data, dict), f"state init(data), {type(data)} != dict"

        self.data_dict = {}
        self.normalize_order = []

        for key in data:
            self.update_data(key, data[key])


        self.set_normalize_order(order)

    def get_data(self, key, normalize = False):
        if key in self.data_dict:
            data_pair = self.data_dict[key]
            if normalize:
                return data_pair[1]
            else:
                return data_pair[0]



    def set_normalize_order(self, order):
        assert isinstance(order, list), f"order info must list[str] , {type(order)}"


        self.normalize_order = order
        self.normalize_data = self.normalize(self.normalize_order)
    def update_data(self, key: str, data_pair: (list, list)):
        """
        환경 상태에 변수를 추가/수정
        정규화에 포함된 키가 업데이트 될경우 다시 정규화
        :param key:
        :param data_pair:
        :return:
        """


        assert isinstance(data_pair, tuple), f"data is not tuple {type(data_pair)}"

        org_data = data_pair[0] #뭐드 상관 x
        normalize_data = data_pair[1]

        assert normalize_data is None or all(type(item) != list for item in normalize_data), f"normalize data must to be simple list or None {normalize_data}"

        self.data_dict[key] = data_pair
        if key in self.normalize_order:
            self.normalize_data = self.normalize(self.normalize_order)
            return True
        else:
            return False

    def normalize(self, order):
        try:
            data_list = []
            for key in order:
                normal_datas = self.data_dict[key][1]  #ndarray[1,-1]
                if normal_datas:
                    for n in normal_datas:
                        data_list.append(n)

        except:
            print(f"{key} try execpt {normal_datas}")

        return data_list
        # return np.array(data_list).reshape(1, -1) , #pos_encode 추가

    def calculate_positional_encoding(self, order, d_model):
        length = len(self.normalize(order).reshape(-1, 1))


        position = np.arange(length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pos_encoding = np.zeros((length, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        return pos_encoding


    def get_positional_encoding(self, position, d_model):
        def get_angle(pos, i, d_model):
            return pos / np.power(10000, (2 * (i // 2)) / d_model)

        encoding = np.zeros(d_model)
        for i in range(d_model):
            if i % 2 == 0:
                encoding[i] = np.sin(get_angle(position, i, d_model))
            else:
                encoding[i] = np.cos(get_angle(position, i, d_model))

        return encoding

# class Environment:
#     def __init__(self, normalize_fun = None):
#         self.state_list = []
#
#     def
