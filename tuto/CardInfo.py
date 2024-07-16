"""
바지도 학습 데이터를 위해 생성
"""
import random as rand

import numpy


class CardInfo:
    """
    마땅한 아이디어가 떠오르지 않아서..
    """
    def __init__(self):
        r = rand.randint(0, 2)
        if r ==0:
            self.typeA()
        elif r==1:
            self.typeB()
        elif r==2:
            self.typeC()
    def typeA(self):
        self.type = 'a'
        self.cost = rand.randint(2, 8)
        d = rand.randint(1, self.cost+2)
        d2 = self.cost+2 - d

        if d==d2:
            d-=1
            d2+=1

        if d < d2:
            self.x = d2
            self.y = d
        else:
            self.x = d
            self.y = d2

    def typeB(self):
        self.type = 'b'
        self.cost = rand.randint(2, 8)
        d = rand.randint(1, self.cost + 2)
        d2 = self.cost + 2 - d

        if d==d2:
            d+=1
            d2-=1

        if d2 < d:
            self.x = d2
            self.y = d
        else:
            self.x = d
            self.y = d2

    def typeC(self):
        self.type = 'c'
        self.cost = rand.randint(2, 8)
        self.x = int(self.cost/2) + 1
        self.y = int(self.cost / 2) + 1

    def to_list(self):
        if self.type == 'a':
            n_type = 0
        elif self.type =='b':
            n_type = 1
        elif self.type =='c':
            n_type = 2
        else:
            n_type = 3

        return [n_type, self.cost, self.x, self.y]

class Deck:
    def __init__(self, size):
        data_list = []
        card_list = []
        type_map = defaultdict(list)
        cost_map = defaultdict(list)
        for i in range(size):
            c = CardInfo()
            c.id = i
            card_list.append(c)
            type_map[c.type].append(c)
            cost_map[c.cost].append(c)
            data_list.append(c.to_list())

        self.data_list = data_list
        self.card_list = card_list
        self.type_map = type_map
        self.cost_map = cost_map

    def draw_type(self):
        import matplotlib.pyplot as plt
        colors=['r', 'g', 'b']
        i = 0
        for cards in [self.type_map[type_name] for type_name in self.type_map]:
            x = [card.x for card in cards]
            y = [card.y for card in cards]

            color = colors[i]
            i+=1

            plt.scatter(x, y, c=color, marker='o', edgecolor='k', s=50)
        plt.title('Type Map Ans Data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def draw_type(self, nd_datas: numpy.ndarray):
        """
        실제 학습? 에 사용된 데이터를(ndarray타입의 데이터) 시각화 하려고 만듬.
        문제는 draw_type 함수는 개수가 너무 많으면 오래걸림
        :param nd_datas:
        :return:
        """
        import matplotlib.pyplot as plt
        colors = ['r', 'g', 'b']
        for data in nd_datas:
            x = data[2]
            y = data[3]
            color = colors[data[0]]
            plt.scatter(x, y, c=color, marker='o', edgecolor='k', s=50)

        plt.title('Type Map Ans Data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig("draw_type_xy.png")
        plt.show()

    def get_numpy_array(self):
        """
        일단 numpy 형태로 모든 데이터를 다 뱉어낸후
        해당 데이터를 재조립하든가 하는 형태로
        :return:
        """
        data_list = []
        for card in self.card_list:
            data_list.append(card.to_list())
        return np.array(data_list)

    def confusion_matrix(self, predict_labels:list, true_labels:list ):
        """
        x축 예측 label
        y축 실제 label
        label 만 출력하여 개수를 시각화
        :param predict_labels:
        :param true_labels:
        :return 정확도 점수:
        """

        from sklearn.metrics import confusion_matrix, accuracy_score
        conf_mat = confusion_matrix(true_labels, predict_labels)
        score = accuracy_score(true_labels, predict_labels)

        import seaborn as sns
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix score({score})')
        plt.xlabel('Cluster Label')
        plt.ylabel('True Label')
        plt.savefig("Confusion.png")
        plt.show()

        return score

    def do_tsne(self):
        """
        tsne를 사용하여 데이터를 2차원으로
        get_numpy_array 로 원본 데이터를 ndarray화 하고
        복사하여 정규화, 필터링 처리
        그리고 2d로 차원 축소한 결과와 원본을 반환
        :return: convert2d_xy, org_datas
        """
        from sklearn.preprocessing import StandardScaler
        org_datas = self.get_numpy_array()
        scaler = StandardScaler()

        filter_datas = np.delete(org_datas, 0, axis=1)
        n_datas = scaler.fit_transform(filter_datas)

        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, init='pca', learning_rate='auto', n_iter=1000, random_state=42)
        tsne_xy = tsne.fit_transform(n_datas)

        # 결과 시각화 (t-SNE)
        plt.scatter(tsne_xy[:, 0], tsne_xy[:, 1], marker='o', edgecolor='k', s=50)
        plt.title('t-SNE Result')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.savefig("t-sne.png")
        plt.show()

        return tsne_xy, org_datas


import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from collections import defaultdict



dect = Deck(1000)
tsne_result, org_datas = dect.do_tsne()

#t-SNE의 결과를 다시 시각화
from sklearn.cluster import KMeans

# K-means 클러스터링 적용
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(tsne_result)

ans_label = [data[0] for data in org_datas]
dect.confusion_matrix(true_labels=ans_label, predict_labels=labels)
dect.draw_type(org_datas)

# 클러스터링 결과 시각화
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('t-SNE Reduced Data with K-means Clustering')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig("cluster.png")
plt.show()


# 원본 데이터로 시각화
colors = ['r', 'g', 'b']
type_color = [colors[d[0]] for d in org_datas]
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=type_color, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('print type color')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig("type.png")
plt.show()
