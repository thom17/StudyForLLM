import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn


# 신경망 모델 정의
class ConnectFourNN(nn.Module):
    def __init__(self):
        super(ConnectFourNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=2)
        # 이 값은 Conv 레이어를 통과한 후의 출력 크기 계산에 따라 설정되어야 합니다.
        self.fc1 = nn.Linear(128 * 8 * 7, 512)
        self.fc2 = nn.Linear(512, 7)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        policy = self.fc2(x)
        value = self.fc3(x)
        return policy, value
