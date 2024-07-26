from connect4env import ConnectFourEnv
from mcts import mcts_search
from conect4nn import  ConnectFourNN

import torch
import torch.optim as optim


# AlphaZero 학습 루프
def train_alpha_zero(env, model, n_games=1000):
    for game in range(n_games):
        state = env.reset()
        done = False
        while not done:
            action = mcts_search(env, model, state)
            next_state, reward, done, _ = env.step(action)
            state = next_state

            optimizer.zero_grad()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            policy, value = model(state_tensor)
            loss = (reward - value) ** 2
            loss.backward()
            optimizer.step()

            env.render()

model = ConnectFourNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

env = ConnectFourEnv()
train_alpha_zero(env, model)

