import torch
import numpy as np

from connect4env import ConnectFourEnv
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0

def mcts_search(env, model, state, n_simulations=100):
    root = MCTSNode(state)

    for _ in range(n_simulations):
        node = root
        sim_env = ConnectFourEnv()
        sim_env.board = np.copy(state)
        sim_env.current_player = env.current_player
        done = False

        # Selection
        while node.children and not done:
            node = max(node.children.values(), key=lambda n: n.value / n.visits + np.sqrt(2 * np.log(node.parent.visits) / n.visits))
            _, _, done, _ = sim_env.step(np.argmax(node.state))

        # Expansion
        if not done:
            actions = sim_env.available_actions()
            for action in actions:
                new_state = np.copy(node.state)
                row = np.argmax(new_state[:, action] == 0)
                new_state[row, action] = sim_env.current_player
                if action not in node.children:
                    node.children[action] = MCTSNode(new_state, parent=node)

        # Simulation
        new_state_tensor = torch.tensor(new_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        _, value = model(new_state_tensor)
        value = value.item()

        # Backpropagation
        while node:
            node.visits += 1
            node.value += value
            node = node.parent

    return max(root.children.items(), key=lambda item: item[1].value / item[1].visits)[0]