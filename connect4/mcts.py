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
            def uct_value(node):
                if node.visits == 0:
                    return float('inf')
                return node.value / node.visits + np.sqrt(2 * np.log(node.parent.visits) / node.visits)

            node = max(node.children.values(), key=uct_value)
            _, _, done, _ = sim_env.step(np.argmax(node.state))

        # Expansion
        new_state = node.state  # Initialize new_state with the current node's state
        if not done:
            actions = sim_env.available_actions()
            for action in actions:
                temp_state = np.copy(node.state)
                row = np.argmax(temp_state[:, action] == 0)
                temp_state[row, action] = sim_env.current_player
                if action not in node.children:
                    node.children[action] = MCTSNode(temp_state, parent=node)
            # Use the first action to initialize new_state for simulation
            if actions:
                row = np.argmax(node.state[:, actions[0]] == 0)
                new_state = np.copy(node.state)
                new_state[row, actions[0]] = sim_env.current_player

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