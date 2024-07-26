import numpy as np


class ConnectFourEnv:
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.current_player = 1
        return self.board

    def step(self, action):
        row = np.argmax(self.board[:, action] == 0)
        self.board[row, action] = self.current_player
        reward, done = self.check_winner()
        self.current_player = 3 - self.current_player  # Switch player
        return self.board, reward, done, {}

    def check_winner(self):
        # Check horizontal, vertical, and diagonal for a win
        for c in range(7 - 3):
            for r in range(6):
                if self.board[r, c] == self.current_player and \
                        self.board[r, c + 1] == self.current_player and \
                        self.board[r, c + 2] == self.current_player and \
                        self.board[r, c + 3] == self.current_player:
                    return 1, True
        for c in range(7):
            for r in range(6 - 3):
                if self.board[r, c] == self.current_player and \
                        self.board[r + 1, c] == self.current_player and \
                        self.board[r + 2, c] == self.current_player and \
                        self.board[r + 3, c] == self.current_player:
                    return 1, True
        for c in range(7 - 3):
            for r in range(6 - 3):
                if self.board[r, c] == self.current_player and \
                        self.board[r + 1, c + 1] == self.current_player and \
                        self.board[r + 2, c + 2] == self.current_player and \
                        self.board[r + 3, c + 3] == self.current_player:
                    return 1, True
        for c in range(7 - 3):
            for r in range(3, 6):
                if self.board[r, c] == self.current_player and \
                        self.board[r - 1, c + 1] == self.current_player and \
                        self.board[r - 2, c + 2] == self.current_player and \
                        self.board[r - 3, c + 3] == self.current_player:
                    return 1, True
        if np.all(self.board != 0):
            return 0, True  # Draw
        return 0, False  # No winner yet

    def available_actions(self):
        return [c for c in range(7) if self.board[0, c] == 0]

    def render(self):
        print(self.board[::-1])
