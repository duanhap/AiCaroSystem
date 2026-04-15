import numpy as np

BOARD_SIZE = 7
WIN_LENGTH = 4
EMPTY, X, O = 0, 1, 2

class CaroEnv:
    def __init__(self):
        self.board = None
        self.current_player = X
        self.done = False
        self.winner = None
        self.reset()

    def reset(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_player = X
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        """Trả về tuple Python int để dùng làm key trong Q-table và serialize JSON"""
        return tuple(int(x) for x in self.board.flatten())

    def get_valid_actions(self):
        return [i for i in range(BOARD_SIZE * BOARD_SIZE) if self.board.flatten()[i] == EMPTY]

    def step(self, action: int):
        if self.done:
            raise ValueError("Game đã kết thúc")

        row, col = divmod(action, BOARD_SIZE)
        if self.board[row][col] != EMPTY:
            raise ValueError(f"Ô {action} đã được đánh")

        self.board[row][col] = self.current_player
        reward, done = self._check_result(row, col)
        self.done = done

        if not done:
            self.current_player = O if self.current_player == X else X

        next_state = self.get_state()
        return next_state, reward, done

    def _check_result(self, row, col):
        player = self.current_player
        directions = [(0,1),(1,0),(1,1),(1,-1)]

        for dr, dc in directions:
            count = 1
            for sign in [1, -1]:
                r, c = row + sign*dr, col + sign*dc
                while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == player:
                    count += 1
                    r += sign*dr
                    c += sign*dc
            if count >= WIN_LENGTH:
                self.winner = player
                return 1.0, True  # thắng

        if not self.get_valid_actions():
            return 0.0, True  # hòa

        return 0.0, False  # chưa xong
