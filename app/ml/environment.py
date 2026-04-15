import numpy as np

BOARD_SIZE = 7
WIN_LENGTH = 4
EMPTY, X, O = 0, 1, 2

# Intermediate rewards
REWARD_WIN        =  1.0    # thắng
REWARD_LOSE       = -1.0    # thua (dùng khi cập nhật cho bên thua)
REWARD_DRAW       =  0.0    # hòa
REWARD_3_IN_ROW   =  0.3    # tạo được 3 quân liền (tấn công)
REWARD_BLOCK_3    =  0.2    # chặn đối thủ 3 quân liền (phòng thủ)
REWARD_2_IN_ROW   =  0.05   # tạo được 2 quân liền
REWARD_BLOCK_WIN  =  0.5    # chặn nước thắng của đối thủ (4 quân)


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

    def _count_consecutive(self, row, col, player):
        """Đếm số quân liền dài nhất qua ô (row,col) của player"""
        directions = [(0,1),(1,0),(1,1),(1,-1)]
        max_count = 1
        for dr, dc in directions:
            count = 1
            for sign in [1, -1]:
                r, c = row + sign*dr, col + sign*dc
                while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == player:
                    count += 1
                    r += sign*dr
                    c += sign*dc
            max_count = max(max_count, count)
        return max_count

    def _opponent_max_threat(self):
        """Tìm chuỗi dài nhất của đối thủ trên toàn bàn (trước khi mình đánh)"""
        opponent = O if self.current_player == X else X
        max_len = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == opponent:
                    l = self._count_consecutive(r, c, opponent)
                    max_len = max(max_len, l)
        return max_len

    def _check_result(self, row, col):
        player = self.current_player
        directions = [(0,1),(1,0),(1,1),(1,-1)]

        # Kiểm tra thắng
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
                return REWARD_WIN, True

        # Hòa
        if not self.get_valid_actions():
            return REWARD_DRAW, True

        # --- Intermediate rewards ---
        my_len = self._count_consecutive(row, col, player)

        # Tính threat của đối thủ SAU khi mình đánh
        # (nếu mình vừa chặn được chuỗi dài của đối thủ)
        opponent = O if player == X else X
        opp_max = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == opponent:
                    opp_max = max(opp_max, self._count_consecutive(r, c, opponent))

        reward = 0.0

        # Thưởng tấn công
        if my_len == 3:
            reward += REWARD_3_IN_ROW
        elif my_len == 2:
            reward += REWARD_2_IN_ROW

        # Thưởng phòng thủ: nếu trước khi đánh đối thủ có 3 quân liền
        # mà sau khi đánh chuỗi đó bị giảm → mình đã chặn
        # (đơn giản: nếu opp_max < WIN_LENGTH-1 thì thưởng block)
        # Cách đơn giản hơn: check xem ô vừa đánh có nằm giữa chuỗi đối thủ không
        # → dùng heuristic: nếu opp_max <= 2 và trước đó có thể là 3 → thưởng block
        if opp_max < WIN_LENGTH - 1:
            # Kiểm tra xem ô này có "cắt" chuỗi đối thủ không
            # bằng cách thử bỏ quân mình ra và đếm lại
            self.board[row][col] = EMPTY
            opp_before = 0
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if self.board[r][c] == opponent:
                        opp_before = max(opp_before, self._count_consecutive(r, c, opponent))
            self.board[row][col] = player  # đặt lại

            if opp_before >= WIN_LENGTH - 1:
                # Vừa chặn được chuỗi WIN_LENGTH-1 của đối thủ
                reward += REWARD_BLOCK_WIN
            elif opp_before >= 3:
                reward += REWARD_BLOCK_3

        return reward, False
