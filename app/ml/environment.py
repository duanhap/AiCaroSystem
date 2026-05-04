import numpy as np

BOARD_SIZE = 7
WIN_LENGTH = 4
EMPTY, X, O = 0, 1, 2

# Intermediate rewards
REWARD_WIN        =  1.0    # thắng
REWARD_LOSE       = -1.0    # thua
REWARD_DRAW       =  0.0    # hòa
REWARD_3_IN_ROW   =  0.4    # tạo được 3 quân liền (tấn công)
REWARD_BLOCK_3    =  0.5    # chặn đối thủ 3 quân liền (phòng thủ > tấn công)
REWARD_2_IN_ROW   =  0.1    # tạo được 2 quân liền
REWARD_BLOCK_WIN  =  0.95   # chặn nước thắng của đối thủ (gần = thắng)
REWARD_STEP       = -0.01   # phạt nhẹ mỗi nước đi để tránh đánh lung tung


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

    def _has_open_ends(self, row, col, player, chain_length):
        """Kiểm tra xem chuỗi quân có ít nhất 1 đầu mở không"""
        directions = [(0,1),(1,0),(1,1),(1,-1)]
        
        for dr, dc in directions:
            # Đếm chuỗi theo hướng này
            count = 1
            left_end = None
            right_end = None
            
            # Đếm về bên trái
            r, c = row - dr, col - dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == player:
                count += 1
                r -= dr
                c -= dc
            left_end = (r, c) if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE else None
            
            # Đếm về bên phải
            r, c = row + dr, col + dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == player:
                count += 1
                r += dr
                c += dc
            right_end = (r, c) if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE else None
            
            # Nếu chuỗi này đủ dài và có ít nhất 1 đầu mở
            if count >= chain_length:
                open_ends = 0
                if left_end and self.board[left_end[0]][left_end[1]] == EMPTY:
                    open_ends += 1
                if right_end and self.board[right_end[0]][right_end[1]] == EMPTY:
                    open_ends += 1
                
                if open_ends > 0:
                    return True
        
        return False

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

        # --- Improved Intermediate Rewards ---
        reward = 0.0
        opponent = O if player == X else X
        
        # 1. Offensive rewards (tấn công)
        my_len = self._count_consecutive(row, col, player)
        if my_len >= 3:
            # Kiểm tra xem chuỗi có open ends không
            if self._has_open_ends(row, col, player, my_len):
                reward += REWARD_3_IN_ROW  # +0.3
        elif my_len == 2:
            if self._has_open_ends(row, col, player, my_len):
                reward += REWARD_2_IN_ROW  # +0.05

        # 2. Defensive rewards (phòng thủ) - QUAN TRỌNG!
        # Kiểm tra xem nước đi này có chặn được đối thủ không
        self.board[row][col] = EMPTY  # Tạm bỏ nước đi để test
        
        # Giả sử đối thủ đánh vào ô này
        self.board[row][col] = opponent
        opp_len = self._count_consecutive(row, col, opponent)
        
        if opp_len >= WIN_LENGTH:
            reward += REWARD_BLOCK_WIN  # +0.5 - Chặn nước thắng!
        elif opp_len >= 3:
            if self._has_open_ends(row, col, opponent, opp_len):
                reward += REWARD_BLOCK_3  # +0.2 - Chặn 3 quân nguy hiểm
        
        # Khôi phục nước đi thực
        self.board[row][col] = player

        # 3. Small penalty cho nước đi xa trung tâm (chỉ khi không có reward khác)
        if reward == 0.0:
            center = BOARD_SIZE // 2
            dist_from_center = abs(row - center) + abs(col - center)
            if dist_from_center > 4:
                reward -= 0.01
        else:
            reward += REWARD_STEP  # phạt nhẹ mọi nước đi để học đánh nhanh

        return reward, False
