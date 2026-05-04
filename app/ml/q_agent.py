import random
import pickle
import math
import numpy as np

BOARD_SIZE = 7
N = BOARD_SIZE * BOARD_SIZE  # 49
CENTER_ROW = BOARD_SIZE // 2
CENTER_COL = BOARD_SIZE // 2
WIN_LENGTH = 4


def _center_score(action: int) -> float:
    row, col = divmod(action, BOARD_SIZE)
    dist = math.sqrt((row - CENTER_ROW)**2 + (col - CENTER_COL)**2)
    return -dist


def _find_critical_moves(board_state: tuple, valid_actions: list, current_player: int) -> dict:
    """
    Ưu tiên:
    1. win_moves  — đánh vào đây thắng ngay
    2. block_moves — đối thủ đánh vào đây thắng ngay (bắt buộc chặn)
    """
    board = np.array(board_state, dtype=np.int8).reshape(BOARD_SIZE, BOARD_SIZE)
    opponent = 2 if current_player == 1 else 1
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    def count_line(r, c, dr, dc, player):
        count = 0
        rr, cc = r, c
        while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[rr][cc] == player:
            count += 1; rr += dr; cc += dc
        rr, cc = r - dr, c - dc
        while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[rr][cc] == player:
            count += 1; rr -= dr; cc -= dc
        return count

    win_moves = []
    block_moves = []

    for action in valid_actions:
        r, c = divmod(action, BOARD_SIZE)

        # Check thắng ngay
        board[r][c] = current_player
        is_win = any(count_line(r, c, dr, dc, current_player) >= WIN_LENGTH
                     for dr, dc in directions)
        board[r][c] = 0
        if is_win:
            win_moves.append(action)
            continue  # đã là win, không cần check block

        # Check đối thủ thắng nếu đánh vào đây
        board[r][c] = opponent
        is_block = any(count_line(r, c, dr, dc, opponent) >= WIN_LENGTH
                       for dr, dc in directions)
        board[r][c] = 0
        if is_block:
            block_moves.append(action)

    return {"win_moves": win_moves, "block_moves": block_moves}


# --- Symmetry helpers ---
# Precompute 8 permutation arrays: mỗi array là mapping action_original → action_transformed
# Dùng để: canonical_state = min(transform(state) for transform in TRANSFORMS)
# Và map action ngược lại: action_original = INV_TRANSFORMS[i][action_canonical]

def _build_transforms():
    """
    Tạo 8 phép biến đổi (4 xoay × 2 lật) dưới dạng permutation array.
    transforms[i][j] = ô nào trong bàn gốc tương ứng với ô j sau phép biến đổi i.
    """
    transforms = []
    inv_transforms = []

    board_idx = np.arange(N).reshape(BOARD_SIZE, BOARD_SIZE)

    b = board_idx.copy()
    for _ in range(4):
        for flip in [False, True]:
            t = np.fliplr(b) if flip else b
            perm = t.flatten()          # perm[j] = ô gốc tương ứng với ô j sau transform
            inv = np.argsort(perm)      # inv[j] = ô sau transform tương ứng với ô gốc j
            transforms.append(perm)
            inv_transforms.append(inv)
        b = np.rot90(b)

    return transforms, inv_transforms

TRANSFORMS, INV_TRANSFORMS = _build_transforms()


def _canonical(state: tuple):
    """
    Trả về (canonical_state, transform_idx).
    canonical_state = dạng nhỏ nhất trong 8 phép biến đổi.
    transform_idx dùng để map action về bàn gốc.
    """
    arr = np.array(state, dtype=np.int8)
    best = None
    best_idx = 0
    for i, perm in enumerate(TRANSFORMS):
        transformed = tuple(arr[perm])
        if best is None or transformed < best:
            best = transformed
            best_idx = i
    return best, best_idx


def _map_action_to_canonical(action: int, transform_idx: int) -> int:
    """Map action từ bàn gốc → canonical"""
    return int(INV_TRANSFORMS[transform_idx][action])


def _map_action_from_canonical(action: int, transform_idx: int) -> int:
    """Map action từ canonical → bàn gốc"""
    return int(TRANSFORMS[transform_idx][action])


class QAgent:
    def __init__(self, alpha=0.2, gamma=0.9, epsilon=1.0,
                 epsilon_min=0.2, epsilon_decay=0.9995,
                 use_symmetry: bool = True):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.use_symmetry = use_symmetry
        # Q-table: {canonical_state: numpy array shape (49,)}
        self.q_table: dict[tuple, np.ndarray] = {}

    def _get_canonical(self, state: tuple):
        if self.use_symmetry:
            return _canonical(state)
        return state, 0

    def _get_q(self, canonical_state: tuple) -> np.ndarray:
        if canonical_state not in self.q_table:
            # Neutral initialization: cho phép cả positive và negative values
            self.q_table[canonical_state] = np.zeros(N, dtype=np.float32)
        return self.q_table[canonical_state]

    def choose_action(self, state: tuple, valid_actions: list, greedy: bool = False,
                      current_player: int = None) -> int:
        # Rule-based override: ưu tiên thắng ngay hoặc chặn thua
        if current_player is not None:
            critical = _find_critical_moves(state, valid_actions, current_player)
            if critical["win_moves"]:
                return critical["win_moves"][0]   # thắng ngay
            if critical["block_moves"]:
                return critical["block_moves"][0]  # chặn thua

        if not greedy and random.random() < self.epsilon:
            return random.choice(valid_actions)

        canonical_state, t_idx = self._get_canonical(state)
        q = self._get_q(canonical_state)

        # Map valid_actions → canonical space
        canonical_valid = np.array([_map_action_to_canonical(a, t_idx) for a in valid_actions])
        q_valid = q[canonical_valid]
        max_q = q_valid.max()

        best_mask = q_valid == max_q
        best_canonical = canonical_valid[best_mask]
        best_original = np.array([_map_action_from_canonical(a, t_idx) for a in best_canonical])

        if len(best_original) == 1:
            return int(best_original[0])

        # Tiebreak: gần trung tâm hơn
        scores = np.array([_center_score(a) for a in best_original])
        return int(best_original[np.argmax(scores)])

    def update(self, state: tuple, action: int, reward: float,
               next_state: tuple, next_valid_actions: list, done: bool):
        canonical_state, t_idx = self._get_canonical(state)
        canonical_action = _map_action_to_canonical(action, t_idx)

        q = self._get_q(canonical_state)
        current_q = q[canonical_action]

        if done or not next_valid_actions:
            target = reward
        else:
            next_canonical, next_t_idx = self._get_canonical(next_state)
            next_q = self._get_q(next_canonical)
            next_canonical_valid = np.array([
                _map_action_to_canonical(a, next_t_idx) for a in next_valid_actions
            ])
            target = reward + self.gamma * next_q[next_canonical_valid].max()

        q[canonical_action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        """
        Lưu Q-table. 
        Sử dụng format pickle trực tiếp với numpy arrays để tối ưu tốc độ và dung lượng.
        """
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f, protocol=4)

    def load(self, path: str):
        """
        Load Q-table.
        Hỗ trợ cả format cũ (dict of lists/dicts) và format mới (dict of numpy arrays).
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.q_table = {}
        if not data:
            return

        # Kiểm tra item đầu tiên để đoán định dạng
        first_val = next(iter(data.values()))
        
        if isinstance(first_val, np.ndarray):
            # Format mới: cực nhanh
            self.q_table = data
        else:
            # Format cũ: cần convert (chỉ chạy 1 lần duy nhất khi load file cũ)
            for k, v in data.items():
                if isinstance(v, dict):
                    arr = np.zeros(N, dtype=np.float32)
                    for action, val in v.items():
                        arr[int(action)] = val
                    self.q_table[k] = arr
                else:
                    self.q_table[k] = np.array(v, dtype=np.float32)

    @property
    def q_table_size(self):
        return len(self.q_table)
