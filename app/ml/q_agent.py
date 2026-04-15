import random
import pickle
import math
import numpy as np
from collections import defaultdict

BOARD_SIZE = 7
CENTER_ROW = BOARD_SIZE // 2
CENTER_COL = BOARD_SIZE // 2


def _center_score(action: int) -> float:
    """Tiebreaker: ô gần trung tâm ưu tiên hơn"""
    row, col = divmod(action, BOARD_SIZE)
    dist = math.sqrt((row - CENTER_ROW)**2 + (col - CENTER_COL)**2)
    return -dist


class QAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # Q-table: {state_tuple: numpy array shape (49,)}
        # Dùng numpy array thay dict để lookup nhanh hơn
        self.q_table: dict[tuple, np.ndarray] = {}

    def _get_q(self, state: tuple) -> np.ndarray:
        """Lấy Q-values cho state, khởi tạo nếu chưa có"""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        return self.q_table[state]

    def choose_action(self, state: tuple, valid_actions: list, greedy: bool = False) -> int:
        """ε-greedy với tiebreaking bằng center score"""
        if not greedy and random.random() < self.epsilon:
            return random.choice(valid_actions)

        q = self._get_q(state)
        # Lấy Q-values của các action hợp lệ
        valid = np.array(valid_actions)
        q_valid = q[valid]
        max_q = q_valid.max()

        # Tiebreak: nếu nhiều action cùng max → ưu tiên gần trung tâm
        best_mask = q_valid == max_q
        best_actions = valid[best_mask]
        if len(best_actions) == 1:
            return int(best_actions[0])

        # Chọn action gần trung tâm nhất trong số best
        scores = np.array([_center_score(a) for a in best_actions])
        return int(best_actions[np.argmax(scores)])

    def update(self, state: tuple, action: int, reward: float,
               next_state: tuple, next_valid_actions: list, done: bool):
        """Cập nhật Q-table theo Bellman"""
        q = self._get_q(state)
        current_q = q[action]

        if done or not next_valid_actions:
            target = reward
        else:
            next_q = self._get_q(next_state)
            next_valid = np.array(next_valid_actions)
            target = reward + self.gamma * next_q[next_valid].max()

        q[action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        # Convert numpy arrays sang list để pickle nhỏ hơn
        data = {k: v.tolist() for k, v in self.q_table.items()}
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=4)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Hỗ trợ cả format cũ (dict of dict) và mới (dict of list/array)
        self.q_table = {}
        for k, v in data.items():
            if isinstance(v, dict):
                # Format cũ: {action: value}
                arr = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
                for action, val in v.items():
                    arr[int(action)] = val
                self.q_table[k] = arr
            else:
                self.q_table[k] = np.array(v, dtype=np.float32)

    @property
    def q_table_size(self):
        return len(self.q_table)
