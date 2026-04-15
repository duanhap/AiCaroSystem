import random
import pickle
import math
import numpy as np

BOARD_SIZE = 7
N = BOARD_SIZE * BOARD_SIZE  # 49
CENTER_ROW = BOARD_SIZE // 2
CENTER_COL = BOARD_SIZE // 2


def _center_score(action: int) -> float:
    row, col = divmod(action, BOARD_SIZE)
    dist = math.sqrt((row - CENTER_ROW)**2 + (col - CENTER_COL)**2)
    return -dist


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
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995,
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
            self.q_table[canonical_state] = np.zeros(N, dtype=np.float32)
        return self.q_table[canonical_state]

    def choose_action(self, state: tuple, valid_actions: list, greedy: bool = False) -> int:
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
        data = {k: v.tolist() for k, v in self.q_table.items()}
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=4)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = {}
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
