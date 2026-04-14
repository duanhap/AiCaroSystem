import random
import pickle
from collections import defaultdict

class QAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # Q-table: {state: {action: value}}
        self.q_table = defaultdict(lambda: defaultdict(float))

    def choose_action(self, state, valid_actions: list, greedy: bool = False):
        """Chọn nước đi theo ε-greedy (hoặc greedy khi test)"""
        if not greedy and random.random() < self.epsilon:
            return random.choice(valid_actions)
        q_vals = {a: self.q_table[state][a] for a in valid_actions}
        return max(q_vals, key=q_vals.get)

    def update(self, state, action, reward, next_state, next_valid_actions, done):
        """Cập nhật Q-table theo công thức Bellman"""
        current_q = self.q_table[state][action]
        if done or not next_valid_actions:
            target = reward
        else:
            max_next_q = max(self.q_table[next_state][a] for a in next_valid_actions)
            target = reward + self.gamma * max_next_q
        self.q_table[state][action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: defaultdict(float), {
            k: defaultdict(float, v) for k, v in data.items()
        })

    @property
    def q_table_size(self):
        return len(self.q_table)
