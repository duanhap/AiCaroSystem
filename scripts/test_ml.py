"""
Script test ML core: environment + q_agent
Chạy: python scripts/test_ml.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.environment import CaroEnv, X, O
from app.ml.q_agent import QAgent

def test_env():
    print("── Test Environment ──")
    env = CaroEnv()
    state = env.reset()
    print(f"State size: {len(state)} ô (7×7 = 49)")
    print(f"Valid actions ban đầu: {len(env.get_valid_actions())} ô")

    # Đánh vài nước
    for action in [0, 1, 2, 3]:
        state, reward, done = env.step(action)
        print(f"  Action {action} → reward={reward}, done={done}")
        if not done:
            state, reward, done = env.step(action + 7)  # O đánh

    print("✅ Environment OK\n")

def test_agent():
    print("── Test QAgent ──")
    agent = QAgent(alpha=0.1, gamma=0.9, epsilon=1.0)
    env = CaroEnv()

    # Chạy 10 episode nhanh
    for ep in range(10):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            valid = env.get_valid_actions()
            action = agent.choose_action(state, valid)
            next_state, reward, done = env.step(action)
            next_valid = env.get_valid_actions() if not done else []
            agent.update(state, action, reward, next_state, next_valid, done)
            state = next_state
            steps += 1
            if not done:
                # Đối thủ đánh ngẫu nhiên
                import random
                valid2 = env.get_valid_actions()
                if valid2:
                    a2 = random.choice(valid2)
                    next_state2, reward2, done = env.step(a2)
                    state = next_state2
        agent.decay_epsilon()

    print(f"Q-table size sau 10 episodes: {agent.q_table_size} states")
    print(f"Epsilon sau decay: {agent.epsilon:.4f}")

    # Test save/load
    import tempfile, pathlib
    tmp = str(pathlib.Path(tempfile.gettempdir()) / "test_q.pkl")
    agent.save(tmp)
    agent2 = QAgent()
    agent2.load(tmp)
    print(f"Load lại Q-table: {agent2.q_table_size} states")
    print("✅ QAgent OK\n")

if __name__ == "__main__":
    test_env()
    test_agent()
    print("🎉 Phase 1 ML core hoàn tất!")
