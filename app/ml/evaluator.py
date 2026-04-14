"""
Đánh giá AI bằng cách cho đấu vs random agent hoặc vs checkpoint cũ.
Không cập nhật Q-table trong quá trình này.
"""
import random
from app.ml.environment import CaroEnv, X, O
from app.ml.q_agent import QAgent


def _run_games(agent_x: QAgent, agent_o: QAgent, n_games: int) -> dict:
    """
    Chạy n_games ván giữa agent_x (X) và agent_o (O).
    Trả về thống kê thắng/thua/hòa.
    """
    wins, losses, draws = 0, 0, 0
    env = CaroEnv()

    for _ in range(n_games):
        state = env.reset()
        done = False

        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break

            # X đánh
            action = agent_x.choose_action(state, valid, greedy=True)
            state, reward, done = env.step(action)
            if done:
                if reward == 1.0:
                    wins += 1
                else:
                    draws += 1
                break

            valid = env.get_valid_actions()
            if not valid:
                draws += 1
                break

            # O đánh
            action = agent_o.choose_action(state, valid, greedy=True)
            state, reward, done = env.step(action)
            if done:
                if reward == 1.0:
                    losses += 1  # O thắng = X thua
                else:
                    draws += 1

    return {"wins": wins, "losses": losses, "draws": draws,
            "win_rate": wins / n_games}


def eval_vs_random(agent: QAgent, n_games: int = 100) -> dict:
    """Test agent vs random agent"""
    random_agent = QAgent(epsilon=1.0)  # epsilon=1 → luôn random
    return _run_games(agent, random_agent, n_games)


def eval_vs_checkpoint(agent: QAgent, opponent: QAgent, n_games: int = 100) -> dict:
    """Test agent vs một checkpoint cụ thể"""
    return _run_games(agent, opponent, n_games)
