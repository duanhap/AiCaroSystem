"""
Đánh giá AI bằng cách cho đấu vs random agent hoặc vs checkpoint cũ.
Không cập nhật Q-table trong quá trình này.
"""
import random
from app.ml.environment import CaroEnv, X, O
from app.ml.q_agent import QAgent


def _run_single_side(agent_x: QAgent, agent_o: QAgent, n_games: int, count_wins_for: str) -> dict:
    """
    Chạy n_games ván, đếm thắng/thua/hòa cho bên được chỉ định.
    count_wins_for: "x" hoặc "o"
    """
    wins, losses, draws = 0, 0, 0
    env = CaroEnv()

    for _ in range(n_games):
        state = env.reset()
        done = False

        while not done:
            valid = env.get_valid_actions()
            if not valid:
                draws += 1
                break

            # X đánh
            action = agent_x.choose_action(state, valid, greedy=True)
            state, reward, done = env.step(action)
            if done:
                if reward == 1.0:
                    wins += 1 if count_wins_for == "x" else 0
                    losses += 1 if count_wins_for == "o" else 0
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
                    wins += 1 if count_wins_for == "o" else 0
                    losses += 1 if count_wins_for == "x" else 0
                else:
                    draws += 1

    return {"wins": wins, "losses": losses, "draws": draws}


def _run_games(agent: QAgent, opponent: QAgent, n_games: int) -> dict:
    """
    Chạy n_games ván, agent đóng cả 2 vai để đánh giá công bằng.
    n_games/2 ván agent=X (đi trước), n_games/2 ván agent=O (đi sau).
    """
    half = n_games // 2

    # Nửa đầu: agent=X, opponent=O
    r1 = _run_single_side(agent, opponent, half, count_wins_for="x")
    # Nửa sau: agent=O, opponent=X
    r2 = _run_single_side(opponent, agent, n_games - half, count_wins_for="o")

    wins   = r1["wins"]   + r2["wins"]
    losses = r1["losses"] + r2["losses"]
    draws  = r1["draws"]  + r2["draws"]

    return {"wins": wins, "losses": losses, "draws": draws,
            "win_rate": wins / n_games}


def eval_vs_random(agent: QAgent, n_games: int = 100) -> dict:
    """Test agent vs random agent (cả 2 vai)"""
    random_agent = QAgent(epsilon=1.0)  # epsilon=1 → luôn random
    return _run_games(agent, random_agent, n_games)


def eval_vs_checkpoint(agent: QAgent, opponent: QAgent, n_games: int = 100) -> dict:
    """Test agent vs một checkpoint cụ thể (cả 2 vai)"""
    return _run_games(agent, opponent, n_games)
