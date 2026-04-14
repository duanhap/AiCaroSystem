"""
Vòng lặp self-play: AI tự đấu với chính nó hoặc với một version cũ.
Hỗ trợ emit progress để hiển thị realtime trên Admin UI.
"""
from typing import Callable, Optional
from app.ml.environment import CaroEnv, X, O
from app.ml.q_agent import QAgent
from app.ml.evaluator import eval_vs_random, eval_vs_checkpoint


def run_self_play(
    agent: QAgent,
    episodes: int,
    opponent: Optional[QAgent] = None,
    test_interval: int = 200,
    test_games: int = 100,
    win_rate_target: float = 0.9,
    compare_agent: Optional[QAgent] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Chạy self-play training.

    Args:
        agent: Agent đang train
        episodes: Số ván tối đa
        opponent: Agent đối thủ (None = tự đấu với chính mình)
        test_interval: Cứ N ván test 1 lần
        test_games: Số ván mỗi lần test
        win_rate_target: Ngưỡng early stopping
        compare_agent: Agent để so sánh khi test (None = chỉ test vs random)
        on_progress: Callback(episode, epsilon, q_size, test_result) để stream realtime

    Returns:
        dict kết quả cuối: win_rate_vs_random, win_rate_vs_version, logs
    """
    env = CaroEnv()
    logs = []
    final_result = {}

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False

        # Xác định đối thủ cho ván này
        opp = opponent if opponent is not None else agent

        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break

            # X đánh (agent đang train)
            action = agent.choose_action(state, valid)
            next_state, reward, done = env.step(action)
            next_valid = env.get_valid_actions() if not done else []
            agent.update(state, action, reward, next_state, next_valid, done)
            state = next_state

            if done:
                break

            valid = env.get_valid_actions()
            if not valid:
                break

            # O đánh (opponent)
            action_o = opp.choose_action(state, valid)
            next_state, reward_o, done = env.step(action_o)

            # Nếu O thắng → X thua → cập nhật Q với reward âm
            if done and reward_o == 1.0:
                agent.update(state, action_o, -1.0, next_state, [], done)

            state = next_state

        agent.decay_epsilon()

        # Test định kỳ
        if ep % test_interval == 0 or ep == episodes:
            result_random = eval_vs_random(agent, test_games)
            result_version = None
            if compare_agent:
                result_version = eval_vs_checkpoint(agent, compare_agent, test_games)

            log_entry = {
                "episode": ep,
                "epsilon": round(agent.epsilon, 4),
                "q_table_size": agent.q_table_size,
                "win_rate_vs_random": round(result_random["win_rate"], 4),
                "win_rate_vs_version": round(result_version["win_rate"], 4) if result_version else None,
            }
            logs.append(log_entry)

            if on_progress:
                on_progress(log_entry)

            # Early stopping
            if result_random["win_rate"] >= win_rate_target:
                final_result["stopped_early"] = True
                break

    final_result.update({
        "episodes_trained": ep,
        "win_rate_vs_random": logs[-1]["win_rate_vs_random"] if logs else None,
        "win_rate_vs_version": logs[-1]["win_rate_vs_version"] if logs else None,
        "logs": logs,
    })
    return final_result
