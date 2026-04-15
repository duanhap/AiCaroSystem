"""
Vòng lặp self-play: AI tự đấu với chính nó hoặc với một version cũ.
Hỗ trợ emit progress để hiển thị realtime trên Admin UI.
"""
from typing import Callable, Optional
from app.ml.environment import CaroEnv, X, O
from app.ml.q_agent import QAgent
from app.ml.evaluator import eval_vs_random, eval_vs_checkpoint
import random


import logging
logger = logging.getLogger("self_play")

def run_self_play(
    agent: QAgent,
    episodes: int,
    opponent: Optional[QAgent] = None,
    test_interval: int = 200,
    test_games: int = 100,
    win_rate_target: float = 0.9,
    compare_agent: Optional[QAgent] = None,
    on_progress: Optional[Callable] = None,
    convergence_threshold: float = 0.0,
    convergence_streak: int = 999,
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
        convergence_threshold: Delta Q trung bình < ngưỡng này thì coi là hội tụ

    Returns:
        dict kết quả cuối: win_rate_vs_random, win_rate_vs_version, logs
    """
    env = CaroEnv()
    logs = []
    final_result = {}
    prev_q_snapshot = {}
    convergence_checks_passed = 0
    CONVERGENCE_STREAK = convergence_streak
    MIN_EPISODES_BEFORE_CONVERGENCE = max(test_interval * 5, 1000)

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False

        # Xác định đối thủ cho ván này
        opp = opponent if opponent is not None else agent

        # Random vai: ván lẻ agent=X (đi trước), ván chẵn agent=O (đi sau)
        # Giúp agent học cả 2 phía tấn công và phản công
        agent_plays_x = (ep % 2 == 1)

        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break

            # Xác định ai đánh lượt này dựa vào current_player của env
            current_is_agent = (env.current_player == X) == agent_plays_x

            if current_is_agent:
                # Lượt của agent → luôn update Q
                action = agent.choose_action(state, valid)
                next_state, reward, done = env.step(action)
                next_valid = env.get_valid_actions() if not done else []
                agent.update(state, action, reward, next_state, next_valid, done)
            else:
                # Lượt của opponent
                action = opp.choose_action(state, valid)
                next_state, reward, done = env.step(action)
                if opponent is None:
                    # Self-play vs chính nó: học cả 2 phía để hội tụ nhanh hơn
                    next_valid = env.get_valid_actions() if not done else []
                    agent.update(state, action, reward, next_state, next_valid, done)
                elif done and reward == 1.0:
                    # Vs version cũ: opponent thắng → agent thua → phạt
                    agent.update(state, action, -1.0, next_state, [], done)

            state = next_state

        agent.decay_epsilon()

        # Emit progress nhẹ mỗi 50 episode (không test, chỉ báo đang chạy)
        if ep % 50 == 0 and ep % test_interval != 0 and on_progress:
            on_progress({
                "episode": ep,
                "epsilon": round(agent.epsilon, 4),
                "q_table_size": agent.q_table_size,
                "win_rate_vs_random": None,
                "win_rate_vs_version": None,
                "heartbeat": True,
            })

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

            # Đo Q convergence
            current_q_snapshot = {k: dict(v) for k, v in agent.q_table.items()}
            avg_delta = None
            if prev_q_snapshot:
                # Chỉ tính delta trên các state/action đã tồn tại ở CẢ HAI snapshot
                # để tránh nhầm "ít delta" do Q-table đang còn nhỏ
                common_states = set(current_q_snapshot.keys()) & set(prev_q_snapshot.keys())
                deltas = [
                    abs(val - prev_q_snapshot[s][a])
                    for s in common_states
                    for a, val in current_q_snapshot[s].items()
                    if a in prev_q_snapshot[s]
                ]
                if deltas:
                    avg_delta = sum(deltas) / len(deltas)
                    log_entry["avg_q_delta"] = round(avg_delta, 6)
                    log_entry["common_states"] = len(common_states)
            prev_q_snapshot = current_q_snapshot

            logs.append(log_entry)

            # Log ra console để theo dõi từ terminal
            wr = log_entry['win_rate_vs_random']
            delta = log_entry.get('avg_q_delta', '-')
            wrv = log_entry.get('win_rate_vs_version')
            logger.info(
                f"[ep {ep:>6}] ε={log_entry['epsilon']:.4f} | "
                f"Q={log_entry['q_table_size']:>6} states | "
                f"vs_random={wr:.1%}" +
                (f" | vs_version={wrv:.1%}" if wrv is not None else "") +
                (f" | Δq={delta}" if delta != '-' else "")
            )

            if on_progress:
                on_progress(log_entry)

            # Early stopping: win rate đạt target
            # Nếu có compare_agent → dùng win_rate_vs_version làm tiêu chí chính
            # Nếu không → dùng win_rate_vs_random
            if compare_agent and result_version is not None:
                target_met = result_version["win_rate"] >= win_rate_target
                stop_metric = f"vs_version={result_version['win_rate']:.1%}"
            else:
                target_met = result_random["win_rate"] >= win_rate_target
                stop_metric = f"vs_random={result_random['win_rate']:.1%}"

            if target_met:
                final_result["stopped_early"] = True
                final_result["stop_reason"] = "win_rate_target"
                logger.info(f"[ep {ep}] Early stop: {stop_metric} >= {win_rate_target:.1%}")
                break

            # Early stopping: Q hội tụ (chỉ check nếu convergence_threshold > 0)
            if (convergence_threshold > 0
                    and avg_delta is not None
                    and ep >= MIN_EPISODES_BEFORE_CONVERGENCE
                    and avg_delta < convergence_threshold):
                convergence_checks_passed += 1
                if convergence_checks_passed >= CONVERGENCE_STREAK:
                    final_result["stopped_early"] = True
                    final_result["stop_reason"] = "q_converged"
                    break
            else:
                convergence_checks_passed = 0  # Reset nếu không liên tiếp

    final_result.update({
        "episodes_trained": ep,
        "win_rate_vs_random": logs[-1]["win_rate_vs_random"] if logs else None,
        "win_rate_vs_version": logs[-1]["win_rate_vs_version"] if logs else None,
        "logs": logs,
    })
    return final_result
