"""
Train từ dữ liệu người chơi đã lưu trong DB (game_steps).
Không dùng epsilon vì dữ liệu cố định.
"""
from typing import Callable, Optional, List
from app.ml.q_agent import QAgent
from app.ml.evaluator import eval_vs_random, eval_vs_checkpoint


def run_offline_train(
    agent: QAgent,
    game_data: List[List[dict]],
    compare_agent: Optional[QAgent] = None,
    test_games: int = 100,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Train Q-table từ dữ liệu ván chơi đã lưu.

    Args:
        agent: Agent cần train (epsilon sẽ bị set = 0 vì dữ liệu cố định)
        game_data: List các ván, mỗi ván là list các step:
                   [{"state": tuple, "action": int, "next_state": tuple,
                     "reward": float, "done": bool}, ...]
        compare_agent: Agent để so sánh khi test sau train
        test_games: Số ván test sau khi train xong
        on_progress: Callback(game_idx, total) để stream realtime

    Returns:
        dict kết quả: win_rate_vs_random, win_rate_vs_version
    """
    # Offline train không cần epsilon
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    total = len(game_data)

    for idx, game_steps in enumerate(game_data):
        for step in game_steps:
            agent.update(
                state=step["state"],
                action=step["action"],
                reward=step["reward"],
                next_state=step["next_state"],
                next_valid_actions=step.get("next_valid_actions", []),
                done=step["done"],
            )

        if on_progress:
            on_progress({"game_idx": idx + 1, "total": total})

    # Khôi phục epsilon
    agent.epsilon = original_epsilon

    # Test sau khi train xong
    result_random = eval_vs_random(agent, test_games)
    result_version = None
    if compare_agent:
        result_version = eval_vs_checkpoint(agent, compare_agent, test_games)

    return {
        "games_trained": total,
        "win_rate_vs_random": round(result_random["win_rate"], 4),
        "win_rate_vs_version": round(result_version["win_rate"], 4) if result_version else None,
    }


def load_game_data_from_db(db, game_ids: List[int]) -> List[List[dict]]:
    """
    Đọc game_steps từ DB và chuyển thành format cho offline_train.
    """
    from app.repositories.game_repo import get_steps
    from app.ml.environment import CaroEnv

    all_games = []
    env = CaroEnv()

    for game_id in game_ids:
        steps = get_steps(db, game_id)
        if not steps:
            continue

        game_steps = []
        for i, step in enumerate(steps):
            is_last = (i == len(steps) - 1)
            next_state = tuple(steps[i + 1].state) if not is_last else tuple(step.state)

            # Tính next_valid_actions từ next_state
            if not is_last:
                next_valid = [j for j, v in enumerate(next_state) if v == 0]
            else:
                next_valid = []

            game_steps.append({
                "state": tuple(step.state) if isinstance(step.state, list) else step.state,
                "action": step.action,
                "next_state": next_state,
                "next_valid_actions": next_valid,
                "reward": step.reward,
                "done": is_last,
            })

        all_games.append(game_steps)

    return all_games
