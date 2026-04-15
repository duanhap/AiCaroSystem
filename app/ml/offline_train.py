"""
Train từ dữ liệu người chơi đã lưu trong DB (game_steps).
Không dùng epsilon vì dữ liệu cố định.
Mỗi ván được replay 2 lần: 1 lần đóng vai X, 1 lần đóng vai O.
"""
from typing import Callable, Optional, List
from app.ml.q_agent import QAgent
from app.ml.evaluator import eval_vs_random, eval_vs_checkpoint


def _replay_as_player(agent: QAgent, game_steps: List[dict], play_as: int):
    """
    Replay 1 ván từ góc nhìn của 1 bên (play_as = 1 là X, 2 là O).

    Vấn đề 1 (reward): chỉ bước cuối mới có reward thực, các bước giữa = 0.
    Vấn đề 2 (vai): chỉ update Q cho các bước của play_as, bỏ qua bước đối thủ.
    Vấn đề 3 (next_state cuối): bước cuối dùng done=True nên next_valid=[] là đúng.
    """
    for step in game_steps:
        if step["player"] != play_as:
            continue

        # Reward: giữ nguyên nếu là bước cuối (terminal), còn lại = 0
        reward = step["reward"] if step["done"] else 0.0

        agent.update(
            state=step["state"],
            action=step["action"],
            reward=reward,
            next_state=step["next_state"],
            next_valid_actions=step["next_valid_actions"],
            done=step["done"],
        )


def run_offline_train(
    agent: QAgent,
    game_data: List[List[dict]],
    compare_agent: Optional[QAgent] = None,
    test_games: int = 100,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Train Q-table từ dữ liệu ván chơi đã lưu.
    Mỗi ván replay 2 lần: đóng vai X và đóng vai O.

    Args:
        agent: Agent cần train
        game_data: List các ván, mỗi ván là list các step dict
        compare_agent: Agent để so sánh khi test sau train
        test_games: Số ván test sau khi train xong
        on_progress: Callback({"game_idx": int, "total": int}) để stream realtime
    """
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # offline train không cần explore

    total = len(game_data)

    for idx, game_steps in enumerate(game_data):
        # Replay 2 lần: học từ cả 2 góc nhìn
        _replay_as_player(agent, game_steps, play_as=1)  # vai X
        _replay_as_player(agent, game_steps, play_as=2)  # vai O

        if on_progress:
            on_progress({"game_idx": idx + 1, "total": total})

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
    Mỗi step có thêm field "player" để biết X hay O đánh bước đó.
    """
    from app.repositories.game_repo import get_steps

    all_games = []

    for game_id in game_ids:
        steps = get_steps(db, game_id)
        if not steps:
            continue

        game_steps = []
        n = len(steps)

        for i, step in enumerate(steps):
            is_last = (i == n - 1)

            # Vấn đề 3: next_state bước cuối dùng state terminal thực sự
            next_state = tuple(steps[i + 1].state) if not is_last else tuple(step.state)
            next_valid = [j for j, v in enumerate(next_state) if v == 0] if not is_last else []

            # step_number bắt đầu từ 1: lẻ = X đánh, chẵn = O đánh
            player = 1 if step.step_number % 2 == 1 else 2

            # Reward thực chỉ ở bước cuối, các bước giữa = 0
            reward = float(step.reward) if is_last else 0.0

            game_steps.append({
                "state": tuple(step.state) if isinstance(step.state, list) else step.state,
                "action": step.action,
                "next_state": next_state,
                "next_valid_actions": next_valid,
                "reward": reward,
                "done": is_last,
                "player": player,
            })

        all_games.append(game_steps)

    return all_games
