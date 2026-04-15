"""
Điều phối self-play và offline retrain.
Lưu training_logs vào DB sau mỗi test interval.
"""
import copy
import time
from typing import Optional, Callable
from sqlalchemy.orm import Session

from app.ml.q_agent import QAgent
from app.ml.self_play import run_self_play
from app.ml.offline_train import run_offline_train, load_game_data_from_db
from app.services.checkpoint_service import save_checkpoint, load_agent, get_next_version
from app.repositories import training_log_repo, checkpoint_repo


def start_self_play(
    db: Session,
    base_version: Optional[str],
    opponent_version: Optional[str],
    compare_version: Optional[str],
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_min: float,
    epsilon_decay: float,
    test_interval: int,
    test_games: int,
    win_rate_target: float,
    convergence_threshold: float = 0.0,
    convergence_streak: int = 999,
    pause_event=None,
    stop_event=None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Chạy self-play và trả về kết quả (chưa lưu checkpoint)."""

    # Load base model
    if base_version:
        agent = load_agent(db, base_version)
        agent.alpha = alpha
        agent.gamma = gamma
        agent.epsilon = epsilon_start
        agent.epsilon_min = epsilon_min
        agent.epsilon_decay = epsilon_decay
    else:
        agent = QAgent(alpha=alpha, gamma=gamma, epsilon=epsilon_start,
                       epsilon_min=epsilon_min, epsilon_decay=epsilon_decay)

    # Load opponent
    opponent = None
    if opponent_version and opponent_version != "self":
        opponent = load_agent(db, opponent_version)
        opponent.epsilon = 0.0

    # Load compare agent
    compare_agent = None
    if compare_version:
        compare_agent = load_agent(db, compare_version)
        compare_agent.epsilon = 0.0

    # Tạo checkpoint tạm để lưu logs (chưa có file)
    new_version = get_next_version(db)

    # Wrapper on_progress để lưu log vào DB
    temp_cp_id = [None]

    def progress_handler(entry):
        # Lần đầu tạo checkpoint tạm — dùng timestamp để tránh duplicate
        if temp_cp_id[0] is None:
            tmp_version = f"{new_version}_tmp_{int(time.time())}"
            # Xóa các _tmp cũ của cùng new_version nếu còn sót (xóa logs trước)
            from app.models.checkpoint import Checkpoint as CpModel
            from app.models.training_log import TrainingLog as LogModel
            old_tmps = db.query(CpModel).filter(
                CpModel.version.like(f"{new_version}_tmp%")
            ).all()
            for old in old_tmps:
                db.query(LogModel).filter(LogModel.checkpoint_id == old.id).delete()
                db.delete(old)
            db.commit()

            cp = checkpoint_repo.create(
                db,
                version=tmp_version,
                file_path="",
                base_version=base_version,
                opponent_version=opponent_version,
                compared_version=compare_version,
                train_mode="selfplay",
                episodes_trained=0,
            )
            temp_cp_id[0] = cp.id

        training_log_repo.add_log(
            db,
            checkpoint_id=temp_cp_id[0],
            episode=entry["episode"],
            win_rate_vs_random=entry.get("win_rate_vs_random"),
            win_rate_vs_version=entry.get("win_rate_vs_version"),
            epsilon=entry.get("epsilon"),
        )
        if on_progress:
            on_progress(entry)

    result = run_self_play(
        agent=agent,
        episodes=episodes,
        opponent=opponent,
        test_interval=test_interval,
        test_games=test_games,
        win_rate_target=win_rate_target,
        compare_agent=compare_agent,
        convergence_threshold=convergence_threshold,
        convergence_streak=convergence_streak,
        pause_event=pause_event,
        stop_event=stop_event,
        on_progress=progress_handler,
    )

    result["agent"] = agent
    result["new_version"] = new_version
    result["base_version"] = base_version
    result["opponent_version"] = opponent_version
    result["compare_version"] = compare_version
    result["temp_cp_id"] = temp_cp_id[0]
    result["train_params"] = {
        "alpha": alpha, "gamma": gamma,
        "epsilon_start": epsilon_start, "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay,
    }
    return result


def start_offline_retrain(
    db: Session,
    base_version: Optional[str],
    game_ids: list,
    compare_version: Optional[str],
    alpha: float,
    gamma: float,
    test_games: int,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Chạy offline retrain từ dữ liệu người chơi."""

    if base_version:
        agent = load_agent(db, base_version)
        agent.alpha = alpha
        agent.gamma = gamma
    else:
        agent = QAgent(alpha=alpha, gamma=gamma, epsilon=0.0)

    compare_agent = None
    if compare_version:
        compare_agent = load_agent(db, compare_version)
        compare_agent.epsilon = 0.0

    game_data = load_game_data_from_db(db, game_ids)

    new_version = get_next_version(db)

    result = run_offline_train(
        agent=agent,
        game_data=game_data,
        compare_agent=compare_agent,
        test_games=test_games,
        on_progress=on_progress,
    )

    result["agent"] = agent
    result["new_version"] = new_version
    result["base_version"] = base_version
    result["compare_version"] = compare_version
    result["train_params"] = {"alpha": alpha, "gamma": gamma}
    return result


def commit_training_result(db: Session, result: dict, action: str) -> Optional[object]:
    """
    Sau khi train xong, admin chọn:
    - 'deploy': lưu + deploy ngay
    - 'save': lưu nhưng chưa deploy
    - 'discard': bỏ qua, không lưu
    """
    if action == "discard":
        # Xóa checkpoint tạm nếu có
        if result.get("temp_cp_id"):
            tmp = db.query(__import__("app.models.checkpoint", fromlist=["Checkpoint"]).Checkpoint)\
                    .filter_by(id=result["temp_cp_id"]).first()
            if tmp:
                db.delete(tmp)
                db.commit()
        return None

    cp = save_checkpoint(
        db=db,
        agent=result["agent"],
        version=result["new_version"],
        train_mode=result.get("train_mode", "selfplay"),
        base_version=result.get("base_version"),
        opponent_version=result.get("opponent_version"),
        compared_version=result.get("compare_version"),
        episodes_trained=result.get("episodes_trained", 0),
        win_rate_vs_random=result.get("win_rate_vs_random"),
        win_rate_vs_version=result.get("win_rate_vs_version"),
        metadata=result.get("train_params"),
    )

    # Cập nhật logs tạm sang checkpoint thật
    if result.get("temp_cp_id"):
        db.query(__import__("app.models.training_log", fromlist=["TrainingLog"]).TrainingLog)\
          .filter_by(checkpoint_id=result["temp_cp_id"])\
          .update({"checkpoint_id": cp.id})
        tmp = db.query(__import__("app.models.checkpoint", fromlist=["Checkpoint"]).Checkpoint)\
                .filter_by(id=result["temp_cp_id"]).first()
        if tmp:
            db.delete(tmp)
        db.commit()

    if action == "deploy":
        from app.services.checkpoint_service import deploy_checkpoint
        deploy_checkpoint(db, cp.version)

    return cp
