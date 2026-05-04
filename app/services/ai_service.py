import os
import time
from app.ml.q_agent import QAgent
from app.repositories import checkpoint_repo
from sqlalchemy.orm import Session

# Cache: chỉ load lại khi checkpoint_id thay đổi
_cached_agent: QAgent = None
_current_checkpoint_id: int = None
_load_time: float = 0.0


def get_ai_agent(db: Session) -> QAgent:
    """
    Trả về AI agent đang deploy.
    Chỉ load lại file pkl khi checkpoint thay đổi — không load lại mỗi nước đi.
    """
    global _cached_agent, _current_checkpoint_id, _load_time

    cp = checkpoint_repo.get_deployed(db)

    # Không có checkpoint deploy → random agent
    if cp is None or not cp.file_path or not os.path.exists(cp.file_path):
        if _cached_agent is None or _current_checkpoint_id is not None:
            _cached_agent = QAgent(epsilon=1.0)
            _current_checkpoint_id = None
        return _cached_agent

    # Checkpoint không đổi → trả về cache ngay
    if cp.id == _current_checkpoint_id and _cached_agent is not None:
        return _cached_agent

    # Checkpoint mới → load lại
    t0 = time.time()
    agent = QAgent(epsilon=0.0)
    agent.load(cp.file_path)
    _load_time = time.time() - t0

    _cached_agent = agent
    _current_checkpoint_id = cp.id

    import logging
    logging.getLogger("ai_service").info(
        f"Loaded checkpoint {cp.version} ({agent.q_table_size:,} states) in {_load_time:.2f}s"
    )
    return _cached_agent


def get_ai_move(state: tuple, valid_actions: list, db: Session, current_player: int = None) -> int:
    agent = get_ai_agent(db)
    return agent.choose_action(state, valid_actions, greedy=True, current_player=current_player)


def invalidate_cache():
    """Gọi khi deploy/undeploy checkpoint để force reload lần sau"""
    global _cached_agent, _current_checkpoint_id
    _cached_agent = None
    _current_checkpoint_id = None
