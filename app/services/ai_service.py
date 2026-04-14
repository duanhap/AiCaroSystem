from app.ml.q_agent import QAgent
from app.ml.environment import CaroEnv
from app.repositories import checkpoint_repo
from sqlalchemy.orm import Session

_loaded_agent: QAgent = None

def get_ai_agent(db: Session) -> QAgent:
    """Load Q-table của version đang deploy"""
    global _loaded_agent
    cp = checkpoint_repo.get_deployed(db)
    if cp is None:
        # Chưa có version nào deploy → trả về agent rỗng
        _loaded_agent = QAgent(epsilon=0.0)
        return _loaded_agent
    agent = QAgent(epsilon=0.0)
    agent.load(cp.file_path)
    _loaded_agent = agent
    return agent

def get_ai_move(state: tuple, valid_actions: list, db: Session) -> int:
    agent = get_ai_agent(db)
    return agent.choose_action(state, valid_actions, greedy=True)
