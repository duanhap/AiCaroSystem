import os
from app.config import settings
from app.repositories import checkpoint_repo
from app.ml.q_agent import QAgent
from sqlalchemy.orm import Session

def get_next_version(db: Session) -> str:
    checkpoints = checkpoint_repo.get_all(db)
    return f"Q_v{len(checkpoints) + 1}"

def save_checkpoint(db: Session, agent: QAgent, version: str, train_mode: str,
                    base_version: str = None, opponent_version: str = None,
                    compared_version: str = None, episodes_trained: int = 0,
                    win_rate_vs_random: float = None, win_rate_vs_version: float = None,
                    metadata: dict = None):
    os.makedirs(settings.CHECKPOINT_DIR, exist_ok=True)
    file_path = os.path.join(settings.CHECKPOINT_DIR, f"{version}.pkl")
    agent.save(file_path)
    return checkpoint_repo.create(
        db,
        version=version,
        file_path=file_path,
        base_version=base_version,
        opponent_version=opponent_version,
        compared_version=compared_version,
        train_mode=train_mode,
        episodes_trained=episodes_trained,
        win_rate_vs_random=win_rate_vs_random,
        win_rate_vs_version=win_rate_vs_version,
        metadata_=metadata,
    )

def deploy_checkpoint(db: Session, version: str):
    return checkpoint_repo.deploy(db, version)

def delete_checkpoint(db: Session, version: str):
    cp = checkpoint_repo.get_by_version(db, version)
    if cp and cp.file_path and os.path.exists(cp.file_path):
        os.remove(cp.file_path)
    return checkpoint_repo.delete(db, version)

def load_agent(db: Session, version: str) -> QAgent:
    cp = checkpoint_repo.get_by_version(db, version)
    if cp is None:
        raise ValueError(f"Không tìm thấy version {version}")
    agent = QAgent(epsilon=0.0)
    agent.load(cp.file_path)
    return agent
