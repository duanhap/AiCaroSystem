from sqlalchemy.orm import Session
from app.models.training_log import TrainingLog

def add_log(db: Session, checkpoint_id: int, episode: int,
            win_rate_vs_random: float = None, win_rate_vs_version: float = None,
            avg_reward: float = None, epsilon: float = None):
    log = TrainingLog(
        checkpoint_id=checkpoint_id,
        episode=episode,
        win_rate_vs_random=win_rate_vs_random,
        win_rate_vs_version=win_rate_vs_version,
        avg_reward=avg_reward,
        epsilon=epsilon,
    )
    db.add(log)
    db.commit()
    return log

def get_logs_by_checkpoint(db: Session, checkpoint_id: int):
    return db.query(TrainingLog).filter(
        TrainingLog.checkpoint_id == checkpoint_id
    ).order_by(TrainingLog.episode).all()

def migrate_logs(db: Session, from_checkpoint_id: int, to_checkpoint_id: int):
    """Chuyển toàn bộ logs từ checkpoint tạm sang checkpoint thật."""
    db.query(TrainingLog)\
      .filter(TrainingLog.checkpoint_id == from_checkpoint_id)\
      .update({"checkpoint_id": to_checkpoint_id})
    db.commit()
