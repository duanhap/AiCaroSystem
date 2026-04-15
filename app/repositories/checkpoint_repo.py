from sqlalchemy.orm import Session
from app.models.checkpoint import Checkpoint

def get_all(db: Session):
    return db.query(Checkpoint).order_by(Checkpoint.created_at.desc()).all()

def get_by_version(db: Session, version: str):
    return db.query(Checkpoint).filter(Checkpoint.version == version).first()

def get_deployed(db: Session):
    return db.query(Checkpoint).filter(Checkpoint.is_deployed == True).first()

def create(db: Session, **kwargs):
    cp = Checkpoint(**kwargs)
    db.add(cp)
    db.commit()
    db.refresh(cp)
    return cp

def deploy(db: Session, version: str):
    # Bỏ deploy tất cả bản cũ
    db.query(Checkpoint).update({"is_deployed": False})
    cp = get_by_version(db, version)
    if cp:
        cp.is_deployed = True
        db.commit()
    return cp

def delete(db: Session, version: str):
    cp = get_by_version(db, version)
    if cp:
        # Xóa training_logs liên quan trước (tránh FK constraint)
        from app.models.training_log import TrainingLog
        db.query(TrainingLog).filter(TrainingLog.checkpoint_id == cp.id).delete()
        db.delete(cp)
        db.commit()
    return cp
