from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.database import Base

class TrainingLog(Base):
    __tablename__ = "training_logs"

    id = Column(Integer, primary_key=True, index=True)
    checkpoint_id = Column(Integer, ForeignKey("checkpoints.id"), nullable=False)
    episode = Column(Integer, nullable=False)
    win_rate_vs_random = Column(Float, nullable=True)
    win_rate_vs_version = Column(Float, nullable=True)
    avg_reward = Column(Float, nullable=True)
    epsilon = Column(Float, nullable=True)
    timestamp = Column(DateTime, server_default=func.now())
