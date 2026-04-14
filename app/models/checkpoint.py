from sqlalchemy import Column, Integer, String, Float, Boolean, Enum, DateTime, JSON
from sqlalchemy.sql import func
from app.database import Base

class Checkpoint(Base):
    __tablename__ = "checkpoints"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String(50), unique=True, nullable=False)   # Q_v1, Q_v2...
    file_path = Column(String(255), nullable=False)
    base_version = Column(String(50), nullable=True)            # train từ version nào
    opponent_version = Column(String(50), nullable=True)        # đấu với version nào khi train
    compared_version = Column(String(50), nullable=True)        # test so sánh với version nào
    train_mode = Column(Enum("selfplay", "offline"), nullable=False)
    episodes_trained = Column(Integer, default=0)
    win_rate_vs_random = Column(Float, nullable=True)
    win_rate_vs_version = Column(Float, nullable=True)
    is_deployed = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())
    metadata_ = Column("metadata", JSON, nullable=True)         # alpha, gamma, epsilon...
