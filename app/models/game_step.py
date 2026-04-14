from sqlalchemy import Column, Integer, Float, JSON, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.database import Base

class GameStep(Base):
    __tablename__ = "game_steps"

    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    step_number = Column(Integer, nullable=False)
    state = Column(JSON, nullable=False)
    action = Column(Integer, nullable=False)  # 0-48 (ô trên bàn 7x7)
    reward = Column(Float, default=0.0)
    timestamp = Column(DateTime, server_default=func.now())
