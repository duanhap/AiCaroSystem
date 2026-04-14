from sqlalchemy import Column, Integer, String, Enum, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.database import Base

class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True, index=True)
    player_x_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    player_o_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    mode = Column(Enum("pvp", "pve"), nullable=False)
    winner = Column(Enum("X", "O", "draw"), nullable=True)
    status = Column(Enum("ongoing", "finished"), default="ongoing")
    created_at = Column(DateTime, server_default=func.now())
