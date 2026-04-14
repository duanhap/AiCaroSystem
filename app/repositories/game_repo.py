from sqlalchemy.orm import Session
from app.models.game import Game
from app.models.game_step import GameStep

def create_game(db: Session, player_x_id, player_o_id, mode: str):
    game = Game(player_x_id=player_x_id, player_o_id=player_o_id, mode=mode)
    db.add(game)
    db.commit()
    db.refresh(game)
    return game

def get_game(db: Session, game_id: int):
    return db.query(Game).filter(Game.id == game_id).first()

def finish_game(db: Session, game_id: int, winner: str):
    game = get_game(db, game_id)
    if game:
        game.winner = winner
        game.status = "finished"
        db.commit()
    return game

def add_step(db: Session, game_id: int, step_number: int, state, action: int, reward: float):
    step = GameStep(game_id=game_id, step_number=step_number,
                    state=state, action=action, reward=reward)
    db.add(step)
    db.commit()
    return step

def get_steps(db: Session, game_id: int):
    return db.query(GameStep).filter(GameStep.game_id == game_id).order_by(GameStep.step_number).all()

def get_user_games(db: Session, user_id: int, limit: int = 20):
    return db.query(Game).filter(
        (Game.player_x_id == user_id) | (Game.player_o_id == user_id)
    ).order_by(Game.created_at.desc()).limit(limit).all()
