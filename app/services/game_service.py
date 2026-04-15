"""
Game service: xử lý logic ván cờ PvE và PvP.
Lưu game_steps vào DB sau mỗi nước đi.
"""
from sqlalchemy.orm import Session
from app.ml.environment import CaroEnv
from app.repositories import game_repo


def create_game(db: Session, player_x_id: int, player_o_id: int, mode: str):
    return game_repo.create_game(db, player_x_id, player_o_id, mode)


def apply_move(db: Session, game_id: int, action: int, env: CaroEnv) -> dict:
    """
    Áp dụng nước đi vào env, lưu step vào DB.
    Trả về dict kết quả để gửi về client.
    """
    state_before = env.get_state()
    current_player = env.current_player

    next_state, reward, done = env.step(action)

    # Lưu step vào DB — convert numpy types sang Python thuần
    steps = game_repo.get_steps(db, game_id)
    step_number = len(steps) + 1
    game_repo.add_step(
        db, game_id=game_id,
        step_number=step_number,
        state=[int(x) for x in state_before],
        action=int(action),
        reward=float(reward),
    )

    result = {
        "action": int(action),
        "player": int(current_player),
        "board": [int(x) for x in next_state],
        "done": bool(done),
        "reward": float(reward),
        "winner": None,
    }

    if done:
        if reward == 1.0:
            winner = "X" if current_player == 1 else "O"
        else:
            winner = "draw"
        game_repo.finish_game(db, game_id, winner)
        result["winner"] = winner

    return result


def get_board_state(env: CaroEnv) -> list:
    return list(env.get_state())
