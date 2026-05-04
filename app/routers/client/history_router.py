"""
Client history router:
- GET /history/       → danh sách ván đã chơi
- GET /history/{id}   → replay từng nước đi
"""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db
from app.repositories import game_repo
from app.routers.client.game_router import get_current_user

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
def history_page(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/auth/login", status_code=302)
    games = game_repo.get_user_games(db, user["id"])

    # Tính kết quả từ góc nhìn user
    def user_result(g):
        if not g.winner or g.status != "finished":
            return None
        if g.winner == "draw":
            return "draw"
        # Xác định user đóng vai gì
        if g.player_x_id == user["id"]:
            user_side = "X"
        elif g.player_o_id == user["id"]:
            user_side = "O"
        else:
            return None
        return "win" if g.winner == user_side else "loss"

    games_with_result = [(g, user_result(g)) for g in games]

    return templates.TemplateResponse("client/history.html", {
        "request": request, "user": user,
        "games": games, "games_with_result": games_with_result
    })


@router.get("/{game_id}", response_class=HTMLResponse)
def replay_page(game_id: int, request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/auth/login", status_code=302)
    game = game_repo.get_game(db, game_id)
    steps = game_repo.get_steps(db, game_id)
    steps_data = [{"step": s.step_number, "state": s.state, "action": s.action} for s in steps]
    return templates.TemplateResponse("client/replay.html", {
        "request": request, "user": user,
        "game": game, "steps": steps_data,
    })
