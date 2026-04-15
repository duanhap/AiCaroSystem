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
    return templates.TemplateResponse("client/history.html", {
        "request": request, "user": user, "games": games
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
