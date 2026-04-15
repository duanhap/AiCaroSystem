"""
Client game router:
- GET  /game/          → trang chọn chế độ chơi
- GET  /game/pve       → trang chơi vs Bot
- POST /game/pve/start → tạo ván mới vs Bot
- POST /game/pve/move  → đánh nước (AI trả lời ngay)
- GET  /game/pvp       → trang chờ PvP (Phase 5)
"""
import json
from typing import Optional
from fastapi import APIRouter, Depends, Request, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from jose import jwt, JWTError

from app.database import get_db
from app.config import settings
from app.ml.environment import CaroEnv
from app.services.game_service import create_game, apply_move
from app.services.ai_service import get_ai_move
from app.repositories import checkpoint_repo

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

# Lưu env của các ván đang chơi trong memory (đơn giản cho demo)
# Production nên dùng Redis
_active_games: dict = {}


def get_current_user(request: Request):
    token = request.cookies.get("user_token")
    if not token:
        return None
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return {"id": int(payload["sub"]), "username": payload.get("username"), "role": payload.get("role")}
    except JWTError:
        return None


@router.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/auth/login", status_code=302)
    deployed = checkpoint_repo.get_deployed(db)
    return templates.TemplateResponse("client/home.html", {
        "request": request,
        "user": user,
        "ai_version": deployed.version if deployed else None,
    })


@router.get("/pve", response_class=HTMLResponse)
def pve_page(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/auth/login", status_code=302)
    deployed = checkpoint_repo.get_deployed(db)
    return templates.TemplateResponse("client/game.html", {
        "request": request,
        "user": user,
        "mode": "pve",
        "ai_version": deployed.version if deployed else "Chưa có AI",
        "game_id": None,
    })


@router.post("/pve/start")
def pve_start(request: Request, player_side: str = Form("X"), db: Session = Depends(get_db)):
    """Tạo ván mới PvE, trả về game_id và board ban đầu"""
    user = get_current_user(request)
    if not user:
        return {"error": "Chưa đăng nhập"}

    # X đi trước, player chọn X hoặc O
    if player_side == "X":
        game = create_game(db, player_x_id=user["id"], player_o_id=None, mode="pve")
    else:
        game = create_game(db, player_x_id=None, player_o_id=user["id"], mode="pve")

    env = CaroEnv()
    board = list(env.get_state())
    _active_games[game.id] = {"env": env, "player_side": player_side}

    # Nếu player chọn O → AI đánh X trước
    ai_action = None
    if player_side == "O":
        valid = env.get_valid_actions()
        ai_action = get_ai_move(env.get_state(), valid, db)
        result = apply_move(db, game.id, ai_action, env)
        board = result["board"]

    return {
        "game_id": game.id,
        "board": [int(x) for x in board],
        "player_side": player_side,
        "ai_action": int(ai_action) if ai_action is not None else None,
    }


@router.post("/pve/move")
def pve_move(request: Request, game_id: int = Form(...),
             action: int = Form(...), db: Session = Depends(get_db)):
    """Người chơi đánh → AI đánh lại"""
    user = get_current_user(request)
    if not user:
        return {"error": "Chưa đăng nhập"}

    game_data = _active_games.get(game_id)
    if not game_data:
        return {"error": "Ván không tồn tại hoặc đã kết thúc"}

    env: CaroEnv = game_data["env"]

    # Kiểm tra nước đi hợp lệ
    if action not in env.get_valid_actions():
        return {"error": "Nước đi không hợp lệ"}

    # Người chơi đánh
    player_result = apply_move(db, game_id, action, env)
    if player_result["done"]:
        _active_games.pop(game_id, None)
        return {"player_move": player_result, "ai_move": None, "done": True,
                "winner": player_result["winner"]}

    # AI đánh lại
    valid = env.get_valid_actions()
    if not valid:
        _active_games.pop(game_id, None)
        return {"player_move": player_result, "ai_move": None, "done": True, "winner": "draw"}

    ai_action = get_ai_move(env.get_state(), valid, db)
    ai_result = apply_move(db, game_id, ai_action, env)

    if ai_result["done"]:
        _active_games.pop(game_id, None)

    return {
        "player_move": player_result,
        "ai_move": ai_result,
        "done": bool(ai_result["done"]),
        "winner": ai_result.get("winner"),
    }
