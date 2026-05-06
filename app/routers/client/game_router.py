"""
Client game router:
- GET  /game/          → trang chọn chế độ chơi
- GET  /game/pve       → trang chơi vs Bot
- POST /game/pve/start → tạo ván mới vs Bot
- POST /game/pve/move  → đánh nước (AI trả lời ngay)
- GET  /game/pvp       → trang PvP (WebSocket realtime)
- WS   /game/pvp/ws/{room_code}/{username} → WebSocket PvP
"""
import json
import random
import string
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
from app.services.checkpoint_service import get_deployed_version
from app.repositories import game_repo

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

# Lưu env của các ván đang chơi trong memory
_active_games: dict = {}

# PvP rooms: {room_code: {"players": [ws1, ws2], "env": CaroEnv, "usernames": [], "game_id": int, "db": Session}}
_pvp_rooms: dict = {}


def get_current_user(request: Request):
    token = request.cookies.get("user_token")
    if not token:
        return None
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return {"id": int(payload["sub"]), "username": payload.get("username"), "role": payload.get("role")}
    except JWTError:
        return None


def _gen_room_code():
    return ''.join(random.choices(string.digits, k=6))


@router.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/auth/login", status_code=302)
    deployed = get_deployed_version(db)
    return templates.TemplateResponse("client/home.html", {
        "request": request,
        "user": user,
        "ai_version": deployed,
    })


@router.get("/pve", response_class=HTMLResponse)
def pve_page(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/auth/login", status_code=302)
    deployed = get_deployed_version(db)
    return templates.TemplateResponse("client/game.html", {
        "request": request,
        "user": user,
        "mode": "pve",
        "ai_version": deployed if deployed else "Chưa có AI",
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
        ai_action = get_ai_move(env.get_state(), valid, db, current_player=env.current_player)
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
    import logging
    logger = logging.getLogger("pve_move")
    
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

    try:
        # Người chơi đánh
        logger.info(f"Game {game_id}: Player move {action}")
        player_result = apply_move(db, game_id, action, env)
        if player_result["done"]:
            _active_games.pop(game_id, None)
            logger.info(f"Game {game_id}: Player won/draw")
            return {"player_move": player_result, "ai_move": None, "done": True,
                    "winner": player_result["winner"]}

        # AI đánh lại
        valid = env.get_valid_actions()
        if not valid:
            _active_games.pop(game_id, None)
            logger.info(f"Game {game_id}: Draw (no valid moves)")
            return {"player_move": player_result, "ai_move": None, "done": True, "winner": "draw"}

        logger.info(f"Game {game_id}: AI thinking... (valid: {len(valid)} moves)")
        ai_action = get_ai_move(env.get_state(), valid, db, current_player=env.current_player)
        logger.info(f"Game {game_id}: AI chose {ai_action}")
        
        ai_result = apply_move(db, game_id, ai_action, env)

        if ai_result["done"]:
            _active_games.pop(game_id, None)
            logger.info(f"Game {game_id}: AI won/draw")

        return {
            "player_move": player_result,
            "ai_move": ai_result,
            "done": bool(ai_result["done"]),
            "winner": ai_result.get("winner"),
        }
    except Exception as e:
        logger.error(f"Game {game_id}: Error - {e}", exc_info=True)
        return {"error": f"Lỗi server: {str(e)}"}


@router.get("/pvp", response_class=HTMLResponse)
def pvp_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/auth/login", status_code=302)
    return templates.TemplateResponse("client/pvp.html", {
        "request": request, "user": user
    })


@router.post("/pvp/create")
def pvp_create(request: Request):
    """Tạo phòng mới, trả về room_code"""
    user = get_current_user(request)
    if not user:
        return {"error": "Chưa đăng nhập"}
    for _ in range(10):
        code = _gen_room_code()
        if code not in _pvp_rooms:
            break
    _pvp_rooms[code] = {
        "players": [],
        "user_ids": [],
        "usernames": [],
        "sides": {},       # user_id -> "X" or "O"
        "env": None,
        "game_id": None,
        "current_turn": 0,
        "timer_task": None,
    }
    return {"room_code": code}


@router.websocket("/pvp/ws/{room_code}")
async def pvp_websocket(websocket: WebSocket, room_code: str):
    """WebSocket endpoint cho PvP — không dùng Depends(get_db) vì không tương thích"""
    import asyncio
    await websocket.accept()

    # Tạo DB session thủ công
    from app.database import SessionLocal
    db = SessionLocal()

    # Lấy user từ cookie
    token = websocket.cookies.get("user_token")
    user = None
    if token:
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            user = {"id": int(payload["sub"]), "username": payload.get("username")}
        except JWTError:
            pass

    if not user:
        await websocket.send_json({"type": "error", "message": "Chưa đăng nhập"})
        await websocket.close()
        db.close()
        return

    if room_code not in _pvp_rooms:
        await websocket.send_json({"type": "error", "message": "Phòng không tồn tại"})
        await websocket.close()
        db.close()
        return

    room = _pvp_rooms[room_code]

    if len(room["players"]) >= 2 and user["id"] not in room["user_ids"]:
        await websocket.send_json({"type": "error", "message": "Phòng đã đầy"})
        await websocket.close()
        db.close()
        return

    # Thêm hoặc reconnect player
    if user["id"] not in room["user_ids"]:
        room["players"].append(websocket)
        room["user_ids"].append(user["id"])
        room["usernames"].append(user["username"])
        player_idx = len(room["players"]) - 1
    else:
        idx = room["user_ids"].index(user["id"])
        room["players"][idx] = websocket
        player_idx = idx

    # Nếu đủ 2 người → random side và bắt đầu
    if len(room["players"]) == 2 and room["env"] is None:
        # Random ai là X (đi trước)
        order = [0, 1]
        random.shuffle(order)
        room["sides"][room["user_ids"][order[0]]] = "X"
        room["sides"][room["user_ids"][order[1]]] = "O"
        # current_turn = index của người chơi X
        room["current_turn"] = order[0]

        env = CaroEnv()
        room["env"] = env
        game = game_repo.create_game(db, room["user_ids"][order[0]], room["user_ids"][order[1]], "pvp")
        room["game_id"] = game.id

        # Gửi start cho cả 2
        for i, ws in enumerate(room["players"]):
            uid = room["user_ids"][i]
            my_side = room["sides"][uid]
            try:
                await ws.send_json({
                    "type": "start",
                    "side": my_side,
                    "board": list(env.get_state()),
                    "players": room["usernames"],
                    "current_turn": room["usernames"][room["current_turn"]],
                    "game_id": game.id,
                })
            except Exception:
                pass

        # Bắt đầu timer cho người đi trước
        _start_turn_timer(room, room_code, db)

    else:
        # Người đầu tiên chờ
        await websocket.send_json({
            "type": "waiting",
            "message": f"Chờ người chơi thứ 2... Mã phòng: {room_code}",
        })
        # Cập nhật side nếu đã có
        if user["id"] in room["sides"]:
            await websocket.send_json({
                "type": "joined",
                "side": room["sides"][user["id"]],
                "room_code": room_code,
            })

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "move":
                action = int(data["action"])
                env = room["env"]

                if env is None or env.done:
                    await websocket.send_json({"type": "error", "message": "Game chưa bắt đầu hoặc đã kết thúc"})
                    continue

                if room["current_turn"] != player_idx:
                    await websocket.send_json({"type": "error", "message": "Chưa đến lượt bạn"})
                    continue

                if action not in env.get_valid_actions():
                    await websocket.send_json({"type": "error", "message": "Nước đi không hợp lệ"})
                    continue

                # Hủy timer hiện tại
                if room.get("timer_task"):
                    room["timer_task"].cancel()
                    room["timer_task"] = None

                side = room["sides"][user["id"]]
                result = apply_move(db, room["game_id"], action, env)

                move_msg = {
                    "type": "move",
                    "action": action,
                    "player": player_idx,
                    "side": side,
                    "board": result["board"],
                    "done": result["done"],
                    "winner": result.get("winner"),
                }

                if not result["done"]:
                    room["current_turn"] = 1 - room["current_turn"]
                    move_msg["current_turn"] = room["usernames"][room["current_turn"]]
                    move_msg["timer"] = 10  # 10 giây cho lượt tiếp

                for ws in room["players"]:
                    try:
                        await ws.send_json(move_msg)
                    except Exception:
                        pass

                if result["done"]:
                    async def cleanup():
                        await asyncio.sleep(30)
                        _pvp_rooms.pop(room_code, None)
                    asyncio.create_task(cleanup())
                else:
                    _start_turn_timer(room, room_code, db)

            elif data.get("type") == "play_again":
                # Đánh dấu người này muốn chơi lại
                if "play_again_votes" not in room:
                    room["play_again_votes"] = set()
                room["play_again_votes"].add(player_idx)

                if len(room["play_again_votes"]) == 2:
                    # Cả 2 đồng ý → reset game
                    room["play_again_votes"] = set()
                    env = CaroEnv()
                    room["env"] = env
                    # Random lại side
                    order = [0, 1]
                    random.shuffle(order)
                    room["sides"][room["user_ids"][order[0]]] = "X"
                    room["sides"][room["user_ids"][order[1]]] = "O"
                    room["current_turn"] = order[0]
                    game = game_repo.create_game(db, room["user_ids"][order[0]], room["user_ids"][order[1]], "pvp")
                    room["game_id"] = game.id

                    for i, ws_p in enumerate(room["players"]):
                        uid = room["user_ids"][i]
                        try:
                            await ws_p.send_json({
                                "type": "play_again",
                                "side": room["sides"][uid],
                                "board": list(env.get_state()),
                                "players": room["usernames"],
                                "current_turn": room["usernames"][room["current_turn"]],
                            })
                        except Exception:
                            pass
                    _start_turn_timer(room, room_code, db)
                else:
                    # Thông báo đang chờ người kia
                    await websocket.send_json({"type": "waiting_play_again", "message": "Chờ đối thủ đồng ý chơi lại..."})
                    # Thông báo người kia có request chơi lại
                    other_idx = 1 - player_idx
                    if other_idx < len(room["players"]):
                        try:
                            await room["players"][other_idx].send_json({
                                "type": "play_again_request",
                                "from": user["username"]
                            })
                        except Exception:
                            pass

    except WebSocketDisconnect:
        other_idx = 1 - player_idx
        env = room.get("env")
        game_done = env is None or env.done

        if not game_done:
            # Đang chơi mà thoát → người thoát thua
            if room.get("timer_task"):
                room["timer_task"].cancel()
            winner_idx = other_idx
            winner_side = room["sides"].get(room["user_ids"][winner_idx], "X") if room.get("sides") else "X"
            if room.get("game_id"):
                game_repo.finish_game(db, room["game_id"], winner_side)
            if other_idx < len(room["players"]):
                try:
                    await room["players"][other_idx].send_json({
                        "type": "opponent_left",
                        "message": f"{user['username']} đã thoát.",
                        "game_over": True,
                    })
                except Exception:
                    pass
        else:
            # Ván đã xong, thoát bình thường → người kia cũng về trang chủ
            if other_idx < len(room["players"]):
                try:
                    await room["players"][other_idx].send_json({
                        "type": "opponent_left",
                        "message": f"{user['username']} đã rời phòng.",
                        "game_over": False,
                    })
                except Exception:
                    pass
        _pvp_rooms.pop(room_code, None)
    finally:
        db.close()


def _start_turn_timer(room: dict, room_code: str, db):
    """Bắt đầu đếm ngược 10s, nếu hết giờ thì xử thua"""
    import asyncio

    async def _timer():
        await asyncio.sleep(10)
        if room_code not in _pvp_rooms:
            return
        env = room.get("env")
        if env is None or env.done:
            return
        # Người đang đến lượt bị xử thua
        loser_idx = room["current_turn"]
        loser_name = room["usernames"][loser_idx]
        winner_idx = 1 - loser_idx
        winner_name = room["usernames"][winner_idx]
        # Kết thúc game
        if room.get("game_id"):
            winner_side = room["sides"][room["user_ids"][winner_idx]]
            game_repo.finish_game(db, room["game_id"], winner_side)
        timeout_msg = {
            "type": "timeout",
            "loser": loser_name,
            "winner": winner_name,
            "done": True,
            "winner_side": room["sides"].get(room["user_ids"][winner_idx], ""),
        }
        for ws in room["players"]:
            try:
                await ws.send_json(timeout_msg)
            except Exception:
                pass
        _pvp_rooms.pop(room_code, None)

    task = asyncio.create_task(_timer())
    room["timer_task"] = task
