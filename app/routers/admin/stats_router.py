"""
Admin stats router:
- GET /admin/stats/              → dashboard tổng quan
- GET /admin/stats/curve/{id}    → learning curve data (JSON)
- GET /admin/stats/games         → thống kê ván chơi theo ngày (JSON)
- GET /admin/stats/games/table   → bảng chọn data cho offline training (HTML)
"""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func, cast, Date

from app.database import get_db
from app.repositories import checkpoint_repo, training_log_repo, game_repo
from app.models.game import Game
from app.models.game_step import GameStep

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
def stats_page(request: Request, db: Session = Depends(get_db)):
    checkpoints = checkpoint_repo.get_all(db)
    deployed = checkpoint_repo.get_deployed(db)
    return templates.TemplateResponse("admin/stats.html", {
        "request": request,
        "checkpoints": checkpoints,
        "deployed": deployed,
    })


@router.get("/curve/{checkpoint_id}")
def learning_curve(checkpoint_id: int, db: Session = Depends(get_db)):
    logs = training_log_repo.get_logs_by_checkpoint(db, checkpoint_id)
    return {
        "labels": [l.episode for l in logs],
        "win_rate_vs_random": [l.win_rate_vs_random for l in logs],
        "win_rate_vs_version": [l.win_rate_vs_version for l in logs],
        "epsilon": [l.epsilon for l in logs],
    }


@router.get("/games")
def game_stats_json(db: Session = Depends(get_db)):
    """Thống kê ván chơi theo ngày — dùng cho offline training"""
    rows = db.query(
        cast(Game.created_at, Date).label("date"),
        func.count(Game.id).label("total_games"),
        func.sum(func.if_(Game.winner == "O", 1, 0)).label("ai_wins"),
        func.sum(func.if_(Game.winner == "X", 1, 0)).label("player_wins"),
        func.sum(func.if_(Game.winner == "draw", 1, 0)).label("draws"),
        func.sum(func.if_(Game.status == "finished", 1, 0)).label("finished"),
    ).filter(Game.mode == "pve").group_by("date").order_by("date").all()

    result = []
    for r in rows:
        total = r.total_games or 0
        ai_wins = int(r.ai_wins or 0)
        player_wins = int(r.player_wins or 0)
        # Lấy game_ids của ngày đó
        game_ids = [g.id for g in db.query(Game.id).filter(
            cast(Game.created_at, Date) == r.date,
            Game.mode == "pve",
            Game.status == "finished",
        ).all()]
        result.append({
            "date": str(r.date),
            "total_games": total,
            "ai_wins": ai_wins,
            "player_wins": player_wins,
            "draws": int(r.draws or 0),
            "ai_lose_rate": round(player_wins / total, 2) if total > 0 else 0,
            "game_ids": game_ids,
        })
    return result


@router.get("/games/table", response_class=HTMLResponse)
def game_stats_table(request: Request, db: Session = Depends(get_db)):
    """Bảng thống kê ván chơi theo ngày để chọn data offline training"""
    stats = game_stats_json(db)
    return templates.TemplateResponse("admin/game_stats.html", {
        "request": request,
        "stats": stats,
    })
