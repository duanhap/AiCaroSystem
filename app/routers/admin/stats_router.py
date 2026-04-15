"""
Admin stats router:
- GET /admin/stats                → dashboard tổng quan
- GET /admin/stats/curve/{cp_id} → learning curve data (JSON) cho chart
- GET /admin/stats/games          → thống kê người dùng
"""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db
from app.repositories import checkpoint_repo, training_log_repo, game_repo

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
    """Trả về data JSON cho Chart.js vẽ learning curve"""
    logs = training_log_repo.get_logs_by_checkpoint(db, checkpoint_id)
    return {
        "labels": [l.episode for l in logs],
        "win_rate_vs_random": [l.win_rate_vs_random for l in logs],
        "win_rate_vs_version": [l.win_rate_vs_version for l in logs],
        "epsilon": [l.epsilon for l in logs],
    }


@router.get("/games")
def game_stats(db: Session = Depends(get_db)):
    """Thống kê ván chơi theo ngày"""
    from sqlalchemy import func, cast, Date
    from app.models.game import Game

    rows = db.query(
        cast(Game.created_at, Date).label("date"),
        func.count(Game.id).label("total"),
        func.sum((Game.winner == "O").cast(db.bind.dialect.name == "mysql" and "SIGNED" or "INTEGER")).label("ai_wins"),
    ).filter(Game.mode == "pve").group_by("date").order_by("date").all()

    return [{"date": str(r.date), "total": r.total, "ai_wins": r.ai_wins or 0} for r in rows]
