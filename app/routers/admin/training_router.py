"""
Admin training router:
- GET  /admin/training          → trang cấu hình
- POST /admin/training/selfplay → bắt đầu self-play (SSE stream)
- POST /admin/training/offline  → bắt đầu offline retrain (SSE stream)
- POST /admin/training/commit   → Deploy / Lưu / Bỏ qua sau khi train xong
"""
import json
import threading
from typing import Optional
from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db
from app.repositories import checkpoint_repo, game_repo
from app.services import training_service

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

# Lưu kết quả train tạm (session đơn giản, production dùng Redis)
_pending_result: dict = {}


@router.get("/", response_class=HTMLResponse)
def training_page(request: Request, db: Session = Depends(get_db)):
    checkpoints = checkpoint_repo.get_all(db)
    return templates.TemplateResponse("admin/training.html", {
        "request": request,
        "checkpoints": checkpoints,
    })


@router.get("/selfplay/stream")
def selfplay_stream(
    base_version: Optional[str] = None,
    opponent_version: Optional[str] = "self",
    compare_version: Optional[str] = None,
    episodes: int = 5000,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    test_interval: int = 200,
    test_games: int = 100,
    win_rate_target: float = 0.9,
    enable_convergence: bool = False,
    convergence_threshold: float = 0.001,
    convergence_streak: int = 3,
    db: Session = Depends(get_db),
):
    """SSE endpoint: stream progress self-play realtime"""

    def event_stream():
        progress_queue = []
        done_event = threading.Event()

        def on_progress(entry):
            progress_queue.append(entry)

        def run():
            result = training_service.start_self_play(
                db=db,
                base_version=base_version if base_version else None,
                opponent_version=opponent_version if opponent_version != "self" else None,
                compare_version=compare_version if compare_version else None,
                episodes=episodes,
                alpha=alpha, gamma=gamma,
                epsilon_start=epsilon_start, epsilon_min=epsilon_min,
                epsilon_decay=epsilon_decay,
                test_interval=test_interval,
                test_games=test_games,
                win_rate_target=win_rate_target,
                convergence_threshold=convergence_threshold if enable_convergence else 0.0,
                convergence_streak=convergence_streak if enable_convergence else 999,
                on_progress=on_progress,
            )
            result["train_mode"] = "selfplay"
            _pending_result.clear()
            _pending_result.update(result)
            done_event.set()

        t = threading.Thread(target=run, daemon=True)
        t.start()

        import time
        while not done_event.is_set() or progress_queue:
            while progress_queue:
                entry = progress_queue.pop(0)
                yield f"data: {json.dumps(entry)}\n\n"
            time.sleep(0.1)

        yield f"data: {json.dumps({'status': 'done', **{k: v for k, v in _pending_result.items() if k != 'agent' and k != 'logs'}})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/offline/stream")
def offline_stream(
    base_version: Optional[str] = None,
    compare_version: Optional[str] = None,
    game_ids: str = "",          # comma-separated game IDs
    alpha: float = 0.1,
    gamma: float = 0.9,
    test_games: int = 100,
    db: Session = Depends(get_db),
):
    """SSE endpoint: stream progress offline retrain realtime"""

    ids = [int(x) for x in game_ids.split(",") if x.strip().isdigit()]

    def event_stream():
        progress_queue = []
        done_event = threading.Event()

        def on_progress(entry):
            progress_queue.append(entry)

        def run():
            result = training_service.start_offline_retrain(
                db=db,
                base_version=base_version if base_version else None,
                game_ids=ids,
                compare_version=compare_version if compare_version else None,
                alpha=alpha, gamma=gamma,
                test_games=test_games,
                on_progress=on_progress,
            )
            result["train_mode"] = "offline"
            _pending_result.clear()
            _pending_result.update(result)
            done_event.set()

        t = threading.Thread(target=run, daemon=True)
        t.start()

        import time
        while not done_event.is_set() or progress_queue:
            while progress_queue:
                entry = progress_queue.pop(0)
                yield f"data: {json.dumps(entry)}\n\n"
            time.sleep(0.1)

        yield f"data: {json.dumps({'status': 'done', **{k: v for k, v in _pending_result.items() if k != 'agent' and k != 'logs'}})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/commit")
def commit_result(
    action: str = Form(...),   # "deploy" | "save" | "discard"
    db: Session = Depends(get_db),
):
    """Xử lý Deploy / Lưu / Bỏ qua sau khi train xong"""
    if not _pending_result:
        return {"error": "Không có kết quả training nào đang chờ"}

    cp = training_service.commit_training_result(db, _pending_result, action)
    _pending_result.clear()

    if action == "discard":
        return {"message": "Đã bỏ qua, giữ nguyên version cũ"}
    return {
        "message": f"{'Deploy' if action == 'deploy' else 'Lưu'} thành công",
        "version": cp.version if cp else None,
    }
