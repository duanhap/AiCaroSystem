"""
Admin evaluation router:
- GET  /admin/evaluation          → trang chạy test thủ công
- POST /admin/evaluation/run      → chạy test, trả về kết quả JSON
"""
from typing import Optional
from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db
from app.repositories import checkpoint_repo
from app.services.evaluation_service import run_manual_eval

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
def evaluation_page(request: Request, db: Session = Depends(get_db)):
    checkpoints = checkpoint_repo.get_all(db)
    return templates.TemplateResponse("admin/evaluation.html", {
        "request": request,
        "checkpoints": checkpoints,
        "result": None,
    })


@router.post("/run", response_class=HTMLResponse)
def run_eval(
    request: Request,
    version: str = Form(...),
    compare_version: Optional[str] = Form(None),
    n_games: int = Form(100),
    db: Session = Depends(get_db),
):
    checkpoints = checkpoint_repo.get_all(db)
    result = run_manual_eval(db, version, compare_version or None, n_games)
    return templates.TemplateResponse("admin/evaluation.html", {
        "request": request,
        "checkpoints": checkpoints,
        "result": result,
    })
