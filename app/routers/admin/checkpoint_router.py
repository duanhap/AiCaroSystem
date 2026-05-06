"""
Admin checkpoint router:
- GET  /admin/checkpoints         → danh sách tất cả version
- POST /admin/checkpoints/deploy  → deploy một version
- POST /admin/checkpoints/delete  → xóa một version
"""
from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db
from app.repositories import checkpoint_repo
from app.services.checkpoint_service import deploy_checkpoint, delete_checkpoint

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
def checkpoints_page(request: Request, db: Session = Depends(get_db)):
    checkpoints = checkpoint_repo.get_all(db)
    return templates.TemplateResponse("admin/checkpoints.html", {
        "request": request,
        "checkpoints": checkpoints,
    })


@router.post("/deploy")
def deploy(version: str = Form(...), db: Session = Depends(get_db)):
    deploy_checkpoint(db, version)
    # Preload checkpoint mới vào cache ngay — user không phải chờ
    from app.services.ai_service import get_ai_agent
    get_ai_agent(db)
    return RedirectResponse("/admin/checkpoints/", status_code=303)


@router.post("/undeploy")
def undeploy(version: str = Form(...), db: Session = Depends(get_db)):
    from app.models.checkpoint import Checkpoint
    from app.services.ai_service import invalidate_cache
    db.query(Checkpoint).filter(Checkpoint.version == version).update({"is_deployed": False})
    db.commit()
    invalidate_cache()  # Không có CP nào deploy → fallback random agent
    return RedirectResponse("/admin/checkpoints/", status_code=303)


@router.post("/delete")
def delete(version: str = Form(...), db: Session = Depends(get_db)):
    delete_checkpoint(db, version)
    return RedirectResponse("/admin/checkpoints/", status_code=303)
