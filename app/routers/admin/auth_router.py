"""
Admin auth router:
- GET  /admin/login   → trang login
- POST /admin/login   → xử lý login
- GET  /admin/logout  → đăng xuất
"""
from fastapi import APIRouter, Depends, Request, Form, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db
from app.repositories import user_repo
from app.services.auth_service import verify_password, create_access_token

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    # Nếu đã login rồi thì redirect thẳng vào dashboard
    if request.cookies.get("admin_token"):
        return RedirectResponse("/admin/stats/", status_code=302)
    return templates.TemplateResponse("admin/login.html", {"request": request, "error": None})


@router.post("/login")
def login(
    request: Request,
    response: Response,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = user_repo.get_by_username(db, username)
    if not user or not verify_password(password, user.password_hash):
        return templates.TemplateResponse("admin/login.html", {
            "request": request,
            "error": "Sai tên đăng nhập hoặc mật khẩu"
        })
    if user.role != "admin":
        return templates.TemplateResponse("admin/login.html", {
            "request": request,
            "error": "Tài khoản không có quyền admin"
        })

    token = create_access_token({"sub": str(user.id), "role": user.role})
    resp = RedirectResponse("/admin/stats/", status_code=302)
    resp.set_cookie("admin_token", token, httponly=True, max_age=86400)
    return resp


@router.get("/logout")
def logout():
    resp = RedirectResponse("/admin/login", status_code=302)
    resp.delete_cookie("admin_token")
    return resp
