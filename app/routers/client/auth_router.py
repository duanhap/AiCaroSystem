"""
Client auth router:
- GET  /auth/login     → trang login
- POST /auth/login     → xử lý login
- GET  /auth/register  → trang đăng ký
- POST /auth/register  → xử lý đăng ký
- GET  /auth/logout    → đăng xuất
"""
from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db
from app.repositories import user_repo
from app.services.auth_service import hash_password, verify_password, create_access_token

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if request.cookies.get("user_token"):
        return RedirectResponse("/game/", status_code=302)
    return templates.TemplateResponse("client/login.html", {"request": request, "error": None})


@router.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...),
          db: Session = Depends(get_db)):
    user = user_repo.get_by_username(db, username)
    if not user or not verify_password(password, user.password_hash):
        return templates.TemplateResponse("client/login.html", {
            "request": request, "error": "Sai tên đăng nhập hoặc mật khẩu"
        })
    token = create_access_token({"sub": str(user.id), "role": user.role, "username": user.username})
    resp = RedirectResponse("/game/", status_code=302)
    resp.set_cookie("user_token", token, httponly=True, max_age=86400)
    return resp


@router.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("client/register.html", {"request": request, "error": None})


@router.post("/register")
def register(request: Request, username: str = Form(...), email: str = Form(...),
             password: str = Form(...), db: Session = Depends(get_db)):
    if user_repo.get_by_username(db, username):
        return templates.TemplateResponse("client/register.html", {
            "request": request, "error": "Tên đăng nhập đã tồn tại"
        })
    if user_repo.get_by_email(db, email):
        return templates.TemplateResponse("client/register.html", {
            "request": request, "error": "Email đã được sử dụng"
        })
    user_repo.create(db, username=username, email=email,
                     password_hash=hash_password(password), role="user")
    return RedirectResponse("/auth/login", status_code=302)


@router.get("/logout")
def logout():
    resp = RedirectResponse("/auth/login", status_code=302)
    resp.delete_cookie("user_token")
    return resp
