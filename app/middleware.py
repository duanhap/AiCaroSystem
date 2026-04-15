from fastapi import Request
from fastapi.responses import RedirectResponse
from jose import jwt, JWTError
from app.config import settings

ADMIN_PUBLIC = {"/admin/login"}

async def admin_auth_middleware(request: Request, call_next):
    path = request.url.path

    # Chỉ bảo vệ route /admin/ (trừ /admin/login)
    if path.startswith("/admin") and path not in ADMIN_PUBLIC:
        token = request.cookies.get("admin_token")
        if not token:
            return RedirectResponse("/admin/login", status_code=302)
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            if payload.get("role") != "admin":
                return RedirectResponse("/admin/login", status_code=302)
        except JWTError:
            return RedirectResponse("/admin/login", status_code=302)

    return await call_next(request)
