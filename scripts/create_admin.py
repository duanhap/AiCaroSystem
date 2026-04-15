"""
Tạo tài khoản admin
Chạy: python scripts/create_admin.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.repositories import user_repo
from app.services.auth_service import hash_password

def create_admin(username: str, email: str, password: str):
    db = SessionLocal()
    try:
        # Kiểm tra đã tồn tại chưa
        existing = user_repo.get_by_username(db, username)
        if existing:
            print(f"⚠ User '{username}' đã tồn tại (role: {existing.role})")
            if existing.role != "admin":
                existing.role = "admin"
                db.commit()
                print(f"✅ Đã nâng quyền '{username}' lên admin")
            return

        user = user_repo.create(
            db,
            username=username,
            email=email,
            password_hash=hash_password(password),
            role="admin",
        )
        print(f"✅ Tạo admin thành công!")
        print(f"   Username : {user.username}")
        print(f"   Email    : {user.email}")
        print(f"   Role     : {user.role}")
        print(f"\n→ Đăng nhập tại: http://localhost:8000/admin/login")
    finally:
        db.close()

if __name__ == "__main__":
    create_admin(
        username="admin",
        email="admin@aicaro.local",
        password="Admin@123",
    )
