"""
Tạo dummy users cho PvP data generation
Chạy: python scripts/create_dummy_users.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.repositories import user_repo
from app.services.auth_service import hash_password

def create_dummy_users():
    db = SessionLocal()
    try:
        # Tạo 2 dummy users cho PvP data
        users_to_create = [
            {"username": "pvp_player1", "email": "player1@aicaro.local", "password": "dummy123"},
            {"username": "pvp_player2", "email": "player2@aicaro.local", "password": "dummy123"},
        ]
        
        for user_data in users_to_create:
            existing = user_repo.get_by_username(db, user_data["username"])
            if existing:
                print(f"⚠ User '{user_data['username']}' đã tồn tại (ID: {existing.id})")
                continue
                
            user = user_repo.create(
                db,
                username=user_data["username"],
                email=user_data["email"],
                password_hash=hash_password(user_data["password"]),
                role="user",
            )
            print(f"✅ Tạo user thành công: {user.username} (ID: {user.id})")
            
        print("\n✅ Dummy users sẵn sàng cho PvP data generation!")
        
    finally:
        db.close()

if __name__ == "__main__":
    create_dummy_users()