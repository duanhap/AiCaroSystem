"""
Script test kết nối DB và tạo toàn bộ bảng
Chạy: python scripts/init_db.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import engine, Base
from app.models import user, game, game_step, checkpoint, training_log  # noqa: import để Base nhận diện

def init():
    print("Đang kết nối MySQL...")
    try:
        with engine.connect() as conn:
            print(f"✅ Kết nối thành công: {engine.url.host}:{engine.url.port}/{engine.url.database}")
    except Exception as e:
        print(f"❌ Kết nối thất bại: {e}")
        sys.exit(1)

    print("Đang tạo bảng...")
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Tạo bảng thành công:")
        for table in Base.metadata.tables:
            print(f"   - {table}")
    except Exception as e:
        print(f"❌ Tạo bảng thất bại: {e}")
        sys.exit(1)

if __name__ == "__main__":
    init()
