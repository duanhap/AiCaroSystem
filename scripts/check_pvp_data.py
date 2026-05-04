"""
Kiểm tra dữ liệu PvP đã tạo
Chạy: python scripts/check_pvp_data.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.models.game import Game
from app.models.game_step import GameStep

def check_pvp_data():
    db = SessionLocal()
    try:
        # Đếm số games PvP
        pvp_games = db.query(Game).filter(Game.mode == "pvp").all()
        print(f"📊 Tổng số games PvP: {len(pvp_games)}")
        
        if pvp_games:
            finished_games = [g for g in pvp_games if g.status == "finished"]
            print(f"✅ Games hoàn thành: {len(finished_games)}")
            
            # Thống kê winner
            winners = {}
            for game in finished_games:
                winner = game.winner or "draw"
                winners[winner] = winners.get(winner, 0) + 1
            print(f"🏆 Kết quả: {winners}")
            
            # Đếm tổng số steps
            total_steps = db.query(GameStep).join(Game).filter(Game.mode == "pvp").count()
            print(f"🎯 Tổng số moves: {total_steps}")
            
            if finished_games:
                avg_moves = total_steps / len(finished_games)
                print(f"📈 Trung bình moves/game: {avg_moves:.1f}")
                
                print(f"\n✅ Dữ liệu PvP sẵn sàng cho training!")
            else:
                print(f"\n⚠ Chưa có games hoàn thành")
        else:
            print("❌ Chưa có dữ liệu PvP nào")
            
    finally:
        db.close()

if __name__ == "__main__":
    check_pvp_data()