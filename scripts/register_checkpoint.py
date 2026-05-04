"""
Đăng ký file pkl có sẵn vào DB khi lưu bị lỗi.
Chạy: python -m scripts.register_checkpoint --file data/checkpoints/Q_v3.pkl --version Q_v3
"""
import sys, os, argparse
sys.path.insert(0, os.path.abspath('.'))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file",     required=True, help="Đường dẫn file pkl, vd: data/checkpoints/Q_v3.pkl")
    p.add_argument("--version",  default="",    help="Tên version, vd: Q_v3. Bỏ trống = tự đặt tên mới")
    p.add_argument("--episodes", type=int, default=0)
    p.add_argument("--winrate",  type=float, default=None)
    p.add_argument("--base",     default=None)
    args = p.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ File không tồn tại: {args.file}")
        return

    from app.database import SessionLocal
    from app.ml.q_agent import QAgent
    from app.ml.evaluator import eval_vs_random
    from app.repositories import checkpoint_repo

    db = SessionLocal()
    try:
        # Tự đặt version nếu không truyền
        if args.version:
            version = args.version
            # Kiểm tra trùng
            if checkpoint_repo.get_by_version(db, version):
                print(f"⚠️  Version '{version}' đã tồn tại trong DB.")
                version = f"{version}_re"
                print(f"   Dùng version mới: {version}")
        else:
            from app.services.checkpoint_service import get_next_version
            version = get_next_version(db)

        # Load agent để đánh giá nếu chưa có win rate
        win_rate = args.winrate
        if win_rate is None:
            print("Loading agent để đánh giá win rate (200 games)...")
            agent = QAgent(epsilon=0.0)
            agent.load(args.file)
            print(f"  Q-table size: {agent.q_table_size:,} states")
            win_rate = eval_vs_random(agent, n_games=200)["win_rate"]
            print(f"  Win rate vs random: {win_rate:.1%}")

        # Lưu vào DB
        abs_path = os.path.abspath(args.file)
        checkpoint_repo.create(
            db,
            version=version,
            file_path=abs_path,
            base_version=args.base,
            train_mode="selfplay",
            episodes_trained=args.episodes,
            win_rate_vs_random=win_rate,
            metadata_={"registered_manually": True}
        )

        print(f"\n✅ Đã đăng ký vào DB: {version}")
        print(f"   File: {abs_path}")
        print(f"   Win rate: {win_rate:.1%}")
        print(f"   ➡️  Vào Admin UI → Deploy '{version}'")

    finally:
        db.close()

if __name__ == "__main__":
    main()
