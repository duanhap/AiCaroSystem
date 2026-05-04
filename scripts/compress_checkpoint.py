"""
Nén Q-table bằng cách giữ lại các states quan trọng nhất.
Chạy: python -m scripts.compress_checkpoint --version Q_v7 --keep 600000
"""
import sys, os, argparse
sys.path.insert(0, os.path.abspath('.'))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True, help="Version cần nén, vd: Q_v7")
    p.add_argument("--keep", type=int, default=600000, help="Số states giữ lại (default: 600k)")
    p.add_argument("--out", default="", help="Version output, mặc định: Q_v7_compressed")
    args = p.parse_args()

    import numpy as np
    from app.ml.q_agent import QAgent
    from app.ml.evaluator import eval_vs_random

    pkl_in = f"data/checkpoints/{args.version}.pkl"
    if not os.path.exists(pkl_in):
        print(f"❌ Không tìm thấy {pkl_in}")
        return

    print(f"Loading {args.version}...")
    agent = QAgent(epsilon=0.0)
    agent.load(pkl_in)
    original_size = agent.q_table_size
    print(f"  Q-table size: {original_size:,} states")
    print(f"  File size: {os.path.getsize(pkl_in)/1024/1024:.0f} MB")

    if original_size <= args.keep:
        print(f"  Đã nhỏ hơn {args.keep:,}, không cần nén.")
        return

    # Tính "tầm quan trọng" của mỗi state = max |Q-value|
    print(f"\nPruning xuống {args.keep:,} states...")
    keys = list(agent.q_table.keys())
    scores = np.array([float(np.abs(agent.q_table[k]).max()) for k in keys])

    # Giữ lại top-K states có Q-value lớn nhất
    top_indices = np.argpartition(scores, -args.keep)[-args.keep:]
    new_qtable = {keys[i]: agent.q_table[keys[i]] for i in top_indices}
    agent.q_table = new_qtable

    print(f"  Sau pruning: {agent.q_table_size:,} states")

    # Đánh giá sau pruning
    print("\nEvaluating (200 games)...")
    wr = eval_vs_random(agent, n_games=200)["win_rate"]
    print(f"  Win rate vs random: {wr:.1%}")

    # Lưu file mới
    out_version = args.out or f"{args.version}_compressed"
    pkl_out = f"data/checkpoints/{out_version}.pkl"
    agent.save(pkl_out)
    size_out = os.path.getsize(pkl_out)/1024/1024
    print(f"\n✅ Saved: {pkl_out} ({size_out:.0f} MB)")
    print(f"   Giảm từ {os.path.getsize(pkl_in)/1024/1024:.0f} MB → {size_out:.0f} MB")

    # Đăng ký vào DB
    print("\nĐăng ký vào DB...")
    from app.database import SessionLocal
    from app.repositories import checkpoint_repo
    db = SessionLocal()
    try:
        existing = checkpoint_repo.get_by_version(db, out_version)
        if existing:
            print(f"  Version '{out_version}' đã tồn tại trong DB.")
        else:
            checkpoint_repo.create(
                db,
                version=out_version,
                file_path=os.path.abspath(pkl_out),
                base_version=args.version,
                train_mode="selfplay",
                episodes_trained=0,
                win_rate_vs_random=wr,
                metadata_={"compressed_from": args.version, "kept_states": args.keep}
            )
            print(f"  ✅ Đã đăng ký: {out_version}")
    finally:
        db.close()

    print(f"\n➡️  Vào Admin UI → Deploy '{out_version}'")

if __name__ == "__main__":
    main()
