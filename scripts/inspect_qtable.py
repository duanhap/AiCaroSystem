"""
Phân tích Q-table của một checkpoint
Chạy: python scripts/inspect_qtable.py Q_v8
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from collections import defaultdict

def inspect(version: str):
    path = f"data/checkpoints/{version}.pkl"
    if not os.path.exists(path):
        print(f"Không tìm thấy {path}")
        return

    with open(path, "rb") as f:
        data = pickle.load(f)

    total_states = len(data)
    total_entries = sum(len(v) for v in data.values())
    all_vals = [v for actions in data.values() for v in actions.values()]

    print(f"\n=== Q-table: {version} ===")
    print(f"Số states đã học:     {total_states:,}")
    print(f"Tổng (state,action):  {total_entries:,}")
    print(f"Q-value trung bình:   {sum(all_vals)/len(all_vals):.4f}")
    print(f"Q-value max:          {max(all_vals):.4f}")
    print(f"Q-value min:          {min(all_vals):.4f}")

    # Phân phối Q-values
    pos = sum(1 for v in all_vals if v > 0.1)
    neg = sum(1 for v in all_vals if v < -0.1)
    neu = len(all_vals) - pos - neg
    print(f"\nPhân phối Q-values:")
    print(f"  Dương (>0.1):  {pos:,} ({pos/len(all_vals):.1%}) ← nước tốt")
    print(f"  Âm (<-0.1):    {neg:,} ({neg/len(all_vals):.1%}) ← nước xấu")
    print(f"  Trung tính:    {neu:,} ({neu/len(all_vals):.1%}) ← chưa học")

    # Số action trung bình mỗi state
    avg_actions = total_entries / total_states if total_states else 0
    print(f"\nSố action TB mỗi state: {avg_actions:.1f} / 49 ô")
    print(f"→ AI đã khám phá {avg_actions/49:.1%} không gian hành động")

if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "Q_v8"
    inspect(version)
