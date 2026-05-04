"""
Test nhanh AI sau khi train.
Chạy: python -m scripts.quick_test data/checkpoints/phase1_offline.pkl
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from app.ml.q_agent import QAgent
from app.ml.evaluator import eval_vs_random


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.quick_test <checkpoint_path>")
        print("Example: python -m scripts.quick_test data/checkpoints/phase1_offline.pkl")
        return
    
    checkpoint_path = sys.argv[1]
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ File không tồn tại: {checkpoint_path}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    agent = QAgent(epsilon=0.0)
    agent.load(checkpoint_path)
    
    print(f"Q-table size: {agent.q_table_size:,} states")
    print(f"Testing with 200 games vs random agent...")
    
    result = eval_vs_random(agent, n_games=200)
    
    print(f"\n📊 Results:")
    print(f"   Win rate:  {result['win_rate']:.1%}")
    print(f"   Wins:      {result['wins']}")
    print(f"   Losses:    {result['losses']}")
    print(f"   Draws:     {result['draws']}")
    
    if result['win_rate'] >= 0.8:
        print("\n✅ AI rất mạnh!")
    elif result['win_rate'] >= 0.65:
        print("\n👍 AI khá tốt, tiếp tục train sẽ tốt hơn.")
    elif result['win_rate'] >= 0.55:
        print("\n⚠️  AI hơi yếu, cần train thêm hoặc kiểm tra dữ liệu.")
    else:
        print("\n❌ AI chưa học được gì, cần kiểm tra lại code hoặc dữ liệu.")


if __name__ == "__main__":
    main()
