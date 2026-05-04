"""
Test AI với reward shaping mới.
Chạy: python -m scripts.test_new_rewards
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from app.ml.q_agent import QAgent
from app.ml.self_play import run_self_play
from app.ml.evaluator import eval_vs_random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def test_new_rewards():
    logger.info("=== TEST NEW REWARD SHAPING ===")
    logger.info("Training AI với defensive rewards và penalty...")
    
    # Agent mới với tham số conservative hơn
    agent = QAgent(
        alpha=0.3,           # Học nhanh hơn
        gamma=0.9,           # Ưu tiên reward ngắn hạn
        epsilon=1.0,         # Khám phá nhiều
        epsilon_min=0.1,     # Vẫn giữ exploration
        epsilon_decay=0.999, # Giảm chậm
        use_symmetry=True
    )
    
    logger.info(f"Tham số: α={agent.alpha}, γ={agent.gamma}, ε={agent.epsilon}")
    logger.info("Bắt đầu training 2000 episodes...")
    
    result = run_self_play(
        agent=agent,
        episodes=2000,
        opponent=None,
        test_interval=200,
        test_games=100,
        win_rate_target=0.8,  # Thấp hơn để dễ đạt
        compare_agent=None,
        on_progress=lambda log: logger.info(
            f"  Ep {log['episode']:4d} | ε={log['epsilon']:.3f} | "
            f"Q={log['q_table_size']:5,} | WR={(log.get('win_rate_vs_random') or 0):5.1%} | "
            f"Δ={(log.get('avg_q_delta') or 0):.6f}"
        ),
        convergence_threshold=0.001
    )
    
    logger.info(f"\n✅ Training done!")
    logger.info(f"   Episodes: {result['episodes_trained']}")
    logger.info(f"   Win rate: {result['win_rate_vs_random']:.1%}")
    logger.info(f"   Q-table size: {agent.q_table_size:,} states")
    
    if result.get('stopped_early'):
        logger.info(f"   Stopped early: {result['stop_reason']}")
    
    # Test chi tiết
    logger.info("\n=== DETAILED EVALUATION ===")
    
    # Test as X (đi trước)
    logger.info("Testing as X (first player)...")
    x_result = eval_vs_random(agent, n_games=50, agent_plays_x=True)
    logger.info(f"   As X: {x_result['win_rate']:.1%} win rate")
    
    # Test as O (đi sau)  
    logger.info("Testing as O (second player)...")
    o_result = eval_vs_random(agent, n_games=50, agent_plays_x=False)
    logger.info(f"   As O: {o_result['win_rate']:.1%} win rate")
    
    # Phân tích Q-values
    logger.info("\n=== Q-VALUE ANALYSIS ===")
    all_q_values = []
    for state_q in agent.q_table.values():
        all_q_values.extend(state_q.tolist())
    
    import numpy as np
    all_q_values = np.array(all_q_values)
    
    positive = np.sum(all_q_values > 0.1)
    negative = np.sum(all_q_values < -0.1)
    neutral = len(all_q_values) - positive - negative
    
    logger.info(f"   Total Q-values: {len(all_q_values):,}")
    logger.info(f"   Positive (>0.1): {positive:,} ({positive/len(all_q_values)*100:.1f}%)")
    logger.info(f"   Negative (<-0.1): {negative:,} ({negative/len(all_q_values)*100:.1f}%)")
    logger.info(f"   Neutral: {neutral:,} ({neutral/len(all_q_values)*100:.1f}%)")
    logger.info(f"   Min Q: {all_q_values.min():.4f}")
    logger.info(f"   Max Q: {all_q_values.max():.4f}")
    logger.info(f"   Mean Q: {all_q_values.mean():.4f}")
    
    # Save checkpoint
    agent.save("data/checkpoints/test_new_rewards.pkl")
    logger.info(f"\n💾 Saved: data/checkpoints/test_new_rewards.pkl")
    logger.info("Bạn có thể deploy checkpoint này từ Admin UI để test")


if __name__ == "__main__":
    test_new_rewards()