"""
Train AI từ đầu bằng self-play (không cần dữ liệu người chơi).
Chạy: python -m scripts.train_from_scratch
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


def main():
    logger.info("=== TRAINING AI FROM SCRATCH ===")
    logger.info("Không cần dữ liệu người chơi, AI sẽ tự học từ đầu")
    
    # Khởi tạo agent mới
    agent = QAgent(
        alpha=0.2,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.2,
        epsilon_decay=0.9995,
        use_symmetry=True
    )
    
    logger.info(f"Tham số: α={agent.alpha}, γ={agent.gamma}, ε={agent.epsilon}")
    logger.info("Bắt đầu self-play 20,000 episodes...")
    
    result = run_self_play(
        agent=agent,
        episodes=20000,
        opponent=None,
        test_interval=500,
        test_games=200,
        win_rate_target=0.85,
        compare_agent=None,
        on_progress=lambda log: logger.info(
            f"  Ep {log['episode']:5d} | ε={log['epsilon']:.3f} | "
            f"Q={log['q_table_size']:6,} | WR={(log.get('win_rate_vs_random') or 0):5.1%} | "
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
    
    # Final evaluation
    logger.info("\n=== FINAL EVALUATION (1000 games) ===")
    final = eval_vs_random(agent, n_games=1000)
    logger.info(f"   Win rate:  {final['win_rate']:.1%}")
    logger.info(f"   Wins:      {final['wins']}")
    logger.info(f"   Losses:    {final['losses']}")
    logger.info(f"   Draws:     {final['draws']}")
    
    # Save
    agent.save("data/checkpoints/trained_from_scratch.pkl")
    logger.info(f"\n💾 Saved: data/checkpoints/trained_from_scratch.pkl")
    logger.info("Bạn có thể deploy checkpoint này từ Admin UI")


if __name__ == "__main__":
    main()
