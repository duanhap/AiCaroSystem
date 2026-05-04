"""
Train AI với dữ liệu PvP giả lập chất lượng cao.
Chạy: python -m scripts.train_with_pvp_data
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from app.database import SessionLocal
from app.ml.q_agent import QAgent
from app.ml.offline_train import run_offline_train, load_game_data_from_db
from app.ml.self_play import run_self_play
from app.ml.evaluator import eval_vs_random
from app.models.game import Game
from app.services.checkpoint_service import save_checkpoint, get_next_version
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def train_with_pvp_data():
    logger.info("=== TRAINING WITH PvP DATA ===")
    
    db = SessionLocal()
    
    try:
        # 1. Lấy dữ liệu PvP từ DB
        logger.info("Loading PvP games from database...")
        pvp_games = db.query(Game).filter(Game.mode == 'pvp', Game.status == 'finished').all()
        
        if len(pvp_games) < 50:
            logger.warning(f"Only {len(pvp_games)} PvP games found. Need at least 50.")
            logger.info("Run 'python -m scripts.generate_pvp_data' first!")
            return
        
        game_ids = [g.id for g in pvp_games]
        logger.info(f"Found {len(game_ids)} PvP games for training")
        
        # 2. Load game data
        game_data = load_game_data_from_db(db, game_ids)
        logger.info(f"Loaded {len(game_data)} games with steps")
        
        # 3. Phase 1: Offline training từ PvP data
        logger.info("\n=== PHASE 1: OFFLINE TRAINING FROM PvP DATA ===")
        
        agent = QAgent(
            alpha=0.2,
            gamma=0.9, 
            epsilon=0.0,  # Không cần exploration trong offline training
            use_symmetry=True
        )
        
        result = run_offline_train(
            agent=agent,
            game_data=game_data,
            test_games=200,
            on_progress=lambda p: logger.info(f"  Progress: {p['game_idx']}/{p['total']}")
        )
        
        logger.info(f"✅ Offline training completed!")
        logger.info(f"   Win rate vs random: {result['win_rate_vs_random']:.1%}")
        
        # Save checkpoint to DB
        version_offline = get_next_version(db)
        save_checkpoint(
            db, agent, version=version_offline, train_mode="offline",
            win_rate_vs_random=result['win_rate_vs_random'],
            metadata={"phase": "pvp_offline", "pvp_games": len(game_ids)}
        )
        logger.info(f"   Saved to DB as version: {version_offline}")
        
        # 4. Phase 2: Self-play refinement
        logger.info("\n=== PHASE 2: SELF-PLAY REFINEMENT ===")
        
        # Điều chỉnh tham số cho self-play
        agent.epsilon = 0.5      # Moderate exploration
        agent.epsilon_min = 0.1
        agent.epsilon_decay = 0.999
        agent.alpha = 0.15       # Slower learning để không quên kiến thức PvP
        
        result = run_self_play(
            agent=agent,
            episodes=5000,
            opponent=None,
            test_interval=500,
            test_games=200,
            win_rate_target=0.85,
            compare_agent=None,
            on_progress=lambda log: logger.info(
                f"  Ep {log['episode']:4d} | ε={log['epsilon']:.3f} | "
                f"Q={log['q_table_size']:5,} | WR={(log.get('win_rate_vs_random') or 0):5.1%} | "
                f"Δ={(log.get('avg_q_delta') or 0):.6f}"
            ),
            convergence_threshold=0.001
        )
        
        logger.info(f"✅ Self-play refinement completed!")
        logger.info(f"   Episodes trained: {result['episodes_trained']}")
        logger.info(f"   Final win rate: {result['win_rate_vs_random']:.1%}")
        
        if result.get('stopped_early'):
            logger.info(f"   Stopped early: {result['stop_reason']}")
        
        # 5. Final evaluation
        logger.info("\n=== FINAL EVALUATION ===")
        
        # Test overall performance (agent plays both X and O)
        logger.info("Testing overall performance (both X and O roles)...")
        final_result = eval_vs_random(agent, n_games=200)
        
        logger.info(f"   Overall: {final_result['win_rate']:.1%} win rate ({final_result['wins']}/{final_result['wins'] + final_result['losses'] + final_result['draws']})")
        logger.info(f"   Wins: {final_result['wins']}, Losses: {final_result['losses']}, Draws: {final_result['draws']}")
        
        # Overall performance
        overall_rate = final_result['win_rate']
        
        # 6. Q-value analysis
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
        
        # 7. Save final checkpoint to DB
        version_final = get_next_version(db)
        save_checkpoint(
            db, agent, version=version_final, train_mode="selfplay",
            base_version=version_offline,
            episodes_trained=result['episodes_trained'],
            win_rate_vs_random=overall_rate,
            metadata={
                "phase": "pvp_final",
                "pvp_games": len(game_ids),
                "q_positive_pct": round(positive / len(all_q_values) * 100, 1),
                "q_negative_pct": round(negative / len(all_q_values) * 100, 1),
            }
        )
        logger.info(f"\n💾 Saved final checkpoint to DB as version: {version_final}")
        logger.info(f"   ➡️  Go to Admin UI to deploy version '{version_final}'")
        
        # 8. Summary
        logger.info(f"\n🎉 TRAINING SUMMARY:")
        logger.info(f"   📊 PvP games used: {len(game_ids)}")
        logger.info(f"   🎯 Offline win rate: {result['win_rate_vs_random']:.1%}")
        logger.info(f"   🚀 Final win rate: {overall_rate:.1%}")
        logger.info(f"   🧠 Q-table size: {agent.q_table_size:,} states")
        logger.info(f"   ⚖️  Q-value distribution: {positive/len(all_q_values)*100:.1f}% pos, {negative/len(all_q_values)*100:.1f}% neg")
        logger.info(f"   🏷️  DB version: {version_final}")
        
        if overall_rate > 0.75:
            logger.info("   ✅ EXCELLENT: AI performance is very good!")
        elif overall_rate > 0.65:
            logger.info("   ✅ GOOD: AI performance is acceptable!")
        else:
            logger.info("   ⚠️  NEEDS IMPROVEMENT: Consider more training or better rewards")
    
    finally:
        db.close()


if __name__ == "__main__":
    train_with_pvp_data()