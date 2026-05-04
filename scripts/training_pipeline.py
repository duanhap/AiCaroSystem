"""
Pipeline huấn luyện AI từ đầu đến cuối.
Chạy: python -m scripts.training_pipeline
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from app.database import SessionLocal
from app.ml.q_agent import QAgent
from app.ml.self_play import run_self_play
from app.ml.offline_train import run_offline_train, load_game_data_from_db
from app.ml.evaluator import eval_vs_random
from app.repositories import game_repo
from app.services.checkpoint_service import save_checkpoint, get_next_version
import copy
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def phase1_offline_training(db, min_games=50):
    """
    Giai đoạn 1: Offline training từ dữ liệu người chơi.
    Cần ít nhất 50-100 ván chất lượng.
    """
    logger.info("=== PHASE 1: OFFLINE TRAINING ===")
    
    # Lấy tất cả game_ids từ DB
    from app.models.game import Game
    games = db.query(Game).filter(Game.status == 'finished').all()
    game_ids = [g.id for g in games]
    
    if len(game_ids) < min_games:
        logger.warning(f"Chỉ có {len(game_ids)} ván, cần ít nhất {min_games} ván")
        logger.warning("Hãy chơi thêm hoặc nhờ bạn bè chơi để thu thập dữ liệu")
        return None
    
    logger.info(f"Tìm thấy {len(game_ids)} ván, bắt đầu offline training...")
    
    agent = QAgent(alpha=0.2, gamma=0.9, epsilon=0.0, use_symmetry=True)
    game_data = load_game_data_from_db(db, game_ids)
    
    result = run_offline_train(
        agent=agent,
        game_data=game_data,
        test_games=200,
        on_progress=lambda p: logger.info(f"  Progress: {p['game_idx']}/{p['total']}")
    )
    
    logger.info(f"✅ Offline training done!")
    logger.info(f"   Win rate vs random: {result['win_rate_vs_random']:.1%}")
    
    if result['win_rate_vs_random'] < 0.55:
        logger.warning("⚠️  Win rate < 55%, dữ liệu có thể yếu. Cần chơi cẩn thận hơn.")
    
    version = get_next_version(db)
    save_checkpoint(
        db, agent, version=version, train_mode="offline",
        win_rate_vs_random=result['win_rate_vs_random'],
        metadata={"phase": "phase1_offline"}
    )
    logger.info(f"   Saved to DB as version: {version}")
    return agent


def phase2_selfplay_explore(agent, db, episodes=10000):
    """
    Giai đoạn 2: Self-play với epsilon cao để khám phá.
    """
    logger.info("=== PHASE 2: SELF-PLAY EXPLORATION ===")
    
    agent.epsilon = 1.0
    agent.epsilon_min = 0.2
    agent.epsilon_decay = 0.9995
    
    result = run_self_play(
        agent=agent,
        episodes=episodes,
        opponent=None,  # tự đấu
        test_interval=500,
        test_games=200,
        win_rate_target=0.85,
        compare_agent=None,
        on_progress=lambda log: logger.info(
            f"  Ep {log['episode']}/{episodes} | ε={log['epsilon']:.3f} | "
            f"Q-size={log['q_table_size']:,} | WR={(log.get('win_rate_vs_random') or 0):.1%}"
        ),
        convergence_threshold=0.001
    )
    
    logger.info(f"✅ Self-play exploration done!")
    logger.info(f"   Episodes trained: {result['episodes_trained']}")
    logger.info(f"   Win rate vs random: {result['win_rate_vs_random']:.1%}")
    
    if result.get('stopped_early'):
        logger.info(f"   Stopped early: {result['stop_reason']}")
    
    version = get_next_version(db)
    save_checkpoint(
        db, agent, version=version, train_mode="selfplay",
        episodes_trained=result['episodes_trained'],
        win_rate_vs_random=result['win_rate_vs_random'],
        metadata={"phase": "phase2_explore"}
    )
    logger.info(f"   Saved to DB as version: {version}")
    return agent


def phase3_selfplay_vs_old(agent, db, episodes=20000):
    """
    Giai đoạn 3: Self-play vs phiên bản cũ để ổn định.
    """
    logger.info("=== PHASE 3: SELF-PLAY VS OLD VERSION ===")
    
    old_agent = copy.deepcopy(agent)
    old_agent.epsilon = 0.0
    
    agent.alpha = 0.15
    agent.gamma = 0.95
    agent.epsilon = 0.3
    agent.epsilon_min = 0.05
    agent.epsilon_decay = 0.999
    
    result = run_self_play(
        agent=agent,
        episodes=episodes,
        opponent=old_agent,
        test_interval=1000,
        test_games=200,
        win_rate_target=0.9,
        compare_agent=old_agent,
        on_progress=lambda log: logger.info(
            f"  Ep {log['episode']}/{episodes} | ε={log['epsilon']:.3f} | "
            f"WR_random={(log.get('win_rate_vs_random') or 0):.1%} | "
            f"WR_old={(log.get('win_rate_vs_version') or 0):.1%}"
        ),
        convergence_threshold=0.001
    )
    
    logger.info(f"✅ Self-play vs old done!")
    logger.info(f"   Episodes trained: {result['episodes_trained']}")
    logger.info(f"   Win rate vs random: {result['win_rate_vs_random']:.1%}")
    logger.info(f"   Win rate vs old: {result.get('win_rate_vs_version', 0):.1%}")
    
    version = get_next_version(db)
    save_checkpoint(
        db, agent, version=version, train_mode="selfplay",
        episodes_trained=result['episodes_trained'],
        win_rate_vs_random=result['win_rate_vs_random'],
        win_rate_vs_version=result.get('win_rate_vs_version'),
        metadata={"phase": "phase3_stable"}
    )
    logger.info(f"   Saved to DB as version: {version}")
    return agent


def final_evaluation(agent, db):
    """
    Đánh giá cuối cùng với 1000 ván.
    """
    logger.info("=== FINAL EVALUATION ===")
    logger.info("Running 1000 games vs random agent...")
    
    result = eval_vs_random(agent, n_games=1000)
    
    logger.info(f"🎉 FINAL RESULTS:")
    logger.info(f"   Win rate:  {result['win_rate']:.1%}")
    logger.info(f"   Wins:      {result['wins']}")
    logger.info(f"   Losses:    {result['losses']}")
    logger.info(f"   Draws:     {result['draws']}")
    logger.info(f"   Q-table size: {agent.q_table_size:,} states")
    
    version = get_next_version(db)
    save_checkpoint(
        db, agent, version=version, train_mode="selfplay",
        win_rate_vs_random=result['win_rate'],
        metadata={"phase": "final", "eval_games": 1000}
    )
    logger.info(f"   Saved to DB as version: {version}")
    logger.info(f"   ➡️  Go to Admin UI to deploy version '{version}'")


def main():
    db = SessionLocal()
    
    try:
        # Phase 1: Offline training
        agent = phase1_offline_training(db, min_games=50)
        if agent is None:
            logger.error("❌ Không đủ dữ liệu để train. Hãy chơi thêm ít nhất 50 ván.")
            return
        
        # Phase 2: Self-play exploration
        agent = phase2_selfplay_explore(agent, db, episodes=10000)
        
        # Phase 3: Self-play vs old
        agent = phase3_selfplay_vs_old(agent, db, episodes=20000)
        
        # Final evaluation
        final_evaluation(agent, db)
        
        logger.info("\n✅ Training pipeline hoàn tất!")
        logger.info("Vào Admin UI để deploy checkpoint vừa tạo.")
        
    finally:
        db.close()


if __name__ == "__main__":
    main()
