"""
CLI Training Script - Tương đương với web UI nhưng chạy từ terminal.
Hỗ trợ đầy đủ các thông số như trên web, tự lưu vào DB sau khi train xong.

Ví dụ:
    # Self-play từ đầu
    python -m scripts.train_cli --episodes 500000

    # Fine-tune từ checkpoint cũ
    python -m scripts.train_cli --base Q_v40 --episodes 200000 --epsilon-start 0.3

    # Train vs version cũ (stable training)
    python -m scripts.train_cli --base Q_v40 --opponent Q_v38 --compare Q_v38 --episodes 100000

    # Tùy chỉnh đầy đủ
    python -m scripts.train_cli \\
        --base Q_v40 \\
        --opponent self \\
        --episodes 500000 \\
        --alpha 0.1 \\
        --gamma 0.9 \\
        --epsilon-start 1.0 \\
        --epsilon-min 0.01 \\
        --epsilon-decay 0.995 \\
        --test-interval 200 \\
        --test-games 100 \\
        --win-rate-target 0.9
"""
import sys
import os
import argparse
import copy
import logging

sys.path.insert(0, os.path.abspath('.'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="AiCaro CLI Trainer")

    # Checkpoint
    p.add_argument("--base", default="",
                   help="Version checkpoint để fine-tune (vd: Q_v40). Bỏ trống = train từ đầu")
    p.add_argument("--opponent", default="self",
                   help="Đối thủ khi train: 'self' = tự đấu, tên version = đấu vs checkpoint đó (vd: Q_v38)")
    p.add_argument("--compare", default="",
                   help="Version để so sánh khi test (vd: Q_v38). Bỏ trống = chỉ test vs random")

    # Hyperparameters
    p.add_argument("--episodes", type=int, default=50000, help="Số episodes (default: 50000)")
    p.add_argument("--alpha", type=float, default=0.1, help="Learning rate (default: 0.1)")
    p.add_argument("--gamma", type=float, default=0.9, help="Discount factor (default: 0.9)")
    p.add_argument("--epsilon-start", type=float, default=1.0, help="Epsilon ban đầu (default: 1.0)")
    p.add_argument("--epsilon-min", type=float, default=0.01, help="Epsilon tối thiểu (default: 0.01)")
    p.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay (default: 0.995)")

    # Test settings
    p.add_argument("--test-interval", type=int, default=200, help="Test mỗi N episodes (default: 200)")
    p.add_argument("--test-games", type=int, default=100, help="Số ván mỗi lần test (default: 100)")
    p.add_argument("--win-rate-target", type=float, default=0.9, help="Early stop khi đạt win rate này (default: 0.9)")

    # Convergence
    p.add_argument("--convergence", type=float, default=0.0,
                   help="Dừng khi avg Q delta < ngưỡng này. 0 = tắt (default: 0)")
    p.add_argument("--convergence-streak", type=int, default=3,
                   help="Số lần liên tiếp hội tụ để dừng (default: 3)")

    return p.parse_args()


def main():
    args = parse_args()

    from app.database import SessionLocal
    from app.ml.q_agent import QAgent
    from app.ml.self_play import run_self_play
    from app.ml.evaluator import eval_vs_random
    from app.services.checkpoint_service import save_checkpoint, get_next_version, load_agent
    from app.repositories import checkpoint_repo

    db = SessionLocal()

    try:
        # ── 1. Load base agent ──────────────────────────────────────────────
        if args.base:
            logger.info(f"Loading base checkpoint: {args.base}")
            agent = load_agent(db, args.base)
            logger.info(f"  Q-table size: {agent.q_table_size:,} states")
        else:
            logger.info("Starting from scratch (no base checkpoint)")
            agent = QAgent(use_symmetry=True)

        # Áp dụng hyperparams
        agent.alpha         = args.alpha
        agent.gamma         = args.gamma
        agent.epsilon       = args.epsilon_start
        agent.epsilon_min   = args.epsilon_min
        agent.epsilon_decay = args.epsilon_decay

        # ── 2. Load opponent ────────────────────────────────────────────────
        opponent = None
        compare_agent = None

        if args.opponent and args.opponent.lower() != "self":
            logger.info(f"Loading opponent checkpoint: {args.opponent}")
            opponent = load_agent(db, args.opponent)
            opponent.epsilon = 0.0
            logger.info(f"  Opponent Q-table size: {opponent.q_table_size:,} states")
        else:
            logger.info("Opponent: self-play")

        if args.compare:
            logger.info(f"Loading compare checkpoint: {args.compare}")
            compare_agent = load_agent(db, args.compare)
            compare_agent.epsilon = 0.0

        # ── 3. Print config ─────────────────────────────────────────────────
        logger.info("=" * 55)
        logger.info("  TRAINING CONFIG")
        logger.info("=" * 55)
        logger.info(f"  Base:          {args.base or '(scratch)'}")
        logger.info(f"  Opponent:      {args.opponent}")
        logger.info(f"  Compare:       {args.compare or '(none)'}")
        logger.info(f"  Episodes:      {args.episodes:,}")
        logger.info(f"  Alpha:         {args.alpha}")
        logger.info(f"  Gamma:         {args.gamma}")
        logger.info(f"  Epsilon:       {args.epsilon_start} → {args.epsilon_min} (decay {args.epsilon_decay})")
        logger.info(f"  Test interval: every {args.test_interval} eps, {args.test_games} games")
        logger.info(f"  Win rate target: {args.win_rate_target:.0%}")
        logger.info(f"  Convergence:   {'OFF' if args.convergence == 0 else f'threshold={args.convergence}, streak={args.convergence_streak}'}")
        logger.info("=" * 55)

        # ── 4. Run self-play ────────────────────────────────────────────────
        def on_progress(log):
            ep       = log["episode"]
            eps      = log["epsilon"]
            q_size   = log["q_table_size"]
            wr       = log.get("win_rate_vs_random")
            wrv      = log.get("win_rate_vs_version")
            delta    = log.get("avg_q_delta")

            if log.get("heartbeat"):
                print(f"  [{ep:>7,}] ε={eps:.4f} | Q={q_size:>7,} states", flush=True)
            else:
                parts = [f"  [{ep:>7,}] ε={eps:.4f} | Q={q_size:>7,}"]
                if wr  is not None: parts.append(f"vs_random={wr:.1%}")
                if wrv is not None: parts.append(f"vs_version={wrv:.1%}")
                if delta is not None: parts.append(f"Δq={delta:.6f}")
                print(" | ".join(parts), flush=True)

        result = run_self_play(
            agent=agent,
            episodes=args.episodes,
            opponent=opponent,
            test_interval=args.test_interval,
            test_games=args.test_games,
            win_rate_target=args.win_rate_target,
            compare_agent=compare_agent,
            on_progress=on_progress,
            convergence_threshold=args.convergence,
            convergence_streak=args.convergence_streak,
        )

        # ── 5. Final eval ───────────────────────────────────────────────────
        logger.info("\nRunning final evaluation (200 games)...")
        final = eval_vs_random(agent, n_games=200)

        logger.info("=" * 55)
        logger.info("  RESULTS")
        logger.info("=" * 55)
        logger.info(f"  Episodes trained: {result['episodes_trained']:,}")
        logger.info(f"  Win rate vs random: {final['win_rate']:.1%}")
        logger.info(f"  Wins: {final['wins']} | Losses: {final['losses']} | Draws: {final['draws']}")
        logger.info(f"  Q-table size: {agent.q_table_size:,} states")
        if result.get("stopped_early"):
            logger.info(f"  Stopped early: {result['stop_reason']}")
        logger.info("=" * 55)

        # ── 6. Save to DB ───────────────────────────────────────────────────
        # Tạo session mới hoàn toàn vì connection cũ đã timeout sau train lâu
        try:
            db.close()
        except Exception:
            pass

        db2 = SessionLocal()
        try:
            version = get_next_version(db2)
            save_checkpoint(
                db2, agent,
                version=version,
                train_mode="selfplay",
                base_version=args.base or None,
                opponent_version=args.opponent if args.opponent != "self" else None,
                compared_version=args.compare or None,
                episodes_trained=result["episodes_trained"],
                win_rate_vs_random=final["win_rate"],
                win_rate_vs_version=result.get("win_rate_vs_version"),
                metadata={
                    "alpha": args.alpha,
                    "gamma": args.gamma,
                    "epsilon_start": args.epsilon_start,
                    "epsilon_min": args.epsilon_min,
                    "epsilon_decay": args.epsilon_decay,
                    "stop_reason": result.get("stop_reason", "completed"),
                }
            )
            logger.info(f"\n✅ Saved to DB as version: {version}")
            logger.info(f"   ➡️  Vào Admin UI → Checkpoints → Deploy '{version}'")
        finally:
            db2.close()

    finally:
        try:
            db.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
