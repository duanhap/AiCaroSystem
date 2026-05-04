"""
Train AI mạnh hơn với chiến lược 3 giai đoạn tối ưu.
Chạy: python -m scripts.train_strong

Chiến lược:
  Phase 1 - Warm-up (200k eps): epsilon cao, học khám phá rộng
  Phase 2 - Focus (500k eps):   epsilon giảm dần, học khai thác
  Phase 3 - Polish (300k eps):  epsilon thấp, đấu vs bản thân tốt nhất

Tổng: ~1M episodes, tự lưu DB sau mỗi phase.
"""
import sys, os, copy, logging, time
sys.path.insert(0, os.path.abspath('.'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _new_db():
    """Luôn tạo engine + session mới để tránh timeout."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import sys, os
    sys.path.insert(0, os.path.abspath('.'))
    from app.config import settings
    eng = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
    Session = sessionmaker(bind=eng)
    return Session()


def _save(agent, phase_name, episodes_trained, win_rate, base_ver=None, meta=None):
    """Lưu checkpoint vào DB với retry khi bị timeout."""
    from app.services.checkpoint_service import save_checkpoint, get_next_version
    from app.config import settings
    import time, os

    for attempt in range(3):
        try:
            db = _new_db()
            version = get_next_version(db)
            save_checkpoint(
                db, agent,
                version=version,
                train_mode="selfplay",
                base_version=base_ver,
                episodes_trained=episodes_trained,
                win_rate_vs_random=win_rate,
                metadata={"phase": phase_name, **(meta or {})}
            )
            try: db.close()
            except: pass
            logger.info(f"  💾 Saved → DB version: {version}")
            return version
        except Exception as e:
            logger.warning(f"  ⚠️  DB attempt {attempt+1}/3 failed: {e}")
            try: db.close()
            except: pass
            if attempt < 2:
                time.sleep(5)

    # Fallback: pkl đã lưu bởi save_checkpoint, chỉ cần báo user đăng ký thủ công
    os.makedirs(settings.CHECKPOINT_DIR, exist_ok=True)
    fallback = os.path.join(settings.CHECKPOINT_DIR, f"{phase_name}_fallback.pkl")
    agent.save(fallback)
    logger.error(f"  ❌ DB lỗi 3 lần. PKL saved: {fallback}")
    logger.error(f"  ➡️  Chạy: python -m scripts.register_checkpoint --file {fallback} --episodes {episodes_trained}")
    return None


def _run_phase(agent, episodes, epsilon_start, epsilon_min, epsilon_decay,
               alpha, opponent=None, test_every=10000, test_games=200,
               phase_name="phase"):
    """
    Vòng lặp train tối ưu — không snapshot Q-table, không overhead.
    Trả về win_rate cuối.
    """
    from app.ml.environment import CaroEnv, X
    from app.ml.evaluator import eval_vs_random

    agent.alpha         = alpha
    agent.epsilon       = epsilon_start
    agent.epsilon_min   = epsilon_min
    agent.epsilon_decay = epsilon_decay

    env = CaroEnv()
    opp = opponent if opponent is not None else agent
    best_wr = 0.0
    t0 = time.time()

    for ep in range(1, episodes + 1):
        state = env.reset()
        done  = False
        agent_is_x = (ep % 2 == 1)

        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break

            is_agent_turn = (env.current_player == X) == agent_is_x
            current_player = env.current_player

            if is_agent_turn:
                action = agent.choose_action(state, valid, current_player=current_player)
                next_state, reward, done = env.step(action)
                next_valid = env.get_valid_actions() if not done else []
                agent.update(state, action, reward, next_state, next_valid, done)
            else:
                action = opp.choose_action(state, valid, current_player=current_player)
                next_state, reward, done = env.step(action)
                if opponent is None:
                    next_valid = env.get_valid_actions() if not done else []
                    agent_reward = -reward if done and reward == 1.0 else reward
                    agent.update(state, action, agent_reward, next_state, next_valid, done)
                elif done and reward == 1.0:
                    agent.update(state, action, -1.0, next_state, [], done)

            state = next_state

        agent.decay_epsilon()

        if ep % test_every == 0 or ep == episodes:
            wr = eval_vs_random(agent, test_games)["win_rate"]
            elapsed = time.time() - t0
            eps_per_sec = ep / elapsed
            eta = (episodes - ep) / eps_per_sec if eps_per_sec > 0 else 0
            best_wr = max(best_wr, wr)
            logger.info(
                f"  [{phase_name}] ep={ep:>7,}/{episodes:,} | "
                f"ε={agent.epsilon:.4f} | Q={agent.q_table_size:>7,} | "
                f"wr={wr:.1%} (best={best_wr:.1%}) | "
                f"ETA={eta/60:.1f}min"
            )

    return best_wr


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="", help="Checkpoint để fine-tune (vd: Q_v41). Bỏ trống = scratch")
    p.add_argument("--phase1-eps", type=int, default=300000, help="Episodes phase 1 (default: 300k)")
    p.add_argument("--phase2-eps", type=int, default=500000, help="Episodes phase 2 (default: 500k)")
    p.add_argument("--phase3-eps", type=int, default=200000, help="Episodes phase 3 (default: 200k)")
    p.add_argument("--skip-phase1", action="store_true", help="Bỏ qua phase 1")
    p.add_argument("--skip-phase2", action="store_true", help="Bỏ qua phase 2")
    p.add_argument("--skip-phase3", action="store_true", help="Bỏ qua phase 3")
    args = p.parse_args()

    from app.ml.q_agent import QAgent

    # ── Load base agent ───────────────────────────────────────────────────
    if args.base:
        db = _new_db()
        try:
            from app.services.checkpoint_service import load_agent
            agent = load_agent(db, args.base)
            logger.info(f"Loaded base: {args.base} ({agent.q_table_size:,} states)")
        finally:
            db.close()
    else:
        agent = QAgent(use_symmetry=True)
        logger.info("Starting from scratch")

    total_start = time.time()
    last_version = args.base or None

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Warm-up: khám phá rộng, epsilon cao
    # Mục tiêu: xây dựng Q-table lớn, học các pattern cơ bản
    # ════════════════════════════════════════════════════════════════════════
    if not args.skip_phase1:
        logger.info("=" * 60)
        logger.info(f"PHASE 1 — Warm-up ({args.phase1_eps:,} episodes)")
        logger.info("  Mục tiêu: khám phá rộng, xây Q-table lớn")
        logger.info("=" * 60)

        wr = _run_phase(
            agent,
            episodes      = args.phase1_eps,
            epsilon_start = 0.8,
            epsilon_min   = 0.1,
            epsilon_decay = 0.99995,
            alpha         = 0.3,
            opponent      = None,
            test_every    = 50000,
            test_games    = 200,
            phase_name    = "P1-warmup",
        )

        last_version = _save(agent, "phase1_warmup", args.phase1_eps, wr,
                             base_ver=last_version,
                             meta={"alpha": 0.3, "epsilon_min": 0.1})
        logger.info(f"Phase 1 done. Win rate: {wr:.1%}\n")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 2 — Focus: epsilon giảm dần, học khai thác
    # Mục tiêu: củng cố kiến thức, tăng win rate vs random
    # ════════════════════════════════════════════════════════════════════════
    if not args.skip_phase2:
        logger.info("=" * 60)
        logger.info(f"PHASE 2 — Focus ({args.phase2_eps:,} episodes)")
        logger.info("  Mục tiêu: giảm epsilon, khai thác kiến thức đã học")
        logger.info("=" * 60)

        wr = _run_phase(
            agent,
            episodes      = args.phase2_eps,
            epsilon_start = 0.4,
            epsilon_min   = 0.05,
            epsilon_decay = 0.99999,   # decay rất chậm
            alpha         = 0.15,      # learning rate thấp hơn để ổn định
            opponent      = None,
            test_every    = 50000,
            test_games    = 200,
            phase_name    = "P2-focus",
        )

        last_version = _save(agent, "phase2_focus", args.phase2_eps, wr,
                             base_ver=last_version,
                             meta={"alpha": 0.15, "epsilon_min": 0.05})
        logger.info(f"Phase 2 done. Win rate: {wr:.1%}\n")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 3 — Polish: đấu vs frozen best version
    # Mục tiêu: ổn định, không quên kiến thức cũ
    # ════════════════════════════════════════════════════════════════════════
    if not args.skip_phase3:
        logger.info("=" * 60)
        logger.info(f"PHASE 3 — Polish ({args.phase3_eps:,} episodes)")
        logger.info("  Mục tiêu: đấu vs bản thân tốt nhất, ổn định Q-table")
        logger.info("=" * 60)

        frozen = copy.deepcopy(agent)
        frozen.epsilon = 0.0  # frozen opponent không explore

        wr = _run_phase(
            agent,
            episodes      = args.phase3_eps,
            epsilon_start = 0.15,
            epsilon_min   = 0.02,
            epsilon_decay = 0.99999,
            alpha         = 0.08,      # learning rate rất thấp để fine-tune nhẹ
            opponent      = frozen,    # đấu vs bản thân đã freeze
            test_every    = 50000,
            test_games    = 200,
            phase_name    = "P3-polish",
        )

        last_version = _save(agent, "phase3_polish", args.phase3_eps, wr,
                             base_ver=last_version,
                             meta={"alpha": 0.08, "epsilon_min": 0.02})
        logger.info(f"Phase 3 done. Win rate: {wr:.1%}\n")

    # ── Final summary ─────────────────────────────────────────────────────
    from app.ml.evaluator import eval_vs_random
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION (500 games)")
    logger.info("=" * 60)
    final = eval_vs_random(agent, n_games=500)
    total_time = (time.time() - total_start) / 60

    logger.info(f"  Win rate:    {final['win_rate']:.1%}")
    logger.info(f"  Wins:        {final['wins']}")
    logger.info(f"  Losses:      {final['losses']}")
    logger.info(f"  Draws:       {final['draws']}")
    logger.info(f"  Q-table:     {agent.q_table_size:,} states")
    logger.info(f"  Total time:  {total_time:.1f} min")
    logger.info(f"  Last version: {last_version}")
    logger.info("=" * 60)
    logger.info(f"➡️  Vào Admin UI → Deploy '{last_version}'")


if __name__ == "__main__":
    main()
