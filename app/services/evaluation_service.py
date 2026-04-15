"""
Chạy test thủ công từ Admin UI.
"""
from typing import Optional
from sqlalchemy.orm import Session
from app.ml.evaluator import eval_vs_random, eval_vs_checkpoint
from app.services.checkpoint_service import load_agent
from app.repositories import checkpoint_repo


def run_manual_eval(
    db: Session,
    version: str,
    compare_version: Optional[str] = None,
    n_games: int = 100,
) -> dict:
    agent = load_agent(db, version)
    agent.epsilon = 0.0

    result = {"version": version, "n_games": n_games}

    r = eval_vs_random(agent, n_games)
    result["vs_random"] = {
        "win_rate": r["win_rate"],
        "wins": r["wins"],
        "losses": r["losses"],
        "draws": r["draws"],
    }

    if compare_version:
        opponent = load_agent(db, compare_version)
        opponent.epsilon = 0.0
        r2 = eval_vs_checkpoint(agent, opponent, n_games)
        result["vs_version"] = {
            "compared_version": compare_version,
            "win_rate": r2["win_rate"],
            "wins": r2["wins"],
            "losses": r2["losses"],
            "draws": r2["draws"],
        }

    return result
