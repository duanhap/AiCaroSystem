"""
Test Phase 2: self_play, evaluator, offline_train
Chạy: python scripts/test_phase2.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.q_agent import QAgent
from app.ml.evaluator import eval_vs_random, eval_vs_checkpoint
from app.ml.self_play import run_self_play
from app.ml.offline_train import run_offline_train
from app.ml.environment import CaroEnv

def test_evaluator():
    print("── Test Evaluator ──")
    agent = QAgent(epsilon=0.0)
    result = eval_vs_random(agent, n_games=50)
    print(f"Agent rỗng vs Random: win={result['wins']}, loss={result['losses']}, draw={result['draws']}")
    print(f"Win rate: {result['win_rate']:.2%}")
    print("✅ Evaluator OK\n")

def test_self_play():
    print("── Test Self-play (500 episodes) ──")
    agent = QAgent(alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995)
    logs = []

    def on_progress(entry):
        logs.append(entry)
        print(f"  ep={entry['episode']:4d} | ε={entry['epsilon']:.3f} | "
              f"Q-size={entry['q_table_size']:5d} | "
              f"vs_random={entry['win_rate_vs_random']:.2%}")

    result = run_self_play(
        agent=agent,
        episodes=500,
        test_interval=100,
        test_games=50,
        win_rate_target=0.95,
        on_progress=on_progress,
    )

    print(f"\nKết quả: {result['episodes_trained']} episodes")
    print(f"Win rate vs random: {result['win_rate_vs_random']:.2%}")
    print(f"Early stop: {result.get('stopped_early', False)}")
    print("✅ Self-play OK\n")
    return agent

def test_self_play_vs_old(trained_agent):
    print("── Test Self-play vs Old Version ──")
    import copy
    old_agent = copy.deepcopy(trained_agent)
    new_agent = QAgent(alpha=0.1, gamma=0.9, epsilon=0.5, epsilon_decay=0.99)

    def on_progress(entry):
        vs_v = entry['win_rate_vs_version']
        print(f"  ep={entry['episode']:4d} | vs_random={entry['win_rate_vs_random']:.2%} "
              f"| vs_old={vs_v:.2%}" if vs_v else f"  ep={entry['episode']:4d}")

    result = run_self_play(
        agent=new_agent,
        episodes=200,
        test_interval=100,
        test_games=50,
        compare_agent=old_agent,
        on_progress=on_progress,
    )
    print(f"Win rate vs old version: {result['win_rate_vs_version']}")
    print("✅ Self-play vs old OK\n")

def test_offline_train(trained_agent):
    print("── Test Offline Train ──")
    # Tạo dữ liệu giả lập (1 ván 5 bước)
    env = CaroEnv()
    fake_game = []
    state = env.reset()
    for i, action in enumerate([0, 7, 1, 8, 2, 9, 3]):
        try:
            next_state, reward, done = env.step(action)
            fake_game.append({
                "state": state,
                "action": action,
                "next_state": next_state,
                "next_valid_actions": env.get_valid_actions() if not done else [],
                "reward": reward,
                "done": done,
            })
            state = next_state
            if done:
                break
        except:
            break

    import copy
    agent = copy.deepcopy(trained_agent)
    old_agent = copy.deepcopy(trained_agent)

    result = run_offline_train(
        agent=agent,
        game_data=[fake_game],
        compare_agent=old_agent,
        test_games=50,
        on_progress=lambda p: print(f"  Ván {p['game_idx']}/{p['total']}"),
    )
    print(f"Win rate vs random: {result['win_rate_vs_random']:.2%}")
    print(f"Win rate vs old: {result['win_rate_vs_version']:.2%}" if result['win_rate_vs_version'] else "")
    print("✅ Offline train OK\n")

if __name__ == "__main__":
    test_evaluator()
    trained = test_self_play()
    test_self_play_vs_old(trained)
    test_offline_train(trained)
    print("🎉 Phase 2 ML Engine hoàn tất!")
