"""
Test agent bằng cách cho đấu vs random và in chi tiết từng ván.
Chạy: python -m scripts.test_agent --version Q_v7 --games 5
"""
import sys, os, argparse
sys.path.insert(0, os.path.abspath('.'))

BOARD_SIZE = 7
SYMBOLS = {0: ".", 1: "X", 2: "O"}

def print_board(state):
    board = list(state)
    print("   " + " ".join(str(i) for i in range(BOARD_SIZE)))
    for r in range(BOARD_SIZE):
        row = board[r*BOARD_SIZE:(r+1)*BOARD_SIZE]
        print(f"{r}  " + " ".join(SYMBOLS[c] for c in row))
    print()

def play_game(agent, agent_plays_x=True, verbose=True):
    from app.ml.environment import CaroEnv, X, O
    from app.ml.q_agent import QAgent

    random_agent = QAgent(epsilon=1.0)
    env = CaroEnv()
    state = env.reset()
    moves = []

    while not env.done:
        valid = env.get_valid_actions()
        if not valid:
            break

        is_agent = (env.current_player == X) == agent_plays_x
        current_player = env.current_player

        if is_agent:
            action = agent.choose_action(state, valid, greedy=True, current_player=current_player)
            who = "AI "
        else:
            action = random_agent.choose_action(state, valid)
            who = "RND"

        r, c = divmod(action, BOARD_SIZE)
        moves.append(f"{who}[{'X' if env.current_player==X else 'O'}] → ({r},{c})")
        state, reward, done = env.step(action)

    if verbose:
        print_board(env.get_state())
        for m in moves[-10:]:
            print(" ", m)

    winner = env.winner
    if winner is None:
        result = "draw"
    elif (winner == 1) == agent_plays_x:
        result = "AI_WIN"
    else:
        result = "AI_LOSE"
    return result, winner

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="Q_v7")
    p.add_argument("--games",   type=int, default=5)
    p.add_argument("--quiet",   action="store_true", help="Không in bàn cờ")
    args = p.parse_args()

    from app.ml.q_agent import QAgent
    agent = QAgent(epsilon=0.0)
    pkl = f"data/checkpoints/{args.version}.pkl"
    if not os.path.exists(pkl):
        print(f"❌ Không tìm thấy {pkl}")
        return
    agent.load(pkl)
    print(f"✅ Loaded {args.version} — Q-table: {agent.q_table_size:,} states\n")

    wins = losses = draws = 0
    for i in range(args.games):
        agent_x = (i % 2 == 0)  # xen kẽ vai
        role = "X (đi trước)" if agent_x else "O (đi sau)"
        print(f"{'='*40}")
        print(f"Ván {i+1}/{args.games} — AI đóng vai {role}")
        print(f"{'='*40}")
        result, winner = play_game(agent, agent_plays_x=agent_x, verbose=not args.quiet)
        print(f"  Kết quả: {result} (winner={winner})\n")
        if result == "AI_WIN":   wins += 1
        elif result == "AI_LOSE": losses += 1
        else:                     draws += 1

    print(f"{'='*40}")
    print(f"TỔNG KẾT {args.games} ván:")
    print(f"  AI thắng:  {wins}")
    print(f"  AI thua:   {losses}")
    print(f"  Hòa:       {draws}")
    print(f"  Win rate:  {wins/args.games:.0%}")

if __name__ == "__main__":
    main()
