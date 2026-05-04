"""
Test thực tế: "người" chơi có chiến lược thật (không random)
để đánh giá AI V7.
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from app.ml.environment import CaroEnv, X, O, BOARD_SIZE
from app.ml.q_agent import QAgent, _find_critical_moves

SYMBOLS = {0: ".", 1: "X", 2: "O"}

def print_board(state, moves_history=None):
    board = list(state)
    print("   " + "  ".join(str(i) for i in range(BOARD_SIZE)))
    for r in range(BOARD_SIZE):
        row = board[r*BOARD_SIZE:(r+1)*BOARD_SIZE]
        print(f"{r}  " + "  ".join(SYMBOLS[c] for c in row))
    print()

def smart_human_move(env, human_player):
    """
    Người chơi thông minh:
    1. Thắng ngay nếu có thể
    2. Tạo fork (2 hướng thắng cùng lúc)
    3. Tạo chuỗi dài nhất có thể
    4. Đánh gần quân mình nhất
    """
    import numpy as np
    state = env.get_state()
    valid = env.get_valid_actions()
    board = np.array(state).reshape(BOARD_SIZE, BOARD_SIZE)
    opponent = O if human_player == X else X
    directions = [(0,1),(1,0),(1,1),(1,-1)]

    def count_line(r, c, dr, dc, player):
        count = 0
        rr, cc = r, c
        while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[rr][cc] == player:
            count += 1; rr += dr; cc += dc
        rr, cc = r-dr, c-dc
        while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[rr][cc] == player:
            count += 1; rr -= dr; cc -= dc
        return count

    # 1. Thắng ngay
    for action in valid:
        r, c = divmod(action, BOARD_SIZE)
        board[r][c] = human_player
        if any(count_line(r,c,dr,dc,human_player) >= 4 for dr,dc in directions):
            board[r][c] = 0
            return action, "WIN NOW"
        board[r][c] = 0

    # 2. Chặn đối thủ thắng
    for action in valid:
        r, c = divmod(action, BOARD_SIZE)
        board[r][c] = opponent
        if any(count_line(r,c,dr,dc,opponent) >= 4 for dr,dc in directions):
            board[r][c] = 0
            return action, "BLOCK WIN"
        board[r][c] = 0

    # 3. Tạo fork: đánh vào ô tạo được 2+ hướng có 3 quân
    best_fork = None
    best_fork_score = 0
    for action in valid:
        r, c = divmod(action, BOARD_SIZE)
        board[r][c] = human_player
        threats = sum(1 for dr,dc in directions if count_line(r,c,dr,dc,human_player) >= 3)
        if threats >= 2 and threats > best_fork_score:
            best_fork_score = threats
            best_fork = action
        board[r][c] = 0
    if best_fork is not None:
        return best_fork, f"FORK ({best_fork_score} threats)"

    # 4. Tạo chuỗi dài nhất
    best_action = valid[0]
    best_score = -1
    for action in valid:
        r, c = divmod(action, BOARD_SIZE)
        board[r][c] = human_player
        score = max(count_line(r,c,dr,dc,human_player) for dr,dc in directions)
        if score > best_score:
            best_score = score
            best_action = action
        board[r][c] = 0
    return best_action, f"EXTEND (len={best_score})"


def play_full_game(version, human_is_x=True, label=""):
    agent = QAgent(epsilon=0.0)
    agent.load(f"data/checkpoints/{version}.pkl")

    env = CaroEnv()
    state = env.reset()
    human_player = X if human_is_x else O
    ai_player = O if human_is_x else X
    step = 0

    print(f"\n{'='*55}")
    print(f"{label}")
    print(f"Người={'X' if human_is_x else 'O'} | AI={'O' if human_is_x else 'X'}")
    print(f"{'='*55}")
    print_board(state)

    while not env.done and step < 49:
        valid = env.get_valid_actions()
        if not valid:
            break

        is_human = (env.current_player == human_player)

        if is_human:
            action, reason = smart_human_move(env, human_player)
            r, c = divmod(action, BOARD_SIZE)
            print(f"  Người [{'X' if env.current_player==X else 'O'}] → ({r},{c})  [{reason}]")
        else:
            # Check what AI sees
            critical = _find_critical_moves(state, valid, env.current_player)
            action = agent.choose_action(state, valid, greedy=True,
                                         current_player=env.current_player)
            r, c = divmod(action, BOARD_SIZE)
            ai_reason = ""
            if action in critical["win_moves"]:
                ai_reason = "WIN NOW ✅"
            elif action in critical["block_moves"]:
                ai_reason = "BLOCK ✅"
            else:
                ai_reason = "Q-table"
            print(f"  AI    [{'X' if env.current_player==X else 'O'}] → ({r},{c})  [{ai_reason}]")

        state, reward, done = env.step(action)
        step += 1
        print_board(state)

    if env.winner == human_player:
        print(f"  ❌ NGƯỜI THẮNG sau {step} nước")
        return "human"
    elif env.winner == ai_player:
        print(f"  ✅ AI THẮNG sau {step} nước")
        return "ai"
    else:
        print(f"  🤝 HÒA sau {step} nước")
        return "draw"


# ── Chạy 2 ván ───────────────────────────────────────────────────────────
VERSION = "Q_v7"

r1 = play_full_game(VERSION, human_is_x=True,
                    label="VÁN 1: Người (X) đi trước, AI (O) đi sau")

r2 = play_full_game(VERSION, human_is_x=False,
                    label="VÁN 2: AI (X) đi trước, Người (O) đi sau")

print(f"\n{'='*55}")
print(f"KẾT QUẢ: Ván 1={r1.upper()} | Ván 2={r2.upper()}")
print(f"{'='*55}")
