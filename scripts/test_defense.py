"""
Test khả năng phòng thủ của AI: người chơi cố tình tạo chuỗi 3 quân
để xem AI có chặn không.
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from app.ml.environment import CaroEnv, X, O, BOARD_SIZE
from app.ml.q_agent import QAgent

SYMBOLS = {0: ".", 1: "X", 2: "O"}

def print_board(state, last_action=None):
    board = list(state)
    print("   " + " ".join(str(i) for i in range(BOARD_SIZE)))
    for r in range(BOARD_SIZE):
        row = board[r*BOARD_SIZE:(r+1)*BOARD_SIZE]
        cells = []
        for c, v in enumerate(row):
            sym = SYMBOLS[v]
            if last_action == r*BOARD_SIZE+c:
                sym = f"[{sym}]"
            else:
                sym = f" {sym} "
            cells.append(sym)
        print(f"{r} " + "".join(cells))
    print()


def play_scripted(agent_version, human_moves, human_is_x=False):
    """
    Chạy 1 ván với nước đi của 'người' được script sẵn.
    human_is_x=False → người chơi O, AI chơi X (đi trước)
    """
    agent = QAgent(epsilon=0.0)
    agent.load(f"data/checkpoints/{agent_version}.pkl")

    env = CaroEnv()
    state = env.reset()
    move_iter = iter(human_moves)

    print(f"\n{'='*50}")
    print(f"AI={agent_version} vs Người | AI={'X' if not human_is_x else 'O'}")
    print(f"{'='*50}")
    print_board(state)

    step = 0
    while not env.done:
        valid = env.get_valid_actions()
        if not valid:
            break

        is_human = (env.current_player == O) if not human_is_x else (env.current_player == X)

        if is_human:
            try:
                action = next(move_iter)
            except StopIteration:
                print("  [Người hết nước script, dừng ván]")
                break
            r, c = divmod(action, BOARD_SIZE)
            print(f"  Người [{'X' if env.current_player==X else 'O'}] → ({r},{c})")
        else:
            action = agent.choose_action(state, valid, greedy=True,
                                         current_player=env.current_player)
            r, c = divmod(action, BOARD_SIZE)
            print(f"  AI    [{'X' if env.current_player==X else 'O'}] → ({r},{c})")

        state, reward, done = env.step(action)
        print_board(state, last_action=action)
        step += 1

    if env.winner == X:
        print("  ✅ X thắng")
    elif env.winner == O:
        print("  ✅ O thắng")
    else:
        print("  🤝 Hòa / Dừng giữa chừng")


# ── Test 1: Người (O) đi trước, tạo chuỗi ngang ──────────────────────────
# Người đánh (3,1) → (3,2) → (3,3) → cố tạo 4 quân hàng ngang
# AI (X) đi trước, xem có chặn hàng ngang của O không
print("\n" + "="*50)
print("TEST 1: Người (O) tạo chuỗi NGANG — AI có chặn không?")
print("Người đánh: (3,1)→(3,2)→(3,3)→(3,4) cố thắng")
play_scripted(
    "Q_v7",
    human_moves=[
        3*BOARD_SIZE+1,  # (3,1)
        3*BOARD_SIZE+2,  # (3,2)
        3*BOARD_SIZE+3,  # (3,3)
        3*BOARD_SIZE+4,  # (3,4) ← nước thắng nếu AI không chặn
    ],
    human_is_x=False  # người=O, AI=X đi trước
)

# ── Test 2: Người (O) tạo chuỗi chéo ─────────────────────────────────────
print("\n" + "="*50)
print("TEST 2: Người (O) tạo chuỗi CHÉO — AI có chặn không?")
print("Người đánh: (1,1)→(2,2)→(3,3)→(4,4) cố thắng")
play_scripted(
    "Q_v7",
    human_moves=[
        1*BOARD_SIZE+1,  # (1,1)
        2*BOARD_SIZE+2,  # (2,2)
        3*BOARD_SIZE+3,  # (3,3)
        4*BOARD_SIZE+4,  # (4,4) ← nước thắng nếu AI không chặn
    ],
    human_is_x=False
)

# ── Test 3: Người (O) tạo chuỗi dọc ──────────────────────────────────────
print("\n" + "="*50)
print("TEST 3: Người (O) tạo chuỗi DỌC — AI có chặn không?")
print("Người đánh: (1,3)→(2,3)→(3,3)→(4,3) cố thắng")
play_scripted(
    "Q_v7",
    human_moves=[
        1*BOARD_SIZE+3,  # (1,3)
        2*BOARD_SIZE+3,  # (2,3)
        3*BOARD_SIZE+3,  # (3,3)
        4*BOARD_SIZE+3,  # (4,3) ← nước thắng nếu AI không chặn
    ],
    human_is_x=False
)
