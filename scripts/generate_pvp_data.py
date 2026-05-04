"""
Giả lập dữ liệu PvP chất lượng cao.
Tạo các ván đấu giữa 2 "người chơi" thông minh với chiến thuật khác nhau.
Chạy: python -m scripts.generate_pvp_data
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import random
import numpy as np
from sqlalchemy.orm import Session
from app.database import SessionLocal

# Import tất cả models để SQLAlchemy nhận diện foreign keys
from app.models.user import User
from app.models.game import Game
from app.models.game_step import GameStep
from app.models.checkpoint import Checkpoint
from app.models.training_log import TrainingLog

from app.ml.environment import CaroEnv, X, O, EMPTY, BOARD_SIZE, WIN_LENGTH
from app.services.game_service import create_game, apply_move
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class SmartPlayer:
    """Base class cho người chơi thông minh"""
    
    def __init__(self, name: str, style: str):
        self.name = name
        self.style = style
    
    def choose_action(self, env: CaroEnv, player: int) -> int:
        """Chọn nước đi thông minh"""
        valid = env.get_valid_actions()
        if not valid:
            return None
            
        # 1. Kiểm tra thắng ngay
        win_move = self._find_winning_move(env, player, valid)
        if win_move is not None:
            return win_move
            
        # 2. Kiểm tra chặn đối thủ thắng
        opponent = O if player == X else X
        block_move = self._find_winning_move(env, opponent, valid)
        if block_move is not None:
            return block_move
            
        # 3. Chiến thuật theo style
        if self.style == "aggressive":
            return self._aggressive_move(env, player, valid)
        elif self.style == "defensive":
            return self._defensive_move(env, player, valid)
        else:
            return self._balanced_move(env, player, valid)
    
    def _find_winning_move(self, env: CaroEnv, player: int, valid: list) -> int:
        """Tìm nước thắng ngay"""
        for action in valid:
            # Thử đánh
            row, col = divmod(action, BOARD_SIZE)
            env.board[row][col] = player
            
            # Kiểm tra thắng
            is_win = self._check_win(env, row, col, player)
            
            # Hoàn tác
            env.board[row][col] = EMPTY
            
            if is_win:
                return action
        return None
    
    def _check_win(self, env: CaroEnv, row: int, col: int, player: int) -> bool:
        """Kiểm tra nước đi có thắng không"""
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        for dr, dc in directions:
            count = 1
            # Đếm về 2 phía
            for sign in [1, -1]:
                r, c = row + sign*dr, col + sign*dc
                while (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and 
                       env.board[r][c] == player):
                    count += 1
                    r += sign*dr
                    c += sign*dc
            if count >= WIN_LENGTH:
                return True
        return False
    
    def _count_consecutive(self, env: CaroEnv, row: int, col: int, player: int) -> int:
        """Đếm chuỗi dài nhất qua ô (row,col)"""
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        max_count = 1
        for dr, dc in directions:
            count = 1
            for sign in [1, -1]:
                r, c = row + sign*dr, col + sign*dc
                while (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and 
                       env.board[r][c] == player):
                    count += 1
                    r += sign*dr
                    c += sign*dc
            max_count = max(max_count, count)
        return max_count
    
    def _aggressive_move(self, env: CaroEnv, player: int, valid: list) -> int:
        """Chiến thuật tấn công: tạo chuỗi dài"""
        best_moves = []
        best_score = -1
        
        for action in valid:
            row, col = divmod(action, BOARD_SIZE)
            env.board[row][col] = player
            
            # Điểm cho việc tạo chuỗi
            my_len = self._count_consecutive(env, row, col, player)
            score = my_len * 10
            
            # Bonus cho vị trí trung tâm
            center = BOARD_SIZE // 2
            dist = abs(row - center) + abs(col - center)
            score += (6 - dist)
            
            # Bonus cho việc tạo nhiều hướng
            score += self._count_directions(env, row, col, player) * 2
            
            env.board[row][col] = EMPTY
            
            if score > best_score:
                best_score = score
                best_moves = [action]
            elif score == best_score:
                best_moves.append(action)
        
        return random.choice(best_moves) if best_moves else random.choice(valid)
    
    def _defensive_move(self, env: CaroEnv, player: int, valid: list) -> int:
        """Chiến thuật phòng thủ: chặn đối thủ"""
        opponent = O if player == X else X
        best_moves = []
        best_score = -1
        
        for action in valid:
            row, col = divmod(action, BOARD_SIZE)
            
            # Điểm cho việc chặn đối thủ
            env.board[row][col] = opponent
            opp_len = self._count_consecutive(env, row, col, opponent)
            block_score = opp_len * 15  # Ưu tiên chặn cao hơn
            
            # Điểm cho việc tạo chuỗi của mình
            env.board[row][col] = player
            my_len = self._count_consecutive(env, row, col, player)
            attack_score = my_len * 5
            
            env.board[row][col] = EMPTY
            
            score = block_score + attack_score
            
            # Bonus vị trí trung tâm
            center = BOARD_SIZE // 2
            dist = abs(row - center) + abs(col - center)
            score += (6 - dist)
            
            if score > best_score:
                best_score = score
                best_moves = [action]
            elif score == best_score:
                best_moves.append(action)
        
        return random.choice(best_moves) if best_moves else random.choice(valid)
    
    def _balanced_move(self, env: CaroEnv, player: int, valid: list) -> int:
        """Chiến thuật cân bằng"""
        # 70% aggressive, 30% defensive
        if random.random() < 0.7:
            return self._aggressive_move(env, player, valid)
        else:
            return self._defensive_move(env, player, valid)
    
    def _count_directions(self, env: CaroEnv, row: int, col: int, player: int) -> int:
        """Đếm số hướng có thể mở rộng"""
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        count = 0
        for dr, dc in directions:
            # Kiểm tra có thể mở rộng theo hướng này không
            can_expand = False
            for sign in [1, -1]:
                r, c = row + sign*dr, col + sign*dc
                if (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and 
                    env.board[r][c] == EMPTY):
                    can_expand = True
                    break
            if can_expand:
                count += 1
        return count


def simulate_pvp_game(db: Session, player1: SmartPlayer, player2: SmartPlayer) -> dict:
    """Giả lập 1 ván PvP giữa 2 người chơi thông minh"""
    
    # Sử dụng dummy user IDs đã tạo
    user1_id = 4  # pvp_player1
    user2_id = 5  # pvp_player2
    
    game = create_game(db, player_x_id=user1_id, player_o_id=user2_id, mode="pvp")
    env = CaroEnv()
    
    moves = []
    current_player_obj = player1  # X đi trước
    
    while not env.done:
        valid = env.get_valid_actions()
        if not valid:
            break
            
        # Người chơi hiện tại chọn nước
        action = current_player_obj.choose_action(env, env.current_player)
        if action is None:
            break
            
        # Áp dụng nước đi
        result = apply_move(db, game.id, action, env)
        moves.append({
            "player": current_player_obj.name,
            "action": action,
            "player_symbol": "X" if env.current_player == X else "O"
        })
        
        if result["done"]:
            break
            
        # Chuyển lượt
        current_player_obj = player2 if current_player_obj == player1 else player1
    
    return {
        "game_id": game.id,
        "winner": result.get("winner"),
        "moves": len(moves),
        "player1": player1.name,
        "player2": player2.name
    }


def generate_pvp_dataset(num_games: int = 100):
    """Tạo dataset PvP giả lập"""
    logger.info(f"=== GENERATING {num_games} PvP GAMES ===")
    
    # Tạo các loại người chơi
    players = [
        SmartPlayer("Aggressive_A", "aggressive"),
        SmartPlayer("Aggressive_B", "aggressive"), 
        SmartPlayer("Defensive_A", "defensive"),
        SmartPlayer("Defensive_B", "defensive"),
        SmartPlayer("Balanced_A", "balanced"),
        SmartPlayer("Balanced_B", "balanced"),
    ]
    
    db = SessionLocal()
    results = []
    
    try:
        for i in range(num_games):
            # Random chọn 2 người chơi khác nhau
            player1, player2 = random.sample(players, 2)
            
            logger.info(f"Game {i+1}/{num_games}: {player1.name} vs {player2.name}")
            
            result = simulate_pvp_game(db, player1, player2)
            results.append(result)
            
            if (i + 1) % 20 == 0:
                logger.info(f"Completed {i+1}/{num_games} games")
    
    finally:
        db.close()
    
    # Thống kê
    logger.info("\n=== DATASET STATISTICS ===")
    total_moves = sum(r["moves"] for r in results)
    winners = {}
    for r in results:
        winner = r["winner"] or "draw"
        winners[winner] = winners.get(winner, 0) + 1
    
    logger.info(f"Total games: {len(results)}")
    logger.info(f"Total moves: {total_moves}")
    logger.info(f"Avg moves per game: {total_moves/len(results):.1f}")
    logger.info(f"Winners: {winners}")
    
    # Phân tích theo style
    style_stats = {}
    for r in results:
        matchup = f"{r['player1'].split('_')[0]} vs {r['player2'].split('_')[0]}"
        if matchup not in style_stats:
            style_stats[matchup] = {"games": 0, "X_wins": 0, "O_wins": 0, "draws": 0}
        
        style_stats[matchup]["games"] += 1
        if r["winner"] == "X":
            style_stats[matchup]["X_wins"] += 1
        elif r["winner"] == "O":
            style_stats[matchup]["O_wins"] += 1
        else:
            style_stats[matchup]["draws"] += 1
    
    logger.info("\n=== STYLE MATCHUPS ===")
    for matchup, stats in style_stats.items():
        logger.info(f"{matchup}: {stats['games']} games, "
                   f"X:{stats['X_wins']} O:{stats['O_wins']} Draw:{stats['draws']}")
    
    logger.info(f"\n✅ Generated {num_games} PvP games successfully!")
    logger.info("Data saved to database. Ready for offline training.")


if __name__ == "__main__":
    # Tạo 20 ván PvP chất lượng cao để test nhanh
    generate_pvp_dataset(20)