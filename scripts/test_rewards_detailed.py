"""
Test chi tiết reward system
Chạy: python scripts/test_rewards_detailed.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.environment import CaroEnv, X, O

def test_rewards_detailed():
    print("=== DETAILED REWARD TESTING ===")
    
    # Test 1: Center move
    print("\n1. Testing center move:")
    env = CaroEnv()
    center_action = 24  # Center của board 7x7 (3,3)
    next_state, reward, done = env.step(center_action)
    print(f"   Center move (3,3): reward = {reward}")
    
    # Test 2: Corner move (should get penalty)
    print("\n2. Testing corner move:")
    env2 = CaroEnv()
    corner_action = 0  # (0,0)
    next_state, reward, done = env2.step(corner_action)
    print(f"   Corner move (0,0): reward = {reward}")
    
    # Test 3: Create 2-in-a-row
    print("\n3. Testing 2-in-a-row creation:")
    env3 = CaroEnv()
    # Đánh 2 nước liên tiếp
    env3.step(24)  # Center
    env3.step(0)   # Opponent move
    action = 25    # Next to center (3,4)
    next_state, reward, done = env3.step(action)
    print(f"   2-in-a-row move: reward = {reward}")
    
    # Test 4: Create 3-in-a-row
    print("\n4. Testing 3-in-a-row creation:")
    env4 = CaroEnv()
    # Tạo tình huống có 2 quân X liên tiếp
    env4.board[3][1] = X  # X
    env4.board[3][2] = X  # X
    env4.current_player = X
    
    print(f"   Board before move:")
    print(f"   Row 3: {env4.board[3]}")
    
    # Đánh tiếp để tạo 3 quân liên tiếp
    action = 3 * 7 + 3  # Row 3, Col 3
    next_state, reward, done = env4.step(action)
    print(f"   3-in-a-row move (3,3): reward = {reward}")
    print(f"   Board after move:")
    print(f"   Row 3: {env4.board[3]}")
    
    # Test 5: Block opponent win
    print("\n5. Testing block opponent win:")
    env5 = CaroEnv()
    # Tạo tình huống đối thủ sắp thắng
    env5.board[2][1] = O  # O
    env5.board[2][2] = O  # O  
    env5.board[2][3] = O  # O (3 quân liên tiếp)
    env5.current_player = X
    
    print(f"   Board before block:")
    print(f"   Row 2: {env5.board[2]}")
    
    # X chặn bằng cách đánh vào (2,4)
    action = 2 * 7 + 4  # Row 2, Col 4
    next_state, reward, done = env5.step(action)
    print(f"   Block win move (2,4): reward = {reward}")
    
    # Test 6: Test _has_open_ends function
    print("\n6. Testing open ends detection:")
    env6 = CaroEnv()
    env6.board[3][1] = X
    env6.board[3][2] = X
    env6.board[3][3] = X  # 3 quân liên tiếp với 2 đầu mở
    
    has_open = env6._has_open_ends(3, 2, X, 3)
    print(f"   3-in-a-row with open ends: {has_open}")
    
    # Chặn 1 đầu
    env6.board[3][0] = O  # Chặn đầu trái
    has_open_blocked = env6._has_open_ends(3, 2, X, 3)
    print(f"   3-in-a-row with 1 end blocked: {has_open_blocked}")
    
    print(f"\n✅ Detailed reward testing complete!")

if __name__ == "__main__":
    test_rewards_detailed()