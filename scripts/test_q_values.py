"""
Test Q-values để hiểu tại sao AI không học được
Chạy: python scripts/test_q_values.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.q_agent import QAgent
from app.ml.environment import CaroEnv
import numpy as np

def test_q_values():
    print("=== TESTING Q-VALUES ===")
    
    # Tạo agent mới với reward system cải tiến
    agent = QAgent(
        alpha=0.2,
        gamma=0.9,
        epsilon=0.0,  # Không random để test
        use_symmetry=True
    )
    
    # Tạo environment
    env = CaroEnv()
    
    # Test 1: Kiểm tra Q-table initialization
    print(f"1. Q-table initialization:")
    print(f"   Initial Q-table size: {agent.q_table_size}")
    
    # Tạo một state và xem Q-values ban đầu
    state = env.reset()
    valid = env.get_valid_actions()
    
    # Force tạo Q-values cho state này
    canonical_state, t_idx = agent._get_canonical(state)
    q_values = agent._get_q(canonical_state)
    print(f"   Initial Q-values for first state: {q_values[:5]}...")  # Show first 5
    print(f"   Min: {q_values.min():.4f}, Max: {q_values.max():.4f}, Mean: {q_values.mean():.4f}")
    
    # Test 2: Simulate một vài moves và xem reward
    print(f"\n2. Testing reward system:")
    
    # Đánh vào center (thường là move tốt)
    center_action = 12  # Center của board 5x5
    next_state, reward, done = env.step(center_action)
    print(f"   Move to center (action {center_action}): reward = {reward}")
    
    # Đánh move tiếp theo
    if not done:
        valid = env.get_valid_actions()
        action = valid[0]  # Đánh move đầu tiên available
        next_state, reward, done = env.step(action)
        print(f"   Next move (action {action}): reward = {reward}")
    
    # Test 3: Kiểm tra reward function trực tiếp
    print(f"\n3. Testing reward function directly:")
    env2 = CaroEnv()
    
    # Tạo tình huống có 2 quân liên tiếp
    env2.board[2][1] = 1  # X
    env2.board[2][2] = 1  # X
    env2.current_player = 1
    
    # Đánh tiếp để tạo 3 quân liên tiếp
    action = 2 * 5 + 3  # Row 2, Col 3
    next_state, reward, done = env2.step(action)
    print(f"   Creating 3-in-a-row: reward = {reward}")
    
    # Test 4: Load checkpoint cũ và xem Q-values
    print(f"\n4. Testing existing checkpoint:")
    try:
        old_agent = QAgent()
        old_agent.load("data/checkpoints/trained_from_scratch.pkl")
        print(f"   Loaded checkpoint Q-table size: {old_agent.q_table_size}")
        
        # Sample một vài Q-values
        sample_states = list(old_agent.q_table.keys())[:3]
        for i, state in enumerate(sample_states):
            q_vals = old_agent.q_table[state]
            print(f"   State {i+1}: Min={q_vals.min():.4f}, Max={q_vals.max():.4f}, Mean={q_vals.mean():.4f}")
            
        # Tổng thống kê
        all_q_values = []
        for state_q in old_agent.q_table.values():
            all_q_values.extend(state_q.tolist())
        
        all_q_values = np.array(all_q_values)
        positive = np.sum(all_q_values > 0.1)
        negative = np.sum(all_q_values < -0.1)
        neutral = len(all_q_values) - positive - negative
        
        print(f"   Total Q-values: {len(all_q_values):,}")
        print(f"   Positive (>0.1): {positive:,} ({positive/len(all_q_values)*100:.1f}%)")
        print(f"   Negative (<-0.1): {negative:,} ({negative/len(all_q_values)*100:.1f}%)")
        print(f"   Neutral: {neutral:,} ({neutral/len(all_q_values)*100:.1f}%)")
        
    except Exception as e:
        print(f"   Could not load checkpoint: {e}")
    
    print(f"\n✅ Q-value analysis complete!")

if __name__ == "__main__":
    test_q_values()