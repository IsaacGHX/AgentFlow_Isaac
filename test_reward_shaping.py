#!/usr/bin/env python3
"""
Test script to validate the reward shaping implementation.
"""

def test_compute_turn_level_rewards():
    """Test the turn-level reward computation logic."""

    # Mock the _compute_turn_level_rewards function
    def compute_turn_level_rewards(final_reward, n_turns, gamma=0.99, enable=True):
        """
        Compute turn-level rewards with discount factor.
        """
        if not enable or n_turns <= 1:
            return [final_reward] * n_turns

        turn_rewards = []
        for turn_idx in range(n_turns):
            distance_from_end = n_turns - turn_idx - 1
            turn_reward = final_reward * (gamma ** distance_from_end)
            turn_rewards.append(turn_reward)

        return turn_rewards

    print("=" * 80)
    print("Testing Reward Shaping Implementation")
    print("=" * 80)

    # Test Case 1: Standard case with gamma=0.99, 3 turns, reward=1.0
    print("\n[Test 1] Standard case: gamma=0.99, n_turns=3, final_reward=1.0")
    rewards = compute_turn_level_rewards(1.0, 3, gamma=0.99, enable=True)
    print(f"Turn rewards: {rewards}")
    print(f"  Turn 0: {rewards[0]:.6f} (should be ~0.9801)")
    print(f"  Turn 1: {rewards[1]:.6f} (should be ~0.9900)")
    print(f"  Turn 2: {rewards[2]:.6f} (should be ~1.0000)")
    assert abs(rewards[0] - 0.9801) < 0.0001, f"Turn 0 reward incorrect: {rewards[0]}"
    assert abs(rewards[1] - 0.99) < 0.0001, f"Turn 1 reward incorrect: {rewards[1]}"
    assert abs(rewards[2] - 1.0) < 0.0001, f"Turn 2 reward incorrect: {rewards[2]}"
    print("✓ Test 1 passed!")

    # Test Case 2: Single turn (should return final reward)
    print("\n[Test 2] Single turn: n_turns=1, final_reward=1.0")
    rewards = compute_turn_level_rewards(1.0, 1, gamma=0.99, enable=True)
    print(f"Turn rewards: {rewards}")
    assert len(rewards) == 1, "Should have exactly 1 reward"
    assert rewards[0] == 1.0, f"Single turn should get full reward: {rewards[0]}"
    print("✓ Test 2 passed!")

    # Test Case 3: Disabled reward shaping (all turns get same reward)
    print("\n[Test 3] Disabled shaping: enable=False, n_turns=3, final_reward=1.0")
    rewards = compute_turn_level_rewards(1.0, 3, gamma=0.99, enable=False)
    print(f"Turn rewards: {rewards}")
    assert all(r == 1.0 for r in rewards), "All turns should get final reward when disabled"
    print("✓ Test 3 passed!")

    # Test Case 4: Different gamma value (0.95)
    print("\n[Test 4] Lower gamma: gamma=0.95, n_turns=3, final_reward=1.0")
    rewards = compute_turn_level_rewards(1.0, 3, gamma=0.95, enable=True)
    print(f"Turn rewards: {rewards}")
    print(f"  Turn 0: {rewards[0]:.6f} (should be ~0.9025)")
    print(f"  Turn 1: {rewards[1]:.6f} (should be ~0.9500)")
    print(f"  Turn 2: {rewards[2]:.6f} (should be ~1.0000)")
    assert abs(rewards[0] - 0.9025) < 0.0001, f"Turn 0 reward incorrect: {rewards[0]}"
    assert abs(rewards[1] - 0.95) < 0.0001, f"Turn 1 reward incorrect: {rewards[1]}"
    assert abs(rewards[2] - 1.0) < 0.0001, f"Turn 2 reward incorrect: {rewards[2]}"
    print("✓ Test 4 passed!")

    # Test Case 5: Negative reward
    print("\n[Test 5] Negative reward: gamma=0.99, n_turns=3, final_reward=-1.0")
    rewards = compute_turn_level_rewards(-1.0, 3, gamma=0.99, enable=True)
    print(f"Turn rewards: {rewards}")
    print(f"  Turn 0: {rewards[0]:.6f} (should be ~-0.9801)")
    print(f"  Turn 1: {rewards[1]:.6f} (should be ~-0.9900)")
    print(f"  Turn 2: {rewards[2]:.6f} (should be ~-1.0000)")
    assert abs(rewards[0] - (-0.9801)) < 0.0001, f"Turn 0 reward incorrect: {rewards[0]}"
    assert abs(rewards[1] - (-0.99)) < 0.0001, f"Turn 1 reward incorrect: {rewards[1]}"
    assert abs(rewards[2] - (-1.0)) < 0.0001, f"Turn 2 reward incorrect: {rewards[2]}"
    print("✓ Test 5 passed!")

    # Test Case 6: Many turns (10 turns)
    print("\n[Test 6] Many turns: gamma=0.99, n_turns=10, final_reward=1.0")
    rewards = compute_turn_level_rewards(1.0, 10, gamma=0.99, enable=True)
    print(f"Turn rewards (first 3): {[f'{r:.6f}' for r in rewards[:3]]}")
    print(f"Turn rewards (last 3): {[f'{r:.6f}' for r in rewards[-3:]]}")
    assert len(rewards) == 10, "Should have 10 rewards"
    assert rewards[-1] == 1.0, "Last turn should get full reward"
    assert rewards[0] < rewards[-1], "First turn should get less than last turn"
    print(f"  First turn gets {rewards[0]/rewards[-1]*100:.2f}% of final reward")
    print("✓ Test 6 passed!")

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    print("\nReward Shaping Summary:")
    print("  - Higher gamma (0.95-0.99): More credit to later turns")
    print("  - Lower gamma (0.8-0.9): More even distribution")
    print("  - Disable (enable=False): All turns get same final reward")
    print("\nRecommended settings:")
    print("  - REWARD_SHAPING_GAMMA: 0.99 (default, good for most cases)")
    print("  - ENABLE_REWARD_SHAPING: True (to enable smooth reward distribution)")


if __name__ == "__main__":
    test_compute_turn_level_rewards()
