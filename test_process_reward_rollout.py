"""
Test script for process reward mode with a deployed model on port 8000.
This script will test the rollout with process reward evaluation.
"""
import asyncio
import os
import sys
from datetime import datetime

# Add the train directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'train'))

from agentflow import NamedResources, LLM
from train.rollout import AgentFlowRollout, get_agent


async def test_single_rollout():
    """Test a single rollout with the deployed model on port 8000."""

    print("=" * 80)
    print("PROCESS REWARD MODE ROLLOUT TEST")
    print("=" * 80)

    # Model configuration
    model_name = "vllm-checkpoints/AgentFlow_general_qwen2.5-3B/rollout_all_qwen2.5-3B_discount_Reward/global_step_18/actor/huggingface"
    base_url = "http://localhost:8000/v1"
    temperature = 0.7

    print(f"\n[CONFIG]")
    print(f"  Model: {model_name}")
    print(f"  Base URL: {base_url}")
    print(f"  Temperature: {temperature}")

    # Tool configuration
    enabled_tools = [
        "Base_Generator_Tool",
        "Python_Coder_Tool",
        "Google_Search_Tool",
        "Wikipedia_Search_Tool"
    ]
    tool_engine = ["dashscope-qwen2.5-3b-instruct", "dashscope-qwen2.5-3b-instruct", "Default", "Default"]

    print(f"  Enabled Tools: {enabled_tools}")
    print(f"  Tool Engines: {tool_engine}")

    # Create resources
    llm = LLM(model=model_name, endpoint=base_url)
    resources: NamedResources = {
        "main_llm": llm
    }

    print(f"\n[INFO] Initializing rollout agent...")

    # Create the rollout agent
    rollout_agent = AgentFlowRollout(
        resources=resources,
        llm_engine_name=model_name,
        enabled_tools=enabled_tools,
        tool_engine=tool_engine,
        output_types="direct",
        max_steps=3,
        max_time=300,
        max_tokens=2048,
        base_url=base_url,
        verbose=True,
        temperature=temperature
    )

    # Test questions
    test_questions = [
        {
            "question": "What is the capital of France? Provide the answer in <answer></answer> tags.",
            "expected": "Paris",
            "type": "Simple Factual"
        },
        {
            "question": "Search for information about the Eiffel Tower and tell me when it was built. Provide the answer in <answer></answer> tags.",
            "expected": "1889",
            "type": "Search & Extraction"
        },
        {
            "question": "Calculate 15 * 23 + 47. Show your work and provide the final answer in <answer></answer> tags.",
            "expected": "392",
            "type": "Math Calculation"
        }
    ]

    print(f"\n[INFO] Running {len(test_questions)} test rollouts...\n")

    results = []
    for i, test in enumerate(test_questions):
        print("=" * 80)
        print(f"TEST {i+1}/{len(test_questions)}: {test['type']}")
        print("=" * 80)
        print(f"Question: {test['question']}")
        print(f"Expected: {test['expected']}")
        print("-" * 80)

        try:
            result = rollout_agent.solve(question=test['question'])

            # Extract answer
            import re
            if "direct_output" in result and result["direct_output"]:
                final_output = result["direct_output"]
                all_matches = re.findall(r"<answer>(.*?)</answer>", final_output, re.DOTALL)
                if all_matches:
                    answer = all_matches[-1].strip()
                else:
                    answer = final_output
            else:
                answer = "No answer generated"

            print(f"\n[RESULT]")
            print(f"  Answer: {answer}")
            print(f"  Expected: {test['expected']}")

            # Check if triplets are available (for process reward)
            has_triplets = "triplets" in result and result["triplets"] is not None and len(result["triplets"]) > 0
            print(f"  Has Triplets (for process reward): {has_triplets}")
            if has_triplets:
                print(f"  Number of turns: {len(result['triplets'])}")

            results.append({
                "question": test['question'],
                "type": test['type'],
                "expected": test['expected'],
                "answer": answer,
                "has_triplets": has_triplets,
                "success": True
            })

        except Exception as e:
            print(f"\n[ERROR] Rollout failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append({
                "question": test['question'],
                "type": test['type'],
                "error": str(e),
                "success": False
            })

        print("\n")

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    successful = sum(1 for r in results if r.get("success", False))
    with_triplets = sum(1 for r in results if r.get("has_triplets", False))

    print(f"Total Tests: {len(results)}")
    print(f"Successful: {successful}/{len(results)}")
    print(f"With Triplets (process reward ready): {with_triplets}/{successful}")

    print("\n[PROCESS REWARD STATUS]")
    if with_triplets > 0:
        print("✓ Process reward mode data is available (triplets generated)")
        print("✓ The rollout is compatible with process reward evaluation")
    else:
        print("✗ No triplets generated - may fall back to discount mode")

    print("\n" + "=" * 80)
    print("Test completed at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)

    return results


async def test_process_reward_computation():
    """Test the process reward computation functions directly."""
    print("\n" + "=" * 80)
    print("TESTING PROCESS REWARD COMPUTATION FUNCTIONS")
    print("=" * 80)

    # Test if we can import the required functions
    try:
        from train.utils import compute_turn_score
        print("\n✓ Successfully imported compute_turn_score from train.utils")

        # Test a sample turn evaluation
        print("\n[TEST] Evaluating a sample turn...")
        test_score = compute_turn_score(
            turn_index=0,
            action_planner_response="I will use Google Search to find the capital of France.",
            tools_used=["Google_Search_Tool"],
            memory_context="User asked: What is the capital of France?",
            is_final_turn=False,
            final_answer_correct=None
        )
        print(f"✓ Sample turn score: {test_score:.3f} (scale 0-1)")
        print("✓ Process reward computation is working!")

        return True

    except ImportError as e:
        print(f"\n✗ Failed to import compute_turn_score: {e}")
        print("  Process reward mode may fall back to discount mode")
        return False
    except Exception as e:
        print(f"\n✗ Error testing process reward computation: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("\n" + "=" * 80)
    print("AGENTFLOW PROCESS REWARD TEST SUITE")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Test 1: Process reward computation functions
    print("\n\nTest 1: Process Reward Computation Functions")
    print("-" * 80)
    process_reward_ok = await test_process_reward_computation()

    # Test 2: Rollout with process reward
    print("\n\nTest 2: Rollout with Deployed Model")
    print("-" * 80)
    rollout_results = await test_single_rollout()

    # Final report
    print("\n\n" + "=" * 80)
    print("FINAL TEST REPORT")
    print("=" * 80)
    print(f"Process Reward Functions: {'✓ OK' if process_reward_ok else '✗ FAILED'}")
    print(f"Rollout Tests: {sum(1 for r in rollout_results if r.get('success', False))}/{len(rollout_results)} passed")

    if process_reward_ok and any(r.get('has_triplets', False) for r in rollout_results):
        print("\n✓✓✓ ALL SYSTEMS READY FOR PROCESS REWARD MODE ✓✓✓")
    else:
        print("\n⚠ Some components need attention")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
