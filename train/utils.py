import re
from pydantic import BaseModel
from agentflow.engine.openai import ChatOpenAI


class AnswerVerification(BaseModel):
    analysis: str
    true_false: bool


class TurnScoreResult(BaseModel):
    reasoning: str
    tool_usage_score: float  # 0-1
    memory_usage_score: float  # 0-1
    correctness_score: float  # 0-1
    overall_score: float  # 0-1

try:
    llm_scorer_engine = ChatOpenAI(
        model_string="gpt-4o", 
        is_multimodal=False, 
        enable_cache=True
    )
    print(f"\nLLM Scorer engine '{llm_scorer_engine.model_string}' initialized successfully.\n")
except Exception as e:
    print(f"Failed to initialize LLM Scorer engine: {e}")
    llm_scorer_engine = None

def compute_score(question: str,  groundtruth: str, answer_extracted: str,) -> bool:
    """
    Uses gpt-4o to determine if the extracted answer matches the groundtruth.
    
    Args:
        question: The full question text, including options.
        answer_extracted: The answer provided by the model being evaluated.
        groundtruth: The correct answer label (e.g., "A").

    Returns:
        A boolean indicating whether the answer is correct.
    """
    if llm_scorer_engine is None:
        raise RuntimeError("LLM Scorer engine is not available.")

    query_prompt = f"""
You are a precise evaluator. Determine if the Model Response is equivalent to the Ground Truth.

**Instructions:**
1.  **Extract:** Isolate the final answer from the Model Response, ignoring reasoning. Look for `\boxed{{...}}` or concluding statements.
2.  **Normalize & Compare:** The extracted answer and Ground Truth must be equivalent after normalization:
    - **Math:** Mathematically identical (e.g., `\\frac{{1}}{{2}}` == `0.5`).
    - **Numbers/Text:** Ignore formatting, case, and currency/units (e.g., `1,000` == `1000`).
    - **MCQ:** Match option content (e.g., "Paris") or number (e.g., `3rd` option) to the correct letter.
3.  **Verdict:** "True" only for semantically or mathematically equivalent answers.

**Inputs:**
Question: {question}
Model Response: {answer_extracted}
Ground Truth: {groundtruth}

**Format:**
<analysis>: Brief analysis of the comparison.
<true_false>: "True" or "False".
"""

    verification_result = llm_scorer_engine(query_prompt, response_format=AnswerVerification)
    
    return verification_result.true_false


def eval(question: str, groundtruth: any, answer_extracted: any, val: bool = False) -> float:
    """
    Evaluates if the extracted answer is correct by calling an LLM judge (gpt-4o).
    It strip(), and matches the final answer.
    """
    question_str = str(question)
    groundtruth_str = str(groundtruth)
    answer_extracted_str = str(answer_extracted)

    is_correct = compute_score(question_str, answer_extracted_str, groundtruth_str)

    return 1.0 if is_correct else 0.0


def compute_turn_score(
    turn_index: int,
    action_planner_response: str,
    tools_used: list,
    memory_context: str,
    is_final_turn: bool,
    final_answer_correct: bool = None
) -> float:
    """
    Uses GPT-4o to evaluate the quality of a single turn's action planner response.

    Args:
        turn_index: The index of the current turn (0-based)
        action_planner_response: The response from the action planner for this turn
        tools_used: List of tools/actions used in this turn
        memory_context: The memory/context available at this turn
        is_final_turn: Whether this is the final turn
        final_answer_correct: Whether the final answer was correct (only for final turn)

    Returns:
        A score between 0 and 1 for this turn
    """
    if llm_scorer_engine is None:
        raise RuntimeError("LLM Scorer engine is not available.")

    # Build the evaluation prompt
    tools_str = ", ".join(tools_used) if tools_used else "None"

    prompt = f"""
You are an expert evaluator of AI agent reasoning steps. Evaluate the quality of this action planner's response for turn {turn_index}.

**Evaluation Criteria:**

1. **Tool/Action Selection (0-1)**:
   - Are the selected tools appropriate for the current task?
   - Is the action logical given the context?

2. **Memory Utilization (0-1)**:
   - Does the response effectively use previous context/memory?
   - Is there good continuity with prior turns?

3. **Step Correctness (0-1)**:
   - Is this step moving toward the solution?
   - Are there logical errors or incorrect reasoning?
   {"- Final answer correctness: " + ("CORRECT" if final_answer_correct else "INCORRECT") if is_final_turn else ""}

**Input Information:**
- Turn Index: {turn_index}
- Is Final Turn: {is_final_turn}
- Tools/Actions Used: {tools_str}
- Memory/Context: {memory_context[:500]}...
- Action Planner Response: {action_planner_response[:1000]}...

**Instructions:**
1. Provide brief reasoning for each criterion
2. Assign scores (0.0 to 1.0) for:
   - tool_usage_score
   - memory_usage_score
   - correctness_score
3. Calculate overall_score as weighted average:
   - If final turn and answer provided: 0.3*tool + 0.2*memory + 0.5*correctness
   - Otherwise: 0.4*tool + 0.3*memory + 0.3*correctness

**Output Format:**
<reasoning>: Your analysis (2-3 sentences)
<tool_usage_score>: float (0.0-1.0)
<memory_usage_score>: float (0.0-1.0)
<correctness_score>: float (0.0-1.0)
<overall_score>: float (0.0-1.0)
"""

    try:
        result = llm_scorer_engine(prompt, response_format=TurnScoreResult)
        return result.overall_score
    except Exception as e:
        print(f"Error evaluating turn {turn_index}: {e}")
        # Fallback: return 0.5 if evaluation fails
        return 0.5


def compute_turn_scores_batch(
    turns_data: list[dict]
) -> list[float]:
    """
    Batch version of compute_turn_score that evaluates multiple turns concurrently.

    This significantly speeds up evaluation when there are multiple turns to score,
    as it sends all requests to GPT-4o in parallel using ThreadPoolExecutor.

    Args:
        turns_data: List of dictionaries, each containing:
            - turn_index: int
            - action_planner_response: str
            - tools_used: list
            - memory_context: str
            - is_final_turn: bool
            - final_answer_correct: bool (optional)

    Returns:
        List of scores (0-1) for each turn, in the same order as input

    Example:
        turns_data = [
            {
                "turn_index": 0,
                "action_planner_response": "I will search...",
                "tools_used": ["Google_Search_Tool"],
                "memory_context": "User asked...",
                "is_final_turn": False,
                "final_answer_correct": None
            },
            {
                "turn_index": 1,
                "action_planner_response": "Based on results...",
                "tools_used": ["Base_Generator_Tool"],
                "memory_context": "Previous search...",
                "is_final_turn": True,
                "final_answer_correct": True
            }
        ]
        scores = compute_turn_scores_batch(turns_data)
        # Returns: [0.85, 0.92]
    """
    if llm_scorer_engine is None:
        raise RuntimeError("LLM Scorer engine is not available.")

    if not turns_data:
        return []

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def score_single_turn(turn_data):
        """Helper function to score a single turn"""
        try:
            return compute_turn_score(
                turn_index=turn_data["turn_index"],
                action_planner_response=turn_data["action_planner_response"],
                tools_used=turn_data["tools_used"],
                memory_context=turn_data["memory_context"],
                is_final_turn=turn_data["is_final_turn"],
                final_answer_correct=turn_data.get("final_answer_correct", None)
            )
        except Exception as e:
            print(f"Error in batch scoring turn {turn_data['turn_index']}: {e}")
            return 0.5  # Fallback score

    # Use ThreadPoolExecutor for concurrent API calls
    # max_workers=10 allows up to 10 concurrent requests to OpenAI
    scores = [None] * len(turns_data)

    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks and keep track of their indices
        future_to_idx = {
            executor.submit(score_single_turn, turn_data): idx
            for idx, turn_data in enumerate(turns_data)
        }

        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                scores[idx] = future.result()
            except Exception as e:
                print(f"Error retrieving result for turn {idx}: {e}")
                scores[idx] = 0.5  # Fallback

    return scores

async def main():
    # ==============================================================================
    # ==============================================================================
    print("--- Running Simple Case ---")
    simple_question = "What is the capital of France?\nA) Berlin\nB) Madrid\nC) Paris\nD) Rome"
    simple_groundtruth = "C"
    simple_model_answer = "The correct answer is C."
    score1 = eval(simple_question, simple_groundtruth, simple_model_answer)
    print(f"Question: {simple_question}")
    print(f"Model Answer: '{simple_model_answer}'")
    print(f"Ground Truth: '{simple_groundtruth}'")
    print(f"==> Score: {score1}\n") # 1.0

    # ==============================================================================
    # ==============================================================================
    print("--- Running Case with LaTeX Formula ---")
    latex_question = r"""
Calculate the definite integral of $f(x) = 2x$ from $x=1$ to $x=3$.
A) 4
B) 6
C) 8
D) 10
"""
    latex_groundtruth = "C"
    latex_model_answer = r"""
To solve this, we need to compute the integral $\int_{1}^{3} 2x \,dx$.
The antiderivative of $2x$ is $x^2$. 
Using the Fundamental Theorem of Calculus, we evaluate this at the bounds:
$F(b) - F(a) = 3^2 - 1^2 = 9 - 1 = 8$.
"""
    score2 = eval(latex_question, latex_groundtruth, latex_model_answer)
    print(f"Question: {latex_question.strip()}")
    print(f"Model Answer: '{latex_model_answer.strip()}'")
    print(f"Ground Truth: '{latex_groundtruth}'")
    print(f"==> Score: {score2}\n") # 1.0

    # ==============================================================================
    # ==============================================================================
    print("--- Running Case with Multiple Intermediate Answers ---")
    multi_answer_question = """
A project has two phases. Phase 1 costs $5,000 and takes 3 months. Phase 2 costs $8,000 and takes 4 months. What is the total duration of the project?
A) $13,000
B) 4 months
C) 7 months
D) $5,000
"""
    multi_answer_groundtruth = "C"
    multi_answer_model_response = """
Let's analyze the problem.
The cost of Phase 1 is $5,000 and the duration is 3 months.
The cost of Phase 2 is $8,000 and the duration is 4 months.
The total cost would be $5,000 + $8,000 = $13,000.
The question asks for the total duration, which is 3 months + 4 months = 7 months.
Therefore, the final answer is 7 months. This matches option C.
"""
    score3 = eval(multi_answer_question, multi_answer_groundtruth, multi_answer_model_response)
    print(f"Question: {multi_answer_question.strip()}")
    print(f"Model Answer: '{multi_answer_model_response.strip()}'")
    print(f"Ground Truth: '{multi_answer_groundtruth}'")
    print(f"==> Score: {score3}\n") # 1.0


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())