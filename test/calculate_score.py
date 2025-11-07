import concurrent.futures
import os, re
import json
import argparse
import tqdm
import sys
from pydantic import BaseModel
from typing import Dict, Tuple

from agentflow.agentflow.engine.openai import ChatOpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import ResultAnalyzer


# ================== Pydantic Models ==================

class AnswerExtraction(BaseModel):
    analysis: str
    extracted_option: str  # e.g., "A. 0.9"


class AnswerVerification(BaseModel):
    analysis: str
    true_false: bool


# ================== Prompt Templates ==================

EXTRACTION_PROMPT = """
You are an impartial judge tasked with extracting the final predicted answer from a model's response to a multiple-choice question.

## Question:
{question}

## Model Response:
{response}

## Answer Choices:
{choices_str}

## Instructions:
1. Carefully read the model’s reasoning and identify the **final concluded answer**.
2. If the model refers to an option letter (A/B/C/D), extract that choice.
3. If the model gives a number (e.g., "Answer: 1"), map it to the corresponding option: 1→A, 2→B, etc.
4. If the model writes the actual content (e.g., "the answer is 0.9") and one of the options contains "0.9", return that full option string.
5. Ignore explanations, disclaimers ("I think", "maybe"), and intermediate steps.
6. Return only the best-matching complete option in the format "X. ...".

## Output Format:
<analysis>: Explain how you located the final answer and which option was selected.
<extracted_option>: One of the provided choices exactly as listed (e.g., "A. 0.9").

Make sure <extracted_option> is one of these:
{choices_str}
"""

VERIFICATION_PROMPT = """
Given the correct answer and the model's extracted prediction, determine if they match semantically.

Correct Answer: {correct_answer}
Extracted Option: {extracted_option}

Instructions:
- Consider them matching if either the option letter or the key content matches.
- Allow small formatting differences (e.g., "0.90" vs "0.9").
- Do NOT consider it correct if the meaning is clearly different.

Response Format:
<analysis>: Briefly explain why it matches or does not match.
<true_false>: True if semantically correct, otherwise False.
"""


# ================== Utility Functions ==================

def find_most_similar_candidate(query: str, candidates: list) -> str:
    """Simple fuzzy matching based on substring or keyword overlap."""
    query_clean = query.lower().strip()
    for opt in candidates:
        if query_clean in opt.lower():
            return opt
    # Fallback: return first candidate containing any digit/number if both are numeric
    try:
        num = re.search(r"\d+\.?\d*", query_clean)
        if num:
            val = num.group()
            for opt in candidates:
                if val in opt:
                    return opt
    except:
        pass
    return candidates[0]  # worst case fallback


# ================== Scorer Class ==================

class ResultScorer:
    def __init__(self, llm_engine=None):
        self.llm_engine = llm_engine or ChatOpenAI(
            model_string="gpt-4o",
            is_multimodal=False,
            enable_cache=True,
            temperature=0.0
        )
        print(f"\nLocal OpenAI engine {self.llm_engine.model_string} initialized.\n")

    def answer_verification(self, question: str, response: str, correct_answer: str, choices: list) -> Tuple[str, bool]:
        # Step 1: Extract raw response inside <answer> tags if present
        all_matches = re.findall(r"<answer>(.*?)</answer>", str(response), re.DOTALL)
        if all_matches:
            response = all_matches[-1].strip()

        # Ensure choices is a list of strings like ["A. 0.9", ...]
        choices_str = "\n".join(choices)

        # --- PHASE 1: Extract the predicted option ---
        extraction_prompt = EXTRACTION_PROMPT.format(
            question=question,
            response=response,
            choices_str=choices_str
        )

        try:
            extraction_result = self.llm_engine(
                extraction_prompt,
                response_format=AnswerExtraction
            )
            extracted_option_raw = extraction_result.extracted_option.strip()
        except Exception as e:
            print(f"[Warning] Extraction failed: {e}")
            extracted_option_raw = response[:50]  # fallback

        # Normalize: make sure we pick an actual option from the list
        extracted_option = find_most_similar_candidate(extracted_option_raw, choices)

        # --- PHASE 2: Verify against correct_answer ---
        verification_prompt = VERIFICATION_PROMPT.format(
            correct_answer=correct_answer,
            extracted_option=extracted_option
        )

        try:
            verification_result = self.llm_engine(
                verification_prompt,
                response_format=AnswerVerification
            )
            final_analysis = verification_result.analysis.strip()
            true_false = verification_result.true_false
        except Exception as e:
            print(f"[Warning] Verification failed: {e}")
            final_analysis = f"Fallback comparison: '{extracted_option}' == '{correct_answer}'"
            true_false = extracted_option == correct_answer

        # Combine all into final analysis
        full_analysis = {
            "extraction_analysis": extraction_result.analysis,
            "extracted_option": extracted_option,
            "verification_analysis": final_analysis,
            "correct_answer": correct_answer,
            "true_false": true_false
        }

        return full_analysis, true_false

    def score_results(self, results: Dict, max_workers=10):
        correct = 0
        total = len(results)

        def process_single_result(pid_data):
            pid, data = pid_data
            question = data.get("question") or data.get("query")
            response = data["response"]
            correct_answer = data["correct_answer"]  # e.g., "A. 0.9"
            choices = data["choices"]  # must exist!

            analysis, is_correct = self.answer_verification(question, response, correct_answer, choices)
            return pid, analysis, is_correct

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_single_result, (pid, data))
                for pid, data in results.items()
                if "choices" in data  # 必须有 choices 才能判断
            ]

            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Scoring results"):
                pid, analysis, is_correct = future.result()
                correct += 1 if is_correct else 0
                results[pid].update({
                    "stepwise_analysis": analysis,
                    "true_false": is_correct
                })

        return results, correct

def load_data(data_file: str, result_dir: str, response_type: str) -> dict:
    """
    加载 benchmark 数据和模型输出结果，并合并为统一格式用于评分
    支持 GPQA、MedQA 等多种数据集格式
    """
    # Load the benchmark data (ground truth)
    with open(data_file, 'r', encoding='utf-8') as f:
        raw_benchmark = json.load(f)

    # Convert to dict by pid
    benchmark_data = {}
    for item in raw_benchmark:
        pid = str(item.get("pid", item.get("idx", "")))

        # Handle different dataset formats
        # GPQA format: has "choices" and "answer_idx"
        if "answer_idx" in item and "choices" in item:
            # Ensure we have full option string as correct_answer
            answer_idx = item["answer_idx"]
            choices = item["choices"]

            # Find the correct answer string
            found = False
            for ch in choices:
                if ch.startswith(f"{chr(65 + answer_idx)}."):  # A., B., C., D.
                    item["correct_answer"] = ch
                    found = True
                    break
            if not found:
                # Fallback: use answer_idx to find it
                if answer_idx < len(choices):
                    item["correct_answer"] = choices[answer_idx]
                else:
                    item["correct_answer"] = choices[0]

        # MedQA format: has "options" and "answer" (text-based)
        elif "options" in item and "answer" in item:
            answer_text = item["answer"]
            options = item["options"]

            # Find which option matches the answer
            found = False
            for opt in options:
                # Extract the text part after the letter (e.g., "A. xxx" -> "xxx")
                opt_text = opt.split(". ", 1)[1] if ". " in opt else opt
                if opt_text.strip() == answer_text.strip():
                    item["correct_answer"] = opt
                    found = True
                    break

            if not found:
                # Fallback: if answer_text matches any option directly
                for opt in options:
                    if answer_text.strip() in opt:
                        item["correct_answer"] = opt
                        found = True
                        break

            if not found:
                # Last resort: just use first option
                item["correct_answer"] = options[0] if options else answer_text

            # Standardize: use "choices" for consistency
            item["choices"] = options

        # Generic fallback
        elif "correct_answer" not in item:
            item["correct_answer"] = str(item.get("answer", ""))

        benchmark_data[pid] = item

    results = {}
    for file in os.listdir(result_dir):
        if file.endswith(".json") and file.startswith("output_"):
            file_path = os.path.join(result_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)

                # Extract index and pid
                index = file.replace(".json", "").replace("output_", "")
                pid = int(index)
                pid_str = str(pid)

                if pid_str not in benchmark_data:
                    print(f"[Warning] PID {pid} not found in benchmark data. Skipping...")
                    continue

                gt_item = benchmark_data[pid_str]

                # Build result entry
                results[pid] = {
                    "pid": pid,
                    "query": gt_item["query"],
                    "question": f"{gt_item['query'].strip()}\n\n" + "\n".join(gt_item.get("choices", [])),
                    "choices": gt_item.get("choices", gt_item.get("options", [])),
                    "correct_answer": gt_item["correct_answer"]
                }

                # Extract model response based on response_type
                if response_type in result:
                    response = result[response_type]
                else:
                    # Fallback: traverse nested structure
                    response = result.get("outputs", {}).get(response_type, "") or result.get("final_output", "")

                results[pid]["response"] = response

            except Exception as e:
                print(f"❌ Failed to load {file}: {e}")

    return dict(sorted(results.items()))  # sort by pid

def parse_args():
    parser = argparse.ArgumentParser(description="Score Benchmarking Results with LLM-as-a-Judge (Two-Stage)")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the benchmark data.json")
    parser.add_argument("--result_dir", type=str, required=True, help="Directory containing output_*.json files")
    parser.add_argument("--output_file", type=str, default="final_results.json", help="File to save detailed results")
    parser.add_argument("--log_dir", type=str, default=None, help="Log directory for step/tool analysis")
    parser.add_argument("--response_type", type=str, default="direct_output",
                        choices=["direct_output", "final_output", "base_response"],
                        help="Which field in result JSON to use as model response")
    parser.add_argument("--max_workers", type=int, default=16, help="Number of parallel workers for scoring")
    parser.add_argument("--dataset", type=str, default="auto",
                        choices=["auto", "gpqa", "medqa", "aime", "amc", "hotpotqa", "2wiki", "musique", "bamboogle", "gaia", "gameof24"],
                        help="Dataset type (auto will infer from data_file path)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Auto-detect dataset type if not specified
    if args.dataset == "auto":
        for ds_name in ["gpqa", "medqa", "aime", "amc", "hotpotqa", "2wiki", "musique", "bamboogle", "gaia", "gameof24"]:
            if ds_name in args.data_file.lower() or ds_name in args.result_dir.lower():
                args.dataset = ds_name
                break
        if args.dataset == "auto":
            args.dataset = "unknown"

    print("#" * 60)
    print(f"# {'Dataset Type':<15} : {args.dataset.upper()}")
    for arg, value in vars(args).items():
        if arg != "dataset":  # Already printed above
            print(f"# {arg:<15} : {value}")
    print("#" * 60)

    # Initialize scorer and analyzer
    scorer = ResultScorer()
    analyzer = ResultAnalyzer()

    # Step 1: Load data
    print(f"\n🔍 Loading data from:")
    print(f"  Dataset:    {args.dataset.upper()}")
    print(f"  Data File:  {args.data_file}")
    print(f"  Result Dir: {args.result_dir}")
    results = load_data(args.data_file, args.result_dir, args.response_type)

    if len(results) == 0:
        print("🛑 No valid results loaded. Exiting.")
        exit(1)

    print(f"✅ Loaded {len(results)} samples.")

    # Step 2: Score results
    print(f"\n🧠 Starting LLM-as-a-judge evaluation using {scorer.llm_engine.model_string}...")
    results, correct = scorer.score_results(results, max_workers=args.max_workers)

    # Step 3: Calculate accuracy
    total = len(results)
    acc = round(correct / total * 100, 2)
    print(f"\n🎉 Accuracy: {acc}% ({correct}/{total})")

    # Step 4: Save detailed results
    output_path = os.path.join(args.result_dir, args.output_file)
    os.makedirs(args.result_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"💾 Results saved to: {output_path}")

    # Step 5: Analyze wrong answers
    wrong_pids = [pid for pid, r in results.items() if not r["true_false"]]
    wrong_pids.sort()
    wrong_indices = [int(pid) for pid in wrong_pids]
    print(f"❌ Wrong PIDs: {wrong_pids}")
    print(f"📉 Wrong Indices: {wrong_indices}")

    # Step 6: Save scores summary
    scores_summary = {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "wrong_pids": wrong_pids,
        "wrong_indices": wrong_indices
    }

    score_file = os.path.join(args.result_dir, f"final_scores_{args.response_type}.json")
    with open(score_file, 'w', encoding='utf-8') as f:
        json.dump(scores_summary, f, indent=4, ensure_ascii=False)
    print(f"📊 Scores saved to: {score_file}")

    # Step 7: Additional stats (time, steps, tools) if log_dir exists
    log_dir = args.log_dir or args.result_dir.replace("results", "logs")
    if os.path.exists(log_dir) and args.response_type != "base_response":
        print(f"\n📈 Calculating additional statistics from logs...")

        step_stats = analyzer.calculate_time_steps(log_dir)
        tool_usage = analyzer.calculate_tool_usage(args.result_dir)

        print("\n⏱️ Step Stats:")
        for k, v in step_stats.items():
            print(f"  - {k}: {v}")

        print("\n🔧 Tool Usage:")
        for tool, ratio in tool_usage.items():
            print(f"  - {tool}: {ratio:.2%}")

        # Update score file
        scores_summary.update({
            "step_stats": step_stats,
            "tool_usage": tool_usage
        })

        # Re-save updated scores
        with open(score_file, 'w', encoding='utf-8') as f:
            json.dump(scores_summary, f, indent=4, ensure_ascii=False)

    elif args.response_type == "base_response":
        print("\nℹ️ Base response mode: Skipping step/tool analysis.")

    print("\n✅ Scoring completed successfully.")