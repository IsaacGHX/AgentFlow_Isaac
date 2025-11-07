#!/bin/bash
# Usage: bash test/exp/run_all_models_all_datasets.sh
# Model format: "port:modelname,label,enabled_tools,tool_engines"
# Example: "8000:vllm-IPF/model1,label1,Tool1|Tool2,engine1|engine2"

############ Configuration ############
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Set project directory to test/ folder (parent of exp/)
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
THREADS=16

# Define all tasks to run
TASKS=(
    "gameof24"
    "aime24"
    "amc23"
    "bamboogle"
    "2wiki"
    "gaia"
    "musique"
    "hotpotqa"
    "medqa"
    "gpqa"
)

# Define models with their tool configurations in format:
# "port:modelname,label,enabled_tools,tool_engines"
# - enabled_tools: use | as separator (will be converted to comma)
# - tool_engines: use | as separator (will be converted to comma)
# Example: "8000:vllm-IPF/model,label,Tool1|Tool2|Tool3,engine1|engine2|Default"
MODELS=(
    # "8000:vllm-AgentFlow/agentflow-planner-7b,AgentFlow-7B,Base_Generator_Tool|Python_Coder_Tool|Google_Search_Tool|Wikipedia_Search_Tool,dashscope-qwen2.5-7b-instruct|dashscope-qwen2.5-7b-instruct|Default|Default"
    # "8000:vllm-checkpoints/AgentFlow_general_qwen2.5-3B/rollout_all_qwen2.5-3B_discount_Reward/global_step_18/actor/huggingface,3B_discount_Reward,Base_Generator_Tool|Python_Coder_Tool|Google_Search_Tool|Wikipedia_Search_Tool,dashscope-qwen2.5-3b-instruct|dashscope-qwen2.5-3b-instruct|Default|Default"
    # "8001:vllm-checkpoints/AgentFlow_process_Reward_qwen2.5-3B/rollout_all_qwen2.5-3B_process_Reward/global_step_18/actor/huggingface,3B_process_Reward,Base_Generator_Tool|Python_Coder_Tool|Google_Search_Tool|Wikipedia_Search_Tool,dashscope-qwen2.5-3b-instruct|dashscope-qwen2.5-3b-instruct|Default|Default"
    # "8001:vllm-checkpoints/AgentFlow_process_Reward_qwen2.5-7B/rollout_all_qwen2.5-7B_process_Reward/global_step_12/actor/huggingface,7B_process_Reward,Base_Generator_Tool|Python_Coder_Tool|Google_Search_Tool|Wikipedia_Search_Tool,dashscope-qwen2.5-7b-instruct|dashscope-qwen2.5-7b-instruct|Default|Default"
    # "8003:vllm-checkpoints/AgentFlow_discount_Reward_qwen2.5-7B/rollout_all_qwen2.5-7B_discount_Reward/global_step_12/actor/huggingface,7B_discount_Reward,Base_Generator_Tool|Python_Coder_Tool|Google_Search_Tool|Wikipedia_Search_Tool,dashscope-qwen2.5-7b-instruct|dashscope-qwen2.5-7b-instruct|Default|Default"
    # "8000:vllm-checkpoints/AgentFlow_reverse_discount_Reward_qwen2.5-3B/rollout_all_qwen2.5-3B_reverse_discount_Reward/global_step_18/actor/huggingface,3B_reverse_discount_Reward,Base_Generator_Tool|Python_Coder_Tool|Google_Search_Tool|Wikipedia_Search_Tool,dashscope-qwen2.5-3b-instruct|dashscope-qwen2.5-3b-instruct|Default|Default"
    # "8001:vllm-checkpoints/AgentFlow_reverse_discount_Reward_qwen2.5-7B/rollout_all_qwen2.5-7B_reverse_discount_Reward/global_step_34/actor/huggingface,7B_reverse_discount_Reward,Base_Generator_Tool|Python_Coder_Tool|Google_Search_Tool|Wikipedia_Search_Tool,dashscope-qwen2.5-7b-instruct|dashscope-qwen2.5-7b-instruct|Default|Default"
    # "8004:vllm-checkpoints/AgentFlow_reverse_discount_Reward_qwen2.5-7B/rollout_all_qwen2.5-7B_reverse_discount_Reward/global_step_34/actor/huggingface,7B_reverse_discount_Reward,Base_Generator_Tool|Python_Coder_Tool|Google_Search_Tool|Wikipedia_Search_Tool,dashscope-qwen2.5-7b-instruct|dashscope-qwen2.5-7b-instruct|Default|Default"
    # "8005:vllm-checkpoints/AgentFlow_Reward_qwen2.5-1.5B/rollout_all_qwen2.5-1.5B/global_step_34/actor/huggingface,qwen2.5-1.5B,Base_Generator_Tool|Python_Coder_Tool|Google_Search_Tool|Wikipedia_Search_Tool,dashscope-qwen2-1.5b-instruct|dashscope-qwen2-1.5b-instruct|Default|Default"
    # "8005:vllm-Qwen/Qwen2.5-1.5B-Instruct,qwen2.5-1.5B-naive,Base_Generator_Tool|Python_Coder_Tool|Google_Search_Tool|Wikipedia_Search_Tool,dashscope-qwen2-1.5b-instruct|dashscope-qwen2-1.5b-instruct|Default|Default"
    # "8006:vllm-checkpoints/AgentFlow_Llama-3.2-3B/rollout_all_Llama-3.2-3B/global_step_28/actor/huggingface,Llama-3.2-3B,Base_Generator_Tool|Python_Coder_Tool|Google_Search_Tool|Wikipedia_Search_Tool,together-meta-llama/Llama-3.2-3B-Instruct-Turbo|together-meta-llama/Llama-3.2-3B-Instruct-Turbo|Default|Default"
    # ":together-meta-llama/Llama-3.2-3B-Instruct-Turbo,Llama-3.2-3B-naive,Base_Generator_Tool|Python_Coder_Tool|Google_Search_Tool|Wikipedia_Search_Tool,together-meta-llama/Llama-3.2-3B-Instruct-Turbo|together-meta-llama/Llama-3.2-3B-Instruct-Turbo|Default|Default"
    # "8007:vllm-checkpoints/AgentFlow_Llama-3.1-8B/rollout_all_Llama-3.1-8B/global_step_18/actor/huggingface,Llama-3.1-8B,Base_Generator_Tool|Python_Coder_Tool|Google_Search_Tool|Wikipedia_Search_Tool,together-meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo|together-meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo|Default|Default"
    # ":together-meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo,Llama-3.1-8B-naive,Base_Generator_Tool|Python_Coder_Tool|Google_Search_Tool|Wikipedia_Search_Tool,together-meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo|together-meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo|Default|Default"
    "8007:vllm-checkpoints/AgentFlow_Qwen2.5-7B-turn5/rollout_all_Qwen2.5-7B-turn5/global_step_26/actor/huggingface,qwen2.5-7b-turn5,Base_Generator_Tool|Python_Coder_Tool|Google_Search_Tool|Wikipedia_Search_Tool,dashscope-qwen2.5-7b-instruct|dashscope-qwen2.5-7b-instruct|Default|Default"
)

DATA_FILE_NAME="data.json"

############################################

cd $PROJECT_DIR

# Loop through all models
for MODEL_SPEC in "${MODELS[@]}"; do
    # Parse model specification: port:modelname,label,enabled_tools,tool_engines
    PORT=$(echo "$MODEL_SPEC" | cut -d':' -f1)
    REST=$(echo "$MODEL_SPEC" | cut -d':' -f2-)

    # Split by comma to get individual fields
    IFS=',' read -r LLM LABEL ENABLED_TOOLS_RAW TOOL_ENGINE_RAW <<< "$REST"

    # Convert | separators to commas for tool configuration
    ENABLED_TOOLS=$(echo "$ENABLED_TOOLS_RAW" | tr '|' ',')
    TOOL_ENGINE=$(echo "$TOOL_ENGINE_RAW" | tr '|' ',')

    # Only set BASE_URL if PORT is not empty
    if [ -n "$PORT" ]; then
        BASE_URL="http://localhost:${PORT}/v1"
        BASE_URL_ARG="--base_url \"$BASE_URL\""
    else
        BASE_URL=""
        BASE_URL_ARG=""
    fi

    echo "========================================"
    echo "MODEL: $LLM"
    echo "LABEL: $LABEL"
    echo "BASE_URL: ${BASE_URL:-'(not set)'}"
    echo "ENABLED_TOOLS: $ENABLED_TOOLS"
    echo "TOOL_ENGINE: $TOOL_ENGINE"
    echo "========================================"

    # Loop through all tasks
    for TASK in "${TASKS[@]}"; do
        echo "============================="
        echo "Starting task: $TASK with model: $LABEL"
        echo "============================="

        DATA_FILE="$TASK/data/$DATA_FILE_NAME"
        LOG_DIR="$TASK/logs/$LABEL"
        OUT_DIR="$TASK/results/$LABEL"
        CACHE_DIR="$TASK/cache"

        mkdir -p "$LOG_DIR"
        mkdir -p "$OUT_DIR"

        # Define indices based on task
        case $TASK in
            bamboogle)
                indices=($(seq 0 124))   # 0~124
                ;;
            gaia)
                indices=($(seq 0 126))   # 0~126
                ;;
            aime24)
                indices=($(seq 0 29))
                ;;
            amc23)
                indices=($(seq 0 39))
                ;;
            2wiki|gameof24|nq|musique|hotpotqa|popqa|medqa|gpqa)
                indices=($(seq 0 99))
                ;;
            *)
                indices=(0)  # dafault
                ;;
        esac

        # Skip already completed indices
        new_indices=()
        for i in "${indices[@]}"; do
            if [ ! -f "$OUT_DIR/output_$i.json" ]; then
                new_indices+=($i)
            else
                echo "Output file already exists: $OUT_DIR/output_$i.json"
            fi
        done
        indices=("${new_indices[@]}")
        echo "Final indices: ${indices[@]}"

        if [ ${#indices[@]} -eq 0 ]; then
            echo "All subtasks completed for $TASK with $LABEL."
        else
            # Define single task function
            echo "Using model: $LLM"
            run_task() {
                local i=$1
                echo "Running $TASK for index $i"

                # Build command with optional --base_url parameter
                local cmd="python solve.py \
                --index $i \
                --task $TASK \
                --data_file $DATA_FILE \
                --llm_engine_name $LLM \
                --root_cache_dir $CACHE_DIR \
                --output_json_dir $OUT_DIR \
                --output_types direct \
                --enabled_tools \"$ENABLED_TOOLS\" \
                --tool_engine \"$TOOL_ENGINE\" \
                --max_time 300 \
                --max_steps 10 \
                --temperature 0.7"

                # Add --base_url only if BASE_URL is not empty
                if [ -n "$BASE_URL" ]; then
                    cmd="$cmd --base_url \"$BASE_URL\""
                fi

                # Execute command
                eval "$cmd" 2>&1 | tee "$LOG_DIR/$i.log"

                echo "Completed $TASK for index $i"
                echo "------------------------"
            }
            export -f run_task
            export TASK DATA_FILE LOG_DIR OUT_DIR CACHE_DIR LLM ENABLED_TOOLS TOOL_ENGINE BASE_URL

            echo "Starting parallel execution for $TASK..."
            parallel -j $THREADS run_task ::: "${indices[@]}"
            echo "All subtasks completed for $TASK with $LABEL."
        fi

        ############ Calculate Scores ############
        RESPONSE_TYPE="direct_output"
        python calculate_score_unified.py \
        --task_name $TASK \
        --data_file $DATA_FILE \
        --result_dir $OUT_DIR \
        --response_type $RESPONSE_TYPE \
        --output_file "finalresults_$RESPONSE_TYPE.json" \
        | tee "$OUT_DIR/finalscore_$RESPONSE_TYPE.log"

    done

    echo "========================================"
    echo "Completed all tasks for model: $LABEL"
    echo "========================================"
done

echo "============================="
echo "All models and tasks finished!"
echo "============================="