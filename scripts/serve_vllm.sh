#!/bin/bash
# scripts/serve_vllm.sh
# ===========================================================================
# Script: serve_vllm.sh
# Description:
#   Launch model using vLLM in a tmux window
#   - Uses GPU 0
#   - tensor-parallel-size=1
#   - Port 8000
# ===========================================================================

# MODEL="checkpoints/AgentFlow_reverse_discount_Reward_qwen2.5-7B/rollout_all_qwen2.5-7B_reverse_discount_Reward/global_step_34/actor/huggingface"
# MODEL="checkpoints/AgentFlow_Reward_qwen2.5-1.5B/rollout_all_qwen2.5-1.5B/global_step_34/actor/huggingface"
# MODEL="checkpoints/AgentFlow_Llama-3.2-3B/rollout_all_Llama-3.2-3B/global_step_28/actor/huggingface"
# MODEL="checkpoints/AgentFlow_Llama-3.1-8B/rollout_all_Llama-3.1-8B/global_step_18/actor/huggingface"
# MODEL="Qwen/Qwen2.5-1.5B-Instruct"
MODEL="checkpoints/AgentFlow_Qwen2.5-7B-turn5/rollout_all_Qwen2.5-7B-turn5/global_step_26/actor/huggingface"
# GPU="4"
# GPU="5"
# GPU="6"
GPU="7"
# PORT=8004
# PORT=8005
# PORT=8006
PORT=8007
# TMUX_SESSION="vllm_agentflow4"
# TMUX_SESSION="vllm_agentflow5"
# TMUX_SESSION="vllm_agentflow6"
TMUX_SESSION="vllm_agentflow7"
TP=1

VENV_ACTIVATE="source .venv/bin/activate"

echo "Launching model: $MODEL"
echo "  Port: $PORT"
echo "  GPU: $GPU"
echo "  Tensor Parallel Size: $TP"

# Create tmux session and run vLLM
tmux new-session -d -s "$TMUX_SESSION"

CMD_START="
    $VENV_ACTIVATE;
    export CUDA_VISIBLE_DEVICES=$GPU;
    echo '--- Starting $MODEL on port $PORT with TP=$TP ---';
    echo 'CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES';
    echo 'Current virtual env: \$(python -c \"import sys; print(sys.prefix)\")';
    vllm serve \"$MODEL\" \
        --host 0.0.0.0 \
        --port $PORT \
        --tensor-parallel-size $TP
"

tmux send-keys -t "${TMUX_SESSION}:0" "$CMD_START" C-m

echo ""
echo "✅ Model launched in tmux session: '$TMUX_SESSION'"
echo "💡 View logs:   tmux attach-session -t $TMUX_SESSION"
echo "💡 Detach:      Ctrl+B, then D"
echo "💡 Kill session: tmux kill-session -t $TMUX_SESSION"
