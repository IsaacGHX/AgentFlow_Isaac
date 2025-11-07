#!/bin/bash

# 脚本用于批量收集所有数据集下指定实验的结果
# 使用方法: ./collect_results.sh <实验名称>
# 示例: ./collect_results.sh 7B_process_Reward

# 检查参数
if [ $# -eq 0 ]; then
    echo "错误: 请提供实验名称"
    echo "使用方法: $0 <实验名称>"
    echo "示例: $0 7B_process_Reward"
    exit 1
fi

EXP_NAME=$1
BASE_DIR="/lambda/nfs/jianwen-us-midwest-1/panlu/iclr/iclr-rebuttal/AgentFlow/test"

echo "========================================"
echo "收集实验结果: ${EXP_NAME}"
echo "========================================"
echo ""

# 遍历所有数据集目录
for dataset_dir in "$BASE_DIR"/*/; do
    dataset_name=$(basename "$dataset_dir")

    # 跳过exp目录本身
    if [ "$dataset_name" == "exp" ]; then
        continue
    fi

    results_path="${dataset_dir}results/${EXP_NAME}"
    score_file="${results_path}/final_scores_direct_output.json"

    # 检查实验结果是否存在
    if [ -f "$score_file" ]; then
        echo "----------------------------------------"
        echo "数据集: ${dataset_name}"
        echo "----------------------------------------"

        # 提取关键指标
        accuracy=$(python3 -c "import json; data=json.load(open('$score_file')); print(f\"{data.get('accuracy', 'N/A'):.2f}\" if isinstance(data.get('accuracy'), (int, float)) else 'N/A')" 2>/dev/null)
        correct=$(python3 -c "import json; data=json.load(open('$score_file')); print(data.get('correct', 'N/A'))" 2>/dev/null)
        total=$(python3 -c "import json; data=json.load(open('$score_file')); print(data.get('total', 'N/A'))" 2>/dev/null)
        avg_step=$(python3 -c "import json; data=json.load(open('$score_file')); print(f\"{data.get('step_stats', {}).get('average_step', 'N/A'):.2f}\" if isinstance(data.get('step_stats', {}).get('average_step'), (int, float)) else 'N/A')" 2>/dev/null)
        avg_time=$(python3 -c "import json; data=json.load(open('$score_file')); print(f\"{data.get('step_stats', {}).get('average_time', 'N/A'):.2f}\" if isinstance(data.get('step_stats', {}).get('average_time'), (int, float)) else 'N/A')" 2>/dev/null)

        echo "准确率: ${accuracy}%"
        echo "正确数/总数: ${correct}/${total}"
        echo "平均步数: ${avg_step}"
        echo "平均时间: ${avg_time}s"
        echo ""
    else
        echo "数据集: ${dataset_name} - 未找到实验结果"
        echo ""
    fi
done

echo "========================================"
echo "结果收集完成"
echo "========================================"
