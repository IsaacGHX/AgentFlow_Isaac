#!/usr/bin/env python3
"""
批量收集所有数据集下指定实验的结果
使用方法: python collect_results.py <实验名称> [--format {table|json|csv}]
示例: python collect_results.py 7B_process_Reward
      python collect_results.py 7B_process_Reward --format table
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path("/lambda/nfs/jianwen-us-midwest-1/panlu/iclr/iclr-rebuttal/AgentFlow/test")


def collect_results(exp_name: str) -> List[Dict]:
    """收集所有数据集的实验结果"""
    results = []

    # 遍历所有数据集目录
    for dataset_dir in sorted(BASE_DIR.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name == 'exp':
            continue

        dataset_name = dataset_dir.name
        score_file = dataset_dir / "results" / exp_name / "final_scores_direct_output.json"

        if score_file.exists():
            try:
                with open(score_file, 'r') as f:
                    data = json.load(f)

                result = {
                    'dataset': dataset_name,
                    'accuracy': data.get('accuracy', None),
                    'correct': data.get('correct', None),
                    'total': data.get('total', None),
                    'average_step': data.get('step_stats', {}).get('average_step', None),
                    'average_time': data.get('step_stats', {}).get('average_time', None),
                    'one_step_rate': data.get('step_stats', {}).get('one_step_rate', None),
                    'path': str(score_file.parent)
                }
                results.append(result)
            except (json.JSONDecodeError, IOError) as e:
                print(f"警告: 读取 {dataset_name} 的结果时出错: {e}")
        else:
            results.append({
                'dataset': dataset_name,
                'accuracy': None,
                'correct': None,
                'total': None,
                'average_step': None,
                'average_time': None,
                'one_step_rate': None,
                'path': None
            })

    return results


def print_table(results: List[Dict], exp_name: str):
    """以表格格式打印结果"""
    print(f"\n{'=' * 100}")
    print(f"实验结果汇总: {exp_name}")
    print(f"{'=' * 100}\n")

    # 打印表头
    header = f"{'数据集':<15} {'准确率':<10} {'正确/总数':<12} {'平均步数':<10} {'平均时间(s)':<12} {'一步解决率':<12}"
    print(header)
    print("-" * 100)

    # 打印每行数据
    for result in results:
        dataset = result['dataset']
        accuracy = f"{result['accuracy']:.2f}%" if result['accuracy'] is not None else "N/A"
        correct_total = f"{result['correct']}/{result['total']}" if result['correct'] is not None else "N/A"
        avg_step = f"{result['average_step']:.2f}" if result['average_step'] is not None else "N/A"
        avg_time = f"{result['average_time']:.2f}" if result['average_time'] is not None else "N/A"
        one_step_rate = f"{result['one_step_rate']:.2f}" if result['one_step_rate'] is not None else "N/A"

        print(f"{dataset:<15} {accuracy:<10} {correct_total:<12} {avg_step:<10} {avg_time:<12} {one_step_rate:<12}")

    print(f"\n{'=' * 100}\n")

    # 计算汇总统计
    valid_results = [r for r in results if r['accuracy'] is not None]
    if valid_results:
        avg_accuracy = sum(r['accuracy'] for r in valid_results) / len(valid_results)
        total_correct = sum(r['correct'] for r in valid_results)
        total_samples = sum(r['total'] for r in valid_results)

        print(f"汇总统计:")
        print(f"  有效数据集数: {len(valid_results)}/{len(results)}")
        print(f"  平均准确率: {avg_accuracy:.2f}%")
        print(f"  总计正确数: {total_correct}/{total_samples}")
        print()


def print_json(results: List[Dict], exp_name: str):
    """以JSON格式打印结果"""
    output = {
        'experiment': exp_name,
        'results': results
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


def print_csv(results: List[Dict], exp_name: str):
    """以CSV格式打印结果"""
    print("dataset,accuracy,correct,total,average_step,average_time,one_step_rate,path")
    for result in results:
        accuracy = result['accuracy'] if result['accuracy'] is not None else ''
        correct = result['correct'] if result['correct'] is not None else ''
        total = result['total'] if result['total'] is not None else ''
        avg_step = result['average_step'] if result['average_step'] is not None else ''
        avg_time = result['average_time'] if result['average_time'] is not None else ''
        one_step_rate = result['one_step_rate'] if result['one_step_rate'] is not None else ''
        path = result['path'] if result['path'] is not None else ''

        print(f"{result['dataset']},{accuracy},{correct},{total},{avg_step},{avg_time},{one_step_rate},{path}")


def main():
    parser = argparse.ArgumentParser(
        description='批量收集所有数据集下指定实验的结果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s 7B_process_Reward
  %(prog)s 7B_process_Reward --format table
  %(prog)s 7B_process_Reward --format json
  %(prog)s 7B_process_Reward --format csv > results.csv
        """
    )
    parser.add_argument('exp_name', help='实验名称，例如: 7B_process_Reward')
    parser.add_argument('--format', '-f', choices=['table', 'json', 'csv'],
                        default='table', help='输出格式 (默认: table)')

    args = parser.parse_args()

    # 收集结果
    results = collect_results(args.exp_name)

    # 根据格式输出
    if args.format == 'table':
        print_table(results, args.exp_name)
    elif args.format == 'json':
        print_json(results, args.exp_name)
    elif args.format == 'csv':
        print_csv(results, args.exp_name)


if __name__ == '__main__':
    main()
