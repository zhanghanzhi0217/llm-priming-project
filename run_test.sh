#!/bin/bash
#SBATCH --job-name=llm_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --time=00:30:00
#SBATCH --account=class
#SBATCH --partition=a100_1
#SBATCH --gres=gpu:1

# 创建日志目录
mkdir -p logs

# 加载Anaconda模块
module load anaconda3/2024.02

# 激活conda环境
source ~/.bashrc
conda activate llm_priming

# 运行测试脚本
echo "开始运行环境测试脚本"
python scripts/utils/test_environment.py

echo "测试完成!"
