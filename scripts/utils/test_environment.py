"""
环境测试脚本 - 验证GPU可用性和库安装
"""
import os
import torch
import transformers
import time
import numpy as np

def test_environment():
    print("=" * 50)
    print("环境测试开始")
    print("=" * 50)
    
    # 1. 基本Python环境
    print(f"Python版本: {os.sys.version}")
    
    # 2. CUDA可用性
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name(0)}")
    
    # 3. PyTorch测试
    print(f"\nPyTorch版本: {torch.__version__}")
    
    # 创建简单张量并移至GPU (如果可用)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 简单矩阵乘法性能测试
    matrix_size = 5000
    print(f"\n执行{matrix_size}x{matrix_size}矩阵乘法...")
    
    # 创建随机矩阵
    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)
    
    # 预热
    torch.matmul(a, b)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # 计时
    start_time = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start_time
    
    print(f"矩阵乘法耗时: {elapsed:.4f}秒")
    
    # 4. Transformers库测试
    print(f"\nTransformers版本: {transformers.__version__}")
    
    print("\n环境测试完成!")
    print("=" * 50)

if __name__ == "__main__":
    test_environment()
