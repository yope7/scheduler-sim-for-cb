#!/usr/bin/env python3
"""
分散PCN実行中のGPU使用状況を確認するスクリプト

別ターミナルで実行:
  watch -n 1 python scripts/check_gpu_usage.py

または学習開始後に1回だけ:
  python scripts/check_gpu_usage.py
"""
import subprocess
import sys

def main():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            print("nvidia-smi が実行できません。GPUが利用可能か確認してください。")
            sys.exit(1)
        
        if not result.stdout.strip():
            print("GPUが検出されませんでした。")
            sys.exit(1)
        
        print("=== GPU 使用状況 ===")
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                idx, name, util, mem_used, mem_total = parts[0], parts[1], parts[2], parts[3], parts[4]
                print(f"GPU {idx}: {name}")
                print(f"  使用率: {util}%, メモリ: {mem_used}/{mem_total} MB")
        
        # PyTorchのCUDA確認
        try:
            import torch
            print(f"\nPyTorch CUDA: is_available={torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  デバイス数: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        except ImportError:
            print("\nPyTorchがインポートできません")
            
    except FileNotFoundError:
        print("nvidia-smi が見つかりません。NVIDIAドライバがインストールされているか確認してください。")
        sys.exit(1)

if __name__ == "__main__":
    main()
