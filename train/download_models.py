# -*- coding: utf-8 -*-
"""
从 ModelScope 下载 MaiLLM 所需的三个模型。
用法: python train/download_models.py
依赖: pip install modelscope
所有模型落在 G:\workspace\MaiLLM\models 下（C 盘空间不足，统一用 G 盘）。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maillm_config import (STUDENT_ID, QWEN_TEACHER_ID, GEMMA_TEACHER_ID,
                           STUDENT_PATH, QWEN_TEACHER_PATH, GEMMA_TEACHER_PATH)


def download(model_id: str, local_dir: str):
    print(f"\n===== 下载 {model_id} -> {local_dir} =====")
    os.makedirs(local_dir, exist_ok=True)
    try:
        from modelscope import snapshot_download
        path = snapshot_download(model_id, local_dir=local_dir)
    except Exception as e:
        # 兜底：用命令行方式
        print(f"snapshot_download 失败，尝试 modelscope CLI: {e}")
        import subprocess
        subprocess.run([
            "modelscope", "download", "--model", model_id,
            "--local_dir", local_dir
        ], check=True)
        path = local_dir
    print(f"完成: {path}")
    return path


if __name__ == "__main__":
    download(STUDENT_ID, STUDENT_PATH)
    download(QWEN_TEACHER_ID, QWEN_TEACHER_PATH)
    download(GEMMA_TEACHER_ID, GEMMA_TEACHER_PATH)
    print("\n全部模型下载完成。")
