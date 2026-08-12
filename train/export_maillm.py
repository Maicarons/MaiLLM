# -*- coding: utf-8 -*-
"""
MaiLLM 后处理:
1) merge-only : 把 Qwen 教师 / Gemma 教师两份 KD 数据合并为训练集 maillm_train.jsonl
2) export-only: 把训练得到的 LoRA 合并回底座, 保存为可部署的 MaiLLM 标准 HF 模型

约束: 本地离线加载, 显式 Qwen3_5 架构类, trust_remote_code=False。

用法:
  python train/export_maillm.py --merge-only
  python train/export_maillm.py --export-only
  python train/export_maillm.py
"""
import sys
import os
import json
import argparse
import torch

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maillm_config import (KD_DATA_ZH, KD_DATA_EN, KD_DATA_ALL,
                           STUDENT_PATH, LORA_OUTPUT, MERGED_OUTPUT)


def merge_data():
    count = 0
    with open(KD_DATA_ALL, "w", encoding="utf-8") as out:
        for src in (KD_DATA_ZH, KD_DATA_EN):
            if not os.path.exists(src):
                print(f"[跳过] 不存在: {src}")
                continue
            with open(src, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.write(line + "\n")
                        count += 1
    print(f"训练集合并完成: {count} 条 -> {KD_DATA_ALL}")


def export_model():
    from transformers import (AutoProcessor, Qwen3_5ForConditionalGeneration)
    from peft import PeftModel

    print(f"合并 LoRA 回底座: {STUDENT_PATH} + {LORA_OUTPUT}")
    processor = AutoProcessor.from_pretrained(
        STUDENT_PATH, trust_remote_code=False, local_files_only=True)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        STUDENT_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=False, local_files_only=True)
    model = PeftModel.from_pretrained(base, LORA_OUTPUT)
    model = model.merge_and_unload()
    os.makedirs(MERGED_OUTPUT, exist_ok=True)
    model.save_pretrained(MERGED_OUTPUT)
    processor.save_pretrained(MERGED_OUTPUT)
    print(f"MaiLLM 合并模型已保存: {MERGED_OUTPUT}")

    total = sum(os.path.getsize(os.path.join(MERGED_OUTPUT, f))
                for f in os.listdir(MERGED_OUTPUT)
                if f.endswith((".safetensors", ".bin")))
    print(f"合并模型体积约: {total/1e9:.2f} GB (bf16)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()
    if args.export_only:
        export_model()
    elif args.merge_only:
        merge_data()
    else:
        merge_data()
        export_model()
