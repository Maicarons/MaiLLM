# -*- coding: utf-8 -*-
"""
MaiLLM 训练链路 - 中央配置
基于本机 GPU: NVIDIA RTX 4060 Laptop (8GB VRAM, CUDA 12.8)
所有模型均从 ModelScope 下载，存放于 G 盘（C 盘仅剩 7GB 不可用）。

选型结论（已核实 ModelScope 实际可用）：
- 学生(产物): Qwen/Qwen3.5-0.8B-Base  —— 原生多模态 VLM, 0.8B, Apache2.0, 端侧
- Qwen 教师:   Qwen/Qwen3.5-4B         —— 同家族 VLM, 4B, 用于同家族 logits 蒸馏 + 中文/推理数据
- Gemma 教师:  google/gemma-4-E4B     —— 跨家族多模态, 用于英文/多语言响应蒸馏
注: Qwen3.5 无 8B (record not found); Qwen3.5 是 VLM 家族(带 Vision Encoder)。
"""
import os

WORKSPACE = r"G:\workspace\MaiLLM"
MODELS_DIR = os.path.join(WORKSPACE, "models")
DATA_DIR = os.path.join(WORKSPACE, "data")
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- ModelScope 模型 ID ----------
STUDENT_ID = "Qwen/Qwen3.5-0.8B-Base"
QWEN_TEACHER_ID = "Qwen/Qwen3.5-4B"
GEMMA_TEACHER_ID = "google/gemma-4-E4B"

# ---------- 本地路径 ----------
STUDENT_PATH = os.path.join(MODELS_DIR, "Qwen3.5-0.8B-Base")
QWEN_TEACHER_PATH = os.path.join(MODELS_DIR, "Qwen3.5-4B")
GEMMA_TEACHER_PATH = os.path.join(MODELS_DIR, "gemma-4-E4B")

# ---------- 蒸馏数据 ----------
SEED_PROMPTS_ZH = os.path.join(DATA_DIR, "seed_prompts_zh.jsonl")   # 中文/推理/工具 -> Qwen 教师
SEED_PROMPTS_EN = os.path.join(DATA_DIR, "seed_prompts_en.jsonl")   # 英文/多语言   -> Gemma 教师
KD_DATA_ZH = os.path.join(DATA_DIR, "kd_qwen_zh.jsonl")
KD_DATA_EN = os.path.join(DATA_DIR, "kd_gemma_en.jsonl")
KD_DATA_ALL = os.path.join(DATA_DIR, "maillm_train.jsonl")          # 合并后的训练集

# ---------- 训练超参 (针对 8GB 显存) ----------
MAX_SEQ_LEN = 2048
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
PER_DEVICE_BATCH = 2
GRAD_ACCUM = 8
WARMUP_STEPS = 10
LORA_OUTPUT = os.path.join(OUTPUT_DIR, "maillm_lora")
MERGED_OUTPUT = os.path.join(OUTPUT_DIR, "maillm_merged")

# 是否尝试同家族实时 logits 蒸馏 (需要把 4B 教师也载入 8GB, 较紧; 失败则回退响应蒸馏)
ENABLE_LIVE_LOGITS_KD = True
KD_TEMPERATURE = 5.0
