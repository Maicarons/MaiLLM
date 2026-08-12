# 快速开始

下面以 **Windows + Git Bash + 项目专用 venv** 为例。所有模型从 ModelScope 下载到本地，
必须**离线加载**（禁止直连 HuggingFace，hf-mirror 连不通）。

## 0. 环境

```bash
# 项目专用 venv（managed python 3.13，已建好；CUDA 12.8）
VENV="G:/workspace/MaiLLM/venv"
PY="$VENV/Scripts/python.exe"
```

依赖（已安装）：`torch cu128` + `transformers 5.14.1` / `peft 0.20` /
`bitsandbytes 0.50` / `trl 1.9` / `modelscope 1.39`。Qwen3.5 / Gemma4 是新架构，
需较新 transformers（5.14.1 原生支持 `Qwen3_5ForConditionalGeneration` / `Gemma4ForConditionalGeneration`）。

> ⚠️ **模型全部存于 G 盘**（C 盘仅剩 ~7GB 不可用）。脚本路径在 `train/maillm_config.py` 中集中管理。

## 1. 下载模型

```bash
$PY train/download_models.py
```

约 20GB，落到 `G:\workspace\MaiLLM\models`（学生 / 两位教师各一份）。

## 2. 双教师生成蒸馏数据

```bash
$PY train/generate_data.py qwen      # 中文/推理 -> data/kd_qwen_zh.jsonl
$PY train/generate_data.py gemma     # 英文/多语言 -> data/kd_gemma_en.jsonl
$PY train/generate_data.py combine   # 合并 -> data/maillm_train.jsonl
# 或一步到位: $PY train/generate_data.py all
```

> 注：Gemma4-E4B 在本机 8GB 显存下无法本地 4-bit 推理（见 [底座与教师选型](/guide/models)），
> 实际英文蒸馏数据可由同家族 Qwen3.5-4B 代生成（`qwen_en` 模式），质量足够。
> 教师顺序加载以省显存，生成完即释放。

## 3. 冒烟测试 + 全量训练

```bash
# 冒烟测试：前向+反向 1 步，确认 4bit+LoRA 在 8GB 上可跑
$PY train/train_maillm.py --smoke

# 全量训练（QLoRA，默认 3 epoch）-> outputs/maillm_lora
$PY train/train_maillm.py
```

## 4. 导出可部署模型

```bash
$PY train/export_maillm.py --export-only   # 合并 LoRA 回底座 -> outputs/maillm_merged
```

合并模型为标准 HF 格式（bf16）。端侧部署见 [端侧部署](/guide/deploy)。

## 5. 基准评测（可选）

```bash
$PY train/benchmark.py            # 全流程（含教师参考生成）
$PY train/benchmark.py --skip-ref # 已有 bench_refs.jsonl 时跳过
```

报告写入 `outputs/benchmark/report.md`，对比 MaiLLM 与原始底座的 PPL / ROUGE-L / BLEU。

## 常见问题

- **`TRANSFORMERS_OFFLINE` 报错 / 模型连网**：脚本已统一设置离线环境变量并走本地路径；
  若仍连网，检查是否误用了 repo id 而非本地路径。
- **显存不足**：教师推理阶段已用 4-bit（Qwen）/ fp16 offload（Gemma）控制；
  训练阶段 0.8B + LoRA 在 8GB 上很宽松。
- **中文字符乱码**：Windows 下确保以 UTF-8 读写 `.jsonl`；本项目脚本均 `encoding="utf-8"`。
