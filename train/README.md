# MaiLLM 训练链路运行指南

> 目标：基于 **Qwen 3.5**（阿里，最新端侧多模态 VLM）与 **Gemma 4**（Google，最新端侧多模态）训练出一个全新的端侧多模态小模型 **MaiLLM**。
> 硬件：NVIDIA RTX 4060 Laptop 8GB · CUDA 12.8 · 模型全部存于 G 盘（C 盘仅 7GB 不可用）。

## 一、选型结论（已核实 ModelScope 真实可用）

| 角色 | 模型 (ModelScope ID) | 说明 |
|------|----------------------|------|
| 学生 / 产物 | `Qwen/Qwen3.5-0.8B-Base` | 0.8B 多模态 VLM，Apache 2.0，端侧；这是最终要部署的 MaiLLM |
| Qwen 教师 | `Qwen/Qwen3.5-4B` | 同家族 VLM（同词表 248320），用于中文/推理数据生成 |
| Gemma 教师 | `google/gemma-4-E4B` | 跨家族多模态（词表 262144），用于英文/多语言响应蒸馏 |

> 关键实测：三者**全是多模态 VLM**（`*ForConditionalGeneration`）。Qwen3.5 与学生在同一词表家族，Gemma 词表不同 → 跨家族只能用**响应蒸馏**（不共享权重）。因此 MaiLLM 天然是多模态端侧模型。

## 二、环境

```bash
# 项目专用 venv (managed python 3.13, 已建好)
VENV="G:/workspace/MaiLLM/venv"
```
依赖（torch cu128 + transformers 5.14.1 / peft 0.20 / bitsandbytes 0.50 / trl 1.9 / modelscope 1.39）已安装完毕，CUDA 可用。

> ⚠️ **离线加载**：所有模型从 ModelScope 下到本地目录，**必须离线加载**。`from_pretrained` 不能直接用 repo id（会去连 hf-mirror.com，连不通）。脚本已统一设置 `TRANSFORMERS_OFFLINE=1` 并走本地路径。

## 三、执行顺序

```bash
PY="G:/workspace/MaiLLM/venv/Scripts/python.exe"

# 1) 下载三个模型 (约 20GB, 存到 G:\workspace\MaiLLM\models)
$PY train/download_models.py

# 2) 双教师离线生成蒸馏数据 (响应蒸馏, 顺序加载以省显存)
$PY train/generate_data.py qwen      # 中文/推理 -> data/kd_qwen_zh.jsonl
$PY train/generate_data.py gemma     # 英文/多语言 -> data/kd_gemma_en.jsonl
$PY train/generate_data.py combine   # 合并 -> data/maillm_train.jsonl
# (或一步到位: $PY train/generate_data.py all)

# 3) 冒烟测试 (前向+反向) 确认 4bit+LoRA 在 8GB 上可跑
$PY train/train_maillm.py --smoke

# 4) 全量训练 MaiLLM (QLoRA)
$PY train/train_maillm.py            # 产出 outputs/maillm_lora

# 5) 合并 LoRA 回底座, 导出可部署模型
$PY train/export_maillm.py --export-only   # 产出 outputs/maillm_merged
```

## 四、方法学要点

- **跨架构结合靠蒸馏，不靠权重融合**：Qwen 与 Gemma 架构/词表不同，直接 MergeKit 融合会互相干扰。本方案用「Qwen 教师 + Gemma 教师」共同生成高质量 SFT 数据，把两家优点"教"进同一个 Qwen 骨架学生。
- **响应蒸馏（本次默认）**：教师仅用于离线生成答案，训练时只需学生 + LoRA，8GB 显存即可。
- **量化部署**：训练产物为 bf16 标准 HF 模型；端侧部署用 llama.cpp 转 GGUF 后 `q4_k_m`（约 0.5–1GB），目标 ≥30–50 tok/s 移动端。

## 五、已知风险

- Qwen 3.5 / Gemma 4 为 2026 新架构，需较新 `transformers`（已装 5.14.1，原生支持 `Qwen3_5ForConditionalGeneration` / `Gemma4ForConditionalGeneration`）。
- 8GB 显存下训练 0.8B + LoRA 很宽松；教师 4bit 顺序加载（Qwen4B ~2GB、Gemma E4B ~2.5GB）也可容纳。
- 蒸馏≠造知识：新事实靠 RAG；本数据仅为风格/格式/已有能力的重组与对齐。
