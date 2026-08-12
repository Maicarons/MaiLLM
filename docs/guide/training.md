# 训练详解

本章逐一拆解 `train/` 下的脚本与各阶段超参。

## 中央配置：`maillm_config.py`

所有路径、选型、超参集中于此，便于修改与复现：

```python
STUDENT_ID   = "Qwen/Qwen3.5-0.8B-Base"
QWEN_TEACHER_ID = "Qwen/Qwen3.5-4B"
GEMMA_TEACHER_ID = "google/gemma-4-E4B"

# 训练超参（针对 8GB 显存）
MAX_SEQ_LEN   = 2048
LORA_R        = 32
LORA_ALPHA    = 64
LORA_DROPOUT  = 0.05
TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
LEARNING_RATE = 2e-4
NUM_EPOCHS    = 3
PER_DEVICE_BATCH = 2
GRAD_ACCUM    = 8
WARMUP_STEPS  = 10

ENABLE_LIVE_LOGITS_KD = True   # 同家族实时 logits 蒸馏（失败自动回退响应蒸馏）
KD_TEMPERATURE = 5.0
```

## 1. 下载：`download_models.py`

用 `modelscope.snapshot_download` 把三个模型落到 `models/`。失败兜底走 `modelscope` CLI。

## 2. 造数据：`generate_data.py`

- `qwen` / `gemma` / `qwen_en`：加载对应教师（Qwen 4-bit；Gemma fp16 offload），
  对内置种子提示词生成回答，写成 `(prompt, response)` 对话对 JSONL。
- `combine`：把中文 + 英文两份合并为 `maillm_train.jsonl`（并做 JSON 校验）。
- `all`：依次 `qwen` → `gemma` → `combine`。

种子提示词覆盖：量化、推理数学、工具调用、Python、RAG、注意力机制、散文 / 诗歌、端侧部署等，
确保蒸馏数据覆盖面广、质量优先。

## 3. 训练：`train_maillm.py`

- 学生底座 **4-bit 量化 + `prepare_model_for_kbit_training` + LoRA(r=32)**。
- 数据集 `SFTDataset`：对话 JSONL → input_ids/labels，屏蔽 prompt 部分；
  兼容 base 模型缺 chat template（含 Qwen 风格 `manual_template` 兜底）。
- `--smoke`：仅前向 + 反向 1 步，验证 4-bit+LoRA 在 8GB 可跑后退出。
- 全量：HuggingFace `Trainer`，bf16，梯度检查点，`save_strategy="epoch"`，
  产出 `outputs/maillm_lora/`。

关键约束（已实测）：

- 模型从本地目录离线加载，禁止直连 HF。
- 4-bit + LoRA 训练必须先 `prepare_model_for_kbit_training`。

## 4. 导出：`export_maillm.py`

- `--export-only`：用 `PeftModel.from_pretrained` 加载 LoRA，`merge_and_unload()`
  合并回底座，保存为标准 HF 模型到 `outputs/maillm_merged/`（bf16）。
- 同时 `merge-only` 可单独重做数据合并。

## 5. 评测：`benchmark.py`

见 [基准评测](/guide/benchmark)。

## 训练产物体积参考

- LoRA 适配器：`adapter_model.safetensors` ≈ 51MB（0.8B 的 LoRA r=32）。
- 合并模型（bf16）：`model.safetensors` ≈ 1.7GB；量化后 q4_k_m 约 0.5–1GB。
