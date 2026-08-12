# 训练流水线

MaiLLM 的完整流程是一条**端到端、可复现**的流水线，每一阶段都有独立脚本。

```
阶段 0  选底座
        └─ 学生: Qwen3.5-0.8B-Base  |  教师: Qwen3.5-4B, Gemma4-E4B
阶段 1  造数据 (响应蒸馏)
        ├─ Qwen3.5-4B 生成中文/推理/工具类 SFT 样本
        ├─ Gemma4-E4B 生成英文/多语言样本
        └─ combine 合并为训练集 maillm_train.jsonl
阶段 2  QLoRA 训练
        ├─ 4-bit 量化底座 + LoRA(r=32)
        └─ 响应蒸馏数据 SFT，3 epoch
阶段 3  导出
        └─ 合并 LoRA 回底座 → 标准 HF 模型
阶段 4  评测
        └─ PPL(域内/域外) + ROUGE-L/BLEU 对比原始底座
阶段 5  部署
        └─ llama.cpp 转 GGUF → q4_k_m
```

## 阶段与脚本对照

| 阶段 | 脚本 | 关键产出 |
|------|------|---------|
| 0 下载 | `train/download_models.py` | `models/` 下三个本地模型 |
| 1 造数据 | `train/generate_data.py` | `data/kd_qwen_zh.jsonl`、`data/kd_gemma_en.jsonl`、`data/maillm_train.jsonl` |
| 2 训练 | `train/train_maillm.py` | `outputs/maillm_lora/`（LoRA 权重） |
| 3 导出 | `train/export_maillm.py` | `outputs/maillm_merged/`（合并后的标准模型） |
| 4 评测 | `train/benchmark.py` | `outputs/benchmark/report.md` |
| 5 部署 | （外部工具 llama.cpp） | `*.gguf` |

## 为什么是"响应蒸馏"而不是"logits 蒸馏"

- **同家族（Qwen 教师 → Qwen 学生）**：理论可上实时 logits 蒸馏（KL 散度），但需把 4B 教师也载入 8GB，
  较紧；配置 `ENABLE_LIVE_LOGITS_KD` 可尝试，失败自动回退响应蒸馏。
- **跨家族（Gemma 教师 → Qwen 学生）**：词表不同（262144 vs 248320）、架构不同，无法共享 logits，
  只能用**响应蒸馏**——教师离线生成答案，学生只学"输入输出对"。

因此本项目默认采用**响应蒸馏**：训练时只需学生 + LoRA，8GB 显存即可，且天然兼容跨家族教师。

## 数据流示意

```
种子提示词(seed) ──► 教师生成回答 ──► (prompt, response) 对
                                          │
                                          ▼
                              合并为 SFT 训练集
                                          │
                                          ▼
                          学生(Qwen3.5-0.8B) + LoRA 训练
                                          │
                              ┌───────────┴───────────┐
                              ▼                       ▼
                      LoRA 权重              合并回底座(导出)
                                                      │
                                                      ▼
                                          标准 HF 模型 → GGUF → 端侧
```
