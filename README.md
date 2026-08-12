# MaiLLM · 端侧多模态小模型训练与蒸馏

> 基于 **Qwen3.5** 与 **Gemma4** 最新端侧多模态模型，通过**双教师响应蒸馏 + QLoRA**，
> 在消费级 8GB 显卡（RTX 4060 Laptop）上训练出一个**端侧（on-device）多模态小模型** MaiLLM。

---

## 一、项目定位

MaiLLM 是一套**端侧小模型训练流水线**，而不是一个单一模型文件。它的目标是：

- 以 **Qwen3.5-0.8B-Base**（原生多模态 VLM，Apache 2.0）为**学生 / 产物骨架**；
- 用 **Qwen3.5-4B**（同家族，中文 / 推理教师）与 **Gemma4-E4B**（跨家族，英文 / 多语言教师）
  共同生成高质量蒸馏数据；
- 通过**响应蒸馏（response distillation）+ QLoRA** 把两家模型的优点"教"进 0.8B 学生，
  得到可部署到手机 / 边缘设备的端侧多模态小模型。

核心设计原则：**跨架构结合靠蒸馏，不靠权重融合**。Qwen 与 Gemma 架构 / 词表不同，
直接 MergeKit 融合会互相干扰；蒸馏是证据最充分、最稳妥的跨家族结合路径。

---

## 二、技术路线

```
阶段 0  选底座：Qwen3.5-0.8B-Base（学生）；Qwen3.5-4B / Gemma4-E4B（双教师）
阶段 1  造数据：双教师离线生成响应蒸馏 SFT 语料（中文走 Qwen，英文走 Gemma）
阶段 2  训练：4-bit 量化底座 + LoRA(r=32) QLoRA 微调（响应蒸馏数据）
阶段 3  导出：合并 LoRA 回底座，得到标准 HF 模型
阶段 4  评测：PPL + ROUGE-L/BLEU 对比原始底座，验证蒸馏有效且无灾难性遗忘
阶段 5  部署：llama.cpp 转 GGUF → q4_k_m，目标 2–4GB、≥30–50 tok/s 移动端
```

对应脚本见 `train/` 目录：

| 脚本 | 作用 |
|------|------|
| `download_models.py` | 从 ModelScope 下载三个模型到本地 |
| `generate_data.py` | 双教师离线生成蒸馏数据并合并 |
| `train_maillm.py` | QLoRA 训练（含 `--smoke` 冒烟测试） |
| `export_maillm.py` | 合并 LoRA 回底座，导出可部署模型 |
| `benchmark.py` | 与原始底座对比的基准测试 |

---

## 三、目录结构

```
MaiLLM/
├── train/                 # 训练流水线源码（核心交付物）
│   ├── maillm_config.py   # 中央配置：路径、选型、超参
│   ├── download_models.py
│   ├── generate_data.py
│   ├── train_maillm.py
│   ├── export_maillm.py
│   ├── benchmark.py
│   └── README.md          # 运行指南
├── data/                  # 种子提示词（源码）；生成数据运行后产生（已 gitignore）
├── models/                # 本地模型权重（已 gitignore，不入库）
├── outputs/               # 训练产物 / 合并模型 / 评测报告（已 gitignore）
├── docs/                  # VitePress 文档站点（本仓库文档管理方案）
├── MaiLLM_训练方案研究.md  # 端侧模型训练方案研究（方法论长篇）
└── README.md              # 本文件
```

> ⚠️ **模型权重、训练产物、虚拟环境均不入库**（见 `.gitignore`）。仓库只保留可复现的
> 源码、配置、示例数据与文档。运行脚本后会从 ModelScope 重新下载模型。

---

## 四、快速开始

```bash
# 1. 准备环境（项目专用 venv，已安装 torch cu128 + transformers 5.14.1 等）
VENV="G:/workspace/MaiLLM/venv"
PY="$VENV/Scripts/python.exe"

# 2. 下载模型（约 20GB，存到 G 盘）
$PY train/download_models.py

# 3. 双教师生成蒸馏数据
$PY train/generate_data.py all          # qwen -> gemma -> combine

# 4. 冒烟测试 + 全量训练
$PY train/train_maillm.py --smoke
$PY train/train_maillm.py

# 5. 导出可部署模型
$PY train/export_maillm.py --export-only
```

详细步骤与参数见文档站 **[快速开始](https://maicarons.github.io/MaiLLM/guide/quickstart)**。

---

## 五、文档

本项目文档使用 **VitePress** 统一管理，源码位于 `docs/`：

- 本地预览：`npm install && npm run docs:dev`
- 构建产物：`npm run docs:build`（输出 `docs/.vitepress/dist/`）
- 线上站点：<https://maicarons.github.io/MaiLLM/>

---

## 六、许可证

- 代码：MIT
- 模型权重：遵循各自上游许可证（Qwen 系列 Apache 2.0；Gemma4 Apache 2.0）。
  MaiLLM 蒸馏产物为派生作品，部署前请核对上游许可条款。
- 研究方案报告 `MaiLLM_训练方案研究.md` 仅作方法论参考，以官方 Model Card 为准。
