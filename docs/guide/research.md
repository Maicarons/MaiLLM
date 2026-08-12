# 端侧模型训练方案研究

> 研究日期：2026-08-07
> 研究目标：梳理如何基于 Qwen 与 Gemma 的最新端侧小模型，通过优化训练（蒸馏 / 微调 / 量化 / 融合）得到一个全新的端侧模型 **MaiLLM**。
> 说明：文中 Qwen 3.5 / Gemma 4 等"最新代"信息来自 2026 年 4 月前后的公开博客与模型卡汇总，具体参数请以 HuggingFace / 官方 Model Card 为准；Qwen3 / Gemma 3n / QLoRA / MergeKit 等为已广泛验证的事实。

---

## 一、目标定位：MaiLLM 是什么

MaiLLM 的定位是一款**端侧（on-device）小模型**，目标是在手机、平板、笔记本、边缘网关等内存与算力受限的设备上，以 2–4 GB 量级（INT4）跑出尽量好的中文 + 多语言能力、推理能力与工具调用能力。

关键约束（端侧现实，来自 2026 实测）：
- 手机"甜点区间" = **2–4 GB 模型**（速度 + 质量平衡最佳）。
- 中文场景首选 Qwen 系；推理/多模态场景 Gemma 系更有优势。
- 7B+ 在手机上"勉强"（速度 + 内存压力），端侧主力是 **1–3B** 级别。

---

## 二、底座选择：最新 Qwen 与 Gemma 端侧小模型

### 2.1 Qwen 最新端侧模型

| 代际 | 端侧小尺寸 | 许可证 | 关键特性 |
|------|-----------|--------|---------|
| **Qwen3**（2025-04，已稳定） | 0.6B / 1.7B / 4B / 8B | Apache 2.0 | 混合思考（Hybrid Thinking，`enable_thinking` 一键切换深浅推理）；0.6B 即带 32K 上下文与思考模式；中文最强小模型；原生支持 MCP / 工具调用 |
| **Qwen 3.5**（2026-02~03） | **0.8B** 起，到 397B MoE | Apache 2.0 | Gated Delta Networks 架构；201 种语言（开源最广）；近 100% 多模态训练效率；小模型"性价比突出" |

> 选型提示：做端侧，**Qwen3-0.6B / 1.7B** 是当前最稳妥的中文底座；若想用最新架构且接受验证成本，可选 **Qwen 3.5-0.8B**（需确认官方已放出 0.8B 权重）。

### 2.2 Gemma 最新端侧模型

| 代际 | 端侧小尺寸 | 许可证 | 关键特性 |
|------|-----------|--------|---------|
| **Gemma 3n**（已发布） | E2B / E4B（MatFormer，原生多模态：文本+图像+音频+视频） | Gemma 可商用 | 与 Gemini Nano 同源架构；Per-Layer Embeddings（PLE）把 5B/8B 参数压到 2B/4B 显存占用；MatFormer 可"切"出子模型做质量-延迟权衡；35 语言多模态 / 140+ 语言文本 |
| **Gemma 4**（2026-04-02） | **E2B / E4B**（端侧），26B A4B，31B | **Apache 2.0** | 继承 PLE + 混合注意力（局部滑窗 + 全局）；可配置思考模式；原生 system 提示 + 函数调用；E2B/E4B 与高通/联发科联合端侧优化；小模型上下文 12.8 万 token |

> 选型提示：Gemma 的强项是**端侧多模态（音频/视频/图像）+ 英文/多语言 + 显存效率**。Gemma 4 改为 Apache 2.0 后商用友好度大幅提升。

### 2.3 对比与底座建议

| 维度 | Qwen 系 | Gemma 系 |
|------|---------|----------|
| 中文 | ⭐⭐⭐⭐⭐ 最强 | ⭐⭐⭐ |
| 英文/多语言 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 端侧多模态 | 较弱（Qwen3-VL 有但偏重） | ⭐⭐⭐⭐⭐ 原生音频/视频 |
| 推理/思考 | 混合思考原生 | 可配置思考模式 |
| 显存效率 | 良好 | 极佳（PLE/MatFormer） |
| 许可证 | Apache 2.0 | Gemma 3n 受限 / **Gemma 4 Apache 2.0** |

**结论**：MaiLLM 建议以 **Qwen 系（中文 + 工具调用）为骨架**，把 **Gemma 系（多模态 + 英文/效率）作为"教师/能力源"通过蒸馏注入**，而不是直接做权重融合（见第四节）。

---

## 三、核心优化训练技术

### 3.1 知识蒸馏（KD）——结合两家之长的最自然手段
蒸馏是用大/同尺寸"教师"模型教小"学生"模型，把"暗知识"传下去。三种主流损失：
- **Logits 蒸馏**：KL 散度匹配教师输出分布（经典 KD，Hinton 等）。
- **特征/注意力蒸馏**：拟合中间层隐藏态或注意力图（Feature Distillation / Attention Distillation）。
- **响应蒸馏**：监督生成结果一致性。

> 实证：医学 QA 论文把 QLoRA 微调的 Meditron-7B 当教师、Qwen2.5-3B 当学生，KD 后学生在 MedMCQA/MMLU-Medical 上提升约 **10 个百分点**，部分超过教师。
> 微软 **BitDistill**：把 Qwen3 压到 **1.58-bit（BitNet）**，用"FP16 教师 → 1.58-bit 学生"做 logits + 多头注意力蒸馏，内存降 ~10 倍、CPU 推理提速 ~2.65 倍且性能接近 FP16。实验以 Qwen3-0.6B/1.7B/4B 为基底，替换 Qwen2.5、Gemma 也稳定——证明蒸馏具备**跨架构通用性**。

**跨架构蒸馏是 Qwen + Gemma "结合"的正解**：让 Qwen（中文教师）与 Gemma（多模态/英文教师）共同生成高质量训练信号，去训练一个 Qwen 骨架的学生。两家的优点可以叠加，而无需在权重层面硬凑。

### 3.2 QLoRA / LoRA 微调
- **QLoRA**：4-bit 量化基座 + 只训练少量 LoRA 低秩适配器（Dettmers 等），显存约为全参微调的 **1/4**。
- 典型配置（Unsloth）：`r=16`、`lora_alpha=16`、目标模块 `q/k/v/o_proj + gate/up/down_proj`、`load_in_4bit=True`、优化器 `adamw_8bit`。
- 数据量：窄任务 500 条可动，通用提升建议 ≥1000 条高质量样本；质量 > 数量。
- 硬件：RTX 4090（24GB）用 QLoRA 可微调到 14B；RTX 5090（32GB）有余量上 30B+。

### 3.3 量化（部署前压缩）
| 方法 | 体积 | 精度损失 | 备注 |
|------|------|---------|------|
| INT8 | 减半 | <1% | 所有现代 NPU 支持 |
| INT4 / Q4_K_M | 1/4 | 2–5% | 端侧主力；llama.cpp GGUF |
| FP8 | 接近 FP16 精度 | 极小 | NVIDIA Hopper/Ada |
| 1.58-bit（BitNet） | ~1/10 | 接近 FP16 | 需 BitDistill 类蒸馏恢复 |
| GPTQ / AWQ | 4-bit | 低 | 训练后量化（PTQ） |

**混合精度**：关键层 INT8、其余 INT4，平衡精度与体积。

### 3.4 剪枝
- **结构化剪枝**：剪整头 Attention 或 FFN 神经元（硬件友好）。Qwen2.5-7B 实测可安全移除每层 4–6 个注意力头（参数减 ~12%），需 ~1000 步微调补偿（性能降 3–5%）。
- **非结构化剪枝**：剪单个权重，需要稀疏计算支持。

### 3.5 模型融合（MergeKit）
把多个同源/微调模型合成一个，无需训练。主要方法：
- **Linear（Model Soup）**：加权平均，适合同家族。
- **SLERP**：超球面插值，适合**两个同尺寸、跨家族**模型（如 Qwen3-4B + Gemma3-4B）。
- **Task Arithmetic / TIES / DARE**：任务向量 + 稀疏化 + 符号共识，解决多模型干扰，适合合并多个同源微调。
- 工具：**MergeKit**（`mergekit-yaml config.yaml out_dir`），支持多阶段合并、LoRA 提取、MoE 合并、tokenizer 移植。

---

## 四、关键技术挑战：Qwen 与 Gemma 跨家族如何"结合"

这是本项目最棘手的一点，必须讲清楚：

1. **直接权重融合有风险**。MergeKit 官方明确指出：同家族（如两个 Qwen）融合像"同套乐高"，可靠；**不同家族（Qwen + Gemma）能融但容易"歪"**——除非同尺寸可用 SLERP 平滑插值，否则参数互相干扰、知识抵消。
2. **架构/词表不兼容**。Qwen 与 Gemma 注意力结构、词表、embedding 维度不同。MergeKit 的 `tokenizer_source: union` 可做词表对齐，但仍是实验性，且 lm_head/embed 只能线性融合，效果有限。
3. **推荐的三条结合路径（按可行性排序）**：
   - ✅ **路径 A（首选）：跨架构知识蒸馏**。以 Qwen 系为学生骨架，用 Qwen 大模型 + Gemma 大模型（或多模态 Gemma）共同生成蒸馏数据/信号，把两家优点"教"进一个模型。这是证据最充分的跨家族结合方式（BitDistill 已验证跨 Qwen/Gemma 稳定）。
   - ✅ **路径 B：同骨架多专家微调后融合**。在**同一个 Qwen 底座**上分别做"中文/推理微调"和"多模态/英文能力注入（用 Gemma 生成数据来 SFT）"，得到两个同家族微调，再用 TIES/DARE 合并——规避跨家族权重冲突。
   - ⚠️ **路径 C：FrankenMerge / 层堆叠**。跨家族直接拼层或 SLERP，探索性强、成功率低，仅作实验性尝试，不建议作为主方案。

---

## 五、MaiLLM 推荐训练流水线（端到端）

```
阶段 0  选底座
        └─ 主骨架：Qwen3-1.7B（或 Qwen 3.5-0.8B，若已放出）Apache 2.0
        └─ 教师：Qwen3-14B/32B（中文/推理）+ Gemma 4 E4B（多模态/英文）
阶段 1  造数据
        ├─ 用教师模型生成高质量 SFT 语料（中文指令、推理链、工具调用）
        ├─ Gemma 教师补充多模态/英文/音频相关样本
        └─ 混入领域/风格数据（≥1k 条起步，质量优先）
阶段 2  蒸馏 + QLoRA 微调（Unsloth / Llama-Factory）
        ├─ 4-bit 加载底座，挂 LoRA（r=16~32）
        ├─ 损失 = α·KL(教师‖学生) + β·MSE(隐藏层) + γ·任务损失
        └─ 200~1k steps（按数据量），bf16，adamw_8bit
阶段 3  （可选）同骨架多微调 + MergeKit 融合
        └─ 若分了多个专家微调，用 TIES/DARE 合并，held-out eval 把关
阶段 4  量化压缩
        ├─ llama.cpp 转 GGUF → q4_k_m（端侧默认）
        └─ 激进场景：BitDistill 式 1.58-bit（需蒸馏恢复）
阶段 5  端侧部署验证
        └─ 目标设备跑 tok/s、内存、精度回归测试
```

---

## 六、工具链清单

| 环节 | 工具 | 说明 |
|------|------|------|
| 微调加速 | **Unsloth** | 4-bit QLoRA、省显存、可直接 `save_pretrained_gguf` 导出 Ollama |
| 可视化/配置式微调 | **Llama-Factory** | YAML 驱动，支持嵌套量化 QLoRA、Qwen3-VL，导出 Ollama/LMDeploy |
| 蒸馏/训练框架 | **TRL（SFTTrainer）** + HuggingFace | 配合 Unsloth 实现 KD + SFT |
| 模型融合 | **MergeKit** | linear/slerp/ties/dare，多阶段合并 |
| 量化/推理 | **llama.cpp** | GGUF 转换 + 量化，CPU/端侧事实标准 |
| 苹果端 | **MLX** | Apple Silicon 近原生性能 |
| 安卓/移动 | **ExecuTorch / MediaPipe / TFLite / Google AI Edge** | 官方端侧推理栈 |
| 高性能服务 | **vLLM / SGLang** | OpenAI 兼容 API（训练/评测用） |
| 桌面 GPU | **TensorRT-LLM** | NVIDIA 端侧/桌面 SOTA |

---

## 七、端侧部署目标参考

- **iOS/iPadOS**：MLX / Core ML / ExecuTorch
- **Android**：MediaPipe / ExecuTorch / TFLite / Google AI Edge（Gemma 4 E2B/E4B 已进 AICore 开发者预览）
- **macOS**：MLX / llama.cpp
- **Windows/Linux 桌面**：llama.cpp / ONNX Runtime / TensorRT-LLM
- **甜点体积**：Q4 下 1–2.5 GB（≈ 1.7B~3B 级），目标 ≥ 30–50 tok/s 移动端。

---

## 八、风险与注意事项

1. **跨家族权重融合慎行**：Qwen + Gemma 直接 merge 大概率"歪"，优先走蒸馏路径 A/B。
2. **许可证**：Gemma 3n 许可证受限（商用需确认条款）；**Gemma 4 为 Apache 2.0** 可自由商用。Qwen 系一贯 Apache 2.0。MaiLLM 若含 Gemma 3n 权重衍生，须核对 Gemma 使用条款。
3. **蒸馏≠造知识**：微调/蒸馏只能学风格、格式、已有能力的重组；**新事实靠 RAG，不靠训练**。
4. **评估不能省**：任何 merge / 量化后必须做 held-out 评测，融合会"静默退化"未测能力。
5. **最新代模型需核验**：Qwen 3.5 / Gemma 4 的具体小尺寸权重是否已在 HuggingFace 放出、许可证细节，请以下载页 Model Card 为准。
6. **数据污染**：测试集绝不能进训练集。

---

## 九、参考资料（来源）

- Qwen3 模型家族与混合思考：local-llm.net / CSDN Qwen3 全解析 / MediaTek 天玑部署
- Qwen3-0.6B 端侧推理：aichina.news model card
- Gemma 3n MatFormer / PLE / 多模态：Google DeepMind、thetechbriefs、thevalue.engineering
- Gemma 4（2026-04）：腾讯新闻、ai2.work、agent-engineering.dev（Apache 2.0、E2B/E4B/26B/31B）
- QLoRA + 知识蒸馏（医学 QA）：ScienceDirect (Procedia Computer Science 270, 2025)
- 微软 BitDistill（1.58-bit 蒸馏）：今日头条 / 腾讯新闻 报道
- Edge 端 LLM 推理 2026 技术栈：CSDN Edge 端 LLM 推理
- Qwen2.5 压缩（量化/剪枝/蒸馏）：CSDN
- LLM 模型融合指南（MergeKit）：engineering.zooz.com、ai-infrastructure.net、github.com/miike-ai/mergekit
- Unsloth / Llama-Factory 微调 2026：unsloth.ai 文档、toolhalla.ai、vucense.com、CSDN Qwen3-VL 流程
- 2026 开源模型对比（含 Qwen 3.5 / Gemma 4）：aiproductivity.ai

---

## 十、下一步建议（可执行）

1. **先定底座版本**：确认 Qwen3-1.7B（或 Qwen 3.5-0.8B）与 Gemma 4 E4B 权重/许可证可用性。
2. **跑通最小闭环**：用 Unsloth 在 Qwen3-1.7B 上做一次 QLoRA + KD 小样本实验，验证跨架构蒸馏管线。
3. **建评测集**：中文/英文/推理/工具调用/多模态各一组 held-out 题，作为 MaiLLM 的回归基线。
4. **量化压测**：GGUF q4_k_m 上真机（Android/iOS）测 tok/s 与内存，定最终目标体积。
