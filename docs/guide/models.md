# 底座与教师选型

选型结论已通过 **ModelScope** 真实下载核实（非纸面假设）。

## 角色分配

| 角色 | 模型 (ModelScope ID) | 参数量 | 许可证 | 说明 |
|------|----------------------|--------|--------|------|
| 学生 / 产物 | `Qwen/Qwen3.5-0.8B-Base` | 0.8B | Apache 2.0 | 原生多模态 VLM，端侧；最终要部署的 MaiLLM |
| Qwen 教师 | `Qwen/Qwen3.5-4B` | 4B | Apache 2.0 | 同家族 VLM（同词表 248320），中文 / 推理数据 |
| Gemma 教师 | `google/gemma-4-E4B` | 4B | Apache 2.0 | 跨家族多模态（词表 262144），英文 / 多语言 |

## 关键实测事实

1. **三者全是多模态 VLM**（`*ForConditionalGeneration`）。Qwen3.5 与学生在同一词表家族，
   Gemma 词表不同 → 跨家族只能用**响应蒸馏**（不共享权重）。
2. **Qwen 3.5 无 8B**（record not found）；Qwen3.5 是 VLM 家族（带 Vision Encoder）。
3. **Gemma4-E4B 本机加载约束**（实测）：bnb **4-bit 不容许 offload**（`device_map` 报 ValueError），
   故 E4B **不能用 4-bit**。但 **fp16 全精度 + `device_map="auto"` + `max_memory` 自动 offload
   溢出层到 CPU 内存**可跑通（本机 29.7GB RAM + 虚拟内存兜底）。英文蒸馏数据改由 Gemma 生成（或同家族 Qwen 代生成）。
4. **Qwen3.5-4B 教师可用 4-bit** 直接放 GPU，省显存。

## 选型逻辑

- **中文 / 推理 → Qwen 教师**：同家族，词表一致，可上实时 logits 蒸馏；中文质量最强。
- **英文 / 多语言 / 多模态 → Gemma 教师**：Gemma 英文与多模态更强；通过响应蒸馏注入。
- **学生骨架固定 Qwen**：保证最终产物词表、生态、中文能力可控，部署链路成熟。

## 为什么不直接融合（MergeKit）

- 同家族融合（两个 Qwen）像"同套乐高"，可靠。
- **不同家族（Qwen + Gemma）能融但容易"歪"**：参数互相干扰、知识抵消，除非同尺寸 SLERP。
- 架构 / 词表不兼容，`tokenizer_source: union` 仅实验性，lm_head/embed 只能线性融合，效果有限。
- **结论**：用蒸馏而非融合来结合两家之长（详见 [训练方案研究](/guide/research)）。
