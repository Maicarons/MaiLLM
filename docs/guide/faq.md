# 常见问题与风险

## 训练相关

**Q：8GB 显存够吗？**
A：够。学生 0.8B + LoRA 在 8GB 上很宽松；教师推理阶段 Qwen4B 用 4-bit（~2GB）、
Gemma E4B 用 fp16 + offload 到 CPU 内存（本机 29.7GB RAM 兜底）。

**Q：Gemma4-E4B 为什么不用 4-bit？**
A：实测 bnb 4-bit 不容许 offload（`device_map` 报 ValueError）。改用 fp16 全精度 +
`device_map="auto"` + `max_memory` 自动把溢出层 offload 到 CPU，可跑通。
英文蒸馏数据也可由同家族 Qwen3.5-4B 代生成。

**Q：可以上实时 logits 蒸馏吗？**
A：同家族（Qwen 教师 → Qwen 学生）可以，配置 `ENABLE_LIVE_LOGITS_KD=True` 尝试；
跨家族（Gemma → Qwen）词表 / 架构不同，只能用响应蒸馏。

**Q：为什么训练数据只有 25 条？**
A：这是小样本快速验证初版。质量 > 数量；后续可扩充到 ≥1000 条高质量样本。

## 工程相关

**Q：为什么模型 / 产物不入库？**
A：`models/`、`outputs/`、`venv/` 含数 GB ~ 数十 GB 权重与虚拟环境，不适合 Git。
仓库只保留可复现的源码、配置、示例数据与文档，运行脚本后从 ModelScope 重新下载。

**Q：必须离线加载吗？**
A：本机环境直连 HuggingFace（含 hf-mirror）不通，故脚本强制 `TRANSFORMERS_OFFLINE=1`
并走本地路径。联网良好的环境可改为在线加载。

**Q：中文字符乱码？**
A：Windows 下确保以 UTF-8 读写 `.jsonl`；本项目脚本均 `encoding="utf-8"`。仓库文档
使用 VitePress，源码 UTF-8。

## 方法论风险

1. **跨家族权重融合慎行**：Qwen + Gemma 直接 merge 大概率"歪"，优先走蒸馏路径。
2. **许可证**：Gemma4 为 Apache 2.0（可商用）；Qwen 系一贯 Apache 2.0。MaiLLM 为派生作品，
   部署前核对上游条款。
3. **蒸馏 ≠ 造知识**：新事实靠 RAG，不靠训练。
4. **评估不能省**：任何 merge / 量化后必须做 held-out 评测，融合会"静默退化"未测能力。
5. **数据污染**：测试集绝不能进训练集。
