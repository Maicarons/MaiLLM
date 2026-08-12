# 端侧部署

训练产物 `outputs/maillm_merged/` 是标准 HuggingFace 模型（bf16）。端侧部署分两步：
**量化压缩** 与 **目标设备推理**。

## 1. 量化压缩（llama.cpp → GGUF）

```bash
# 安装 llama.cpp 并转换
# 1) 将 HF 模型转 GGUF（fp16 中间格式）
python llama.cpp/convert_hf_to_gguf.py outputs/maillm_merged --outfile maillm-f16.gguf

# 2) 量化为 q4_k_m（端侧默认，约 0.5–1GB）
llama-quantize maillm-f16.gguf maillm-q4_k_m.gguf q4_k_m
```

> 本项目训练阶段已用 4-bit 量化底座，但**导出的是 bf16 标准模型**；
> 部署前再用量化工具压到 q4_k_m，是端侧体积 / 精度的甜点。

## 2. 目标设备推理栈

| 平台 | 推荐推理栈 |
|------|-----------|
| iOS / iPadOS | MLX / Core ML / ExecuTorch |
| Android | MediaPipe / ExecuTorch / TFLite / Google AI Edge（Gemma4 E2B/E4B 已进 AICore 预览） |
| macOS | MLX / llama.cpp |
| Windows / Linux 桌面 | llama.cpp / ONNX Runtime / TensorRT-LLM |
| 苹果端 | MLX（近原生性能） |

## 3. 部署目标参考

- **甜点体积**：q4 下 1–2.5GB（≈ 1.7B 级以内），目标 ≥ 30–50 tok/s 移动端。
- **离线优先**：MaiLLM 定位就是端侧离线运行，不依赖云端 API。
- **多模态**：底座为原生 VLM，图像输入走 processor 的视觉分支；纯文本路径已验证，
  视觉分支端侧推理需目标设备 NPU 支持。

## 4. 验证清单

部署后建议回归验证：

- [ ] 内存占用在目标设备预算内（≤ 甜点区间）。
- [ ] tok/s 达到目标（移动端 ≥ 30–50）。
- [ ] 中文 / 英文生成质量对比训练前无显著退化（参考 [基准评测](/guide/benchmark)）。
- [ ] 测试集未泄漏进训练集（数据污染检查）。
