---
layout: home

hero:
  name: MaiLLM
  text: 端侧多模态小模型训练与蒸馏
  tagline: 基于 Qwen3.5 + Gemma4 双教师响应蒸馏 + QLoRA，在 8GB 显卡上训练你的端侧模型
  actions:
    - theme: brand
      text: 快速开始
      link: /guide/quickstart
    - theme: alt
      text: 项目定位
      link: /guide/introduction

features:
  - title: 双教师响应蒸馏
    details: Qwen3.5-4B（同家族，中文/推理）与 Gemma4-E4B（跨家族，英文/多语言）共同生成蒸馏数据，把两家优点教进 0.8B 学生。
  - title: QLoRA 低显存训练
    details: 4-bit 量化底座 + LoRA(r=32)，在 RTX 4060 8GB 显存上即可完成训练，无需昂贵硬件。
  - title: 跨架构不靠融合
    details: Qwen 与 Gemma 架构/词表不同，直接权重融合会互相干扰；本项目用蒸馏而非 MergeKit 解决跨家族结合。
  - title: 端到端可复现
    details: 从下载模型、造数据、训练、合并导出到基准评测，全部脚本化，离线加载，一键复现。
  - title: 评测驱动
    details: 内置 PPL + ROUGE-L/BLEU 对比评测，验证蒸馏有效且未破坏底座基础语言能力。
  - title: 原生端侧部署
    details: 训练产物为标准 HF 模型，llama.cpp 转 GGUF 后 q4_k_m 量化，目标 2–4GB、≥30–50 tok/s 移动端。
---

## 这是什么

**MaiLLM** 是一套在消费级显卡上把最新的端侧多模态大模型"蒸馏"成小模型的开源训练流水线。
它不追求 SOTA 数字，而是给出一个**可复现、低门槛、讲清楚原理**的端侧小模型训练范式。

> 想直接跑起来？看 [快速开始](/guide/quickstart)。想先理解为什么这么做？看 [项目定位](/guide/introduction) 与 [训练方案研究](/guide/research)。
