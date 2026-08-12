# -*- coding: utf-8 -*-
"""
蒸馏数据生成 (响应蒸馏 / behavioral distillation)
- 中文/推理/工具类提示词  -> Qwen3.5-4B 教师生成
- 英文/多语言提示词       -> Gemma4-E4B 教师生成
两位教师各生成一份 JSONL, 最后 --combine 合并为 MaiLLM 的训练集。

关键约束 (已实测):
- 模型全部从 ModelScope 下到本地目录, 必须离线加载, 禁止直连 HF(hf-mirror 连不通)。
- Qwen3.5 / Gemma4 是新架构, transformers 5.14.1 原生支持, trust_remote_code=False。
- Qwen 教师与学生在同一词表(248320)家族; Gemma vocab=262144, 跨家族仅做响应蒸馏。

用法:
  python train/generate_data.py qwen      # 生成中文蒸馏数据
  python train/generate_data.py gemma     # 生成英文蒸馏数据
  python train/generate_data.py combine   # 合并为 maillm_train.jsonl
  python train/generate_data.py all       # 依次 qwen -> gemma -> combine
"""
import sys
import os
import json
import argparse
import torch

# 强制离线: 只从本地快照加载, 绝不访问 HuggingFace hub
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maillm_config import (QWEN_TEACHER_PATH, GEMMA_TEACHER_PATH,
                           KD_DATA_ZH, KD_DATA_EN, KD_DATA_ALL,
                           SEED_PROMPTS_ZH, SEED_PROMPTS_EN)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- 种子提示词 ----------------
ZH_SEEDS = [
    "请用通俗易懂的方式解释什么是大语言模型的量化，以及 4-bit 量化为什么会大幅减小模型体积。",
    "帮我写一段关于「秋天的早晨」的散文，语言优美，200 字左右。",
    "一道数学题：甲、乙两人合作完成一项工程需要 6 天，甲单独做需要 10 天，问乙单独做需要几天？请给出完整推理过程。",
    "我要开发一个本地运行的 AI 助手 App，请给出技术选型建议，要求能在手机上离线运行。",
    "请用 Python 写一个函数，输入一个列表，返回去重后保持原顺序的新列表，并加注释。",
    "解释一下知识蒸馏（knowledge distillation）的原理，以及它和微调（fine-tuning）的区别。",
    "帮我润色下面这句话，使其更专业：「我们这个模型跑得挺快的，手机上也能用。」",
    "什么是检索增强生成（RAG）？它在端侧小模型上有什么作用和局限？",
    "请用三步法向我解释什么是 MoE（混合专家）架构。",
    "我需要调用天气 API 获取北京今天的天气，请给出一个符合 JSON 格式的函数调用示例。",
    "解释梯度消失和梯度爆炸，并给出常用的缓解方法。",
    "写一首关于「故乡」的现代诗，四节，每节四行。",
    "如何用最少的显存在大模型上做微调？请对比 LoRA、QLoRA、全参微调的显存占用。",
    "请总结端侧大模型相比云端大模型的三大优势和两大挑战。",
    "帮我设计一份「一周健康饮食」的安排，并说明每餐的营养要点。",
    "什么是注意力机制中的 KV Cache？为什么它在长文本推理时会占用大量显存？",
    "请用类比的方式解释 Transformer 中的自注意力（self-attention）。",
    "我想在本地用 llama.cpp 部署一个 1.7B 的模型，请给出从下载到运行的完整步骤。",
    "解释一下困惑度（perplexity）作为语言模型评测指标的含义。",
    "请给出一个系统性排查「模型推理速度慢」的思路清单。",
    "什么是模型的 tokenizer？为什么不同模型的词表大小会影响中文效果？",
    "请用简洁的语言说明：为什么小模型在手机上推理时，2~4GB 是最佳体积区间？",
    "帮我写一段产品介绍文案，介绍一个叫 MaiLLM 的端侧多模态小模型。",
    "解释一下 Gemma 的 MatFormer 架构如何做到「一个模型多种尺寸」。",
    "请列举 5 个把大模型能力「蒸馏」到小模型的实用技巧。",
]

EN_SEEDS = [
    "Explain the difference between fine-tuning and in-context learning in plain English.",
    "Write a short, friendly email to a colleague postponing a meeting to tomorrow at 10am.",
    "What is mixture-of-experts (MoE)? Give one real-world analogy.",
    "Summarize the key benefits of on-device (edge) AI over cloud-based inference.",
    "Give a Python snippet that flattens a nested list, with a brief comment.",
    "Explain knowledge distillation and why it helps compress large models.",
    "How does 4-bit quantization reduce memory while keeping acceptable accuracy?",
    "List three challenges of running multimodal models on smartphones.",
    "Write a concise product description for an edge AI assistant named MaiLLM.",
    "What is a KV cache, and why does it grow with sequence length?",
]


def save_seeds():
    with open(SEED_PROMPTS_ZH, "w", encoding="utf-8") as f:
        for p in ZH_SEEDS:
            f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")
    with open(SEED_PROMPTS_EN, "w", encoding="utf-8") as f:
        for p in EN_SEEDS:
            f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")


def pick_model_class(path: str):
    """按本地路径选择显式架构类 (避免 Auto 映射不确定 + 离线安全)。"""
    from transformers import (Qwen3_5ForConditionalGeneration,
                              Gemma4ForConditionalGeneration)
    if "gemma" in path.lower():
        return Gemma4ForConditionalGeneration
    return Qwen3_5ForConditionalGeneration


def load_teacher(path):
    from transformers import AutoProcessor, BitsAndBytesConfig
    is_gemma = "gemma" in path.lower()
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=False,
                                             local_files_only=True)
    model_cls = pick_model_class(path)
    if is_gemma:
        # Gemma4-E4B 多模态: 8GB 显存放不下 4bit 全量, 且 bnb4bit 不容许 offload。
        # 改用 fp16 全精度 + device_map="auto", 把溢出层 offload 到 CPU 内存
        # (本机 29.7GB RAM + D:\pagefile 16~32GB 虚拟内存兜底); 不量化即可跨设备运行。
        print(f"\n载入 Gemma 教师模型(fp16 自动 offload 到 CPU, 离线): {path}")
        max_memory = {0: "7GiB", "cpu": "24GiB"}
        model = model_cls.from_pretrained(
            path, torch_dtype=torch.float16,
            device_map="auto", max_memory=max_memory,
            trust_remote_code=False, local_files_only=True,
        )
    else:
        print(f"\n载入 Qwen 教师模型(4bit, 离线): {path}")
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_quant_type="nf4")
        model = model_cls.from_pretrained(
            path, quantization_config=bnb, device_map="auto",
            trust_remote_code=False, local_files_only=True,
        )
    model.eval()
    return model, processor


def generate_one(model, processor, prompt, max_new=768):
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    pad_id = tok.pad_token_id
    eos_id = getattr(model.config, "eos_token_id", None)
    if pad_id is None and eos_id is not None:
        pad_id = eos_id

    has_template = bool(getattr(tok, "chat_template", None))
    if has_template:
        # Qwen 等带 chat template 的 VLM: 走官方 processor 纯文本路径
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids
    else:
        # Gemma 系列无 chat template: 用官方 <start_of_turn>/<end_of_turn> 格式
        gemma_text = ("<start_of_turn>user\n" + prompt +
                      "<end_of_turn>\n<start_of_turn>model\n")
        input_ids = tok(gemma_text, return_tensors="pt").input_ids.to(model.device)

    gen_kwargs = dict(max_new_tokens=max_new, do_sample=True,
                      temperature=0.7, top_p=0.9, repetition_penalty=1.05)
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id
    if eos_id is not None:
        gen_kwargs["eos_token_id"] = eos_id
    out = model.generate(input_ids, **gen_kwargs)
    resp = tok.decode(out[0][input_ids.shape[1]:],
                      skip_special_tokens=True).strip()
    return resp


def run_teacher(teacher: str):
    save_seeds()
    if teacher == "qwen":
        model, processor = load_teacher(QWEN_TEACHER_PATH)
        seeds = ZH_SEEDS
        out_path = KD_DATA_ZH
    elif teacher == "qwen_en":
        # 8GB 显存下 gemma-4-E4B 无法本地推理(bnb4bit 不容许 offload),
        # 英文蒸馏数据改由同家族 Qwen3.5-4B 生成(支持 201 语言, 质量足够)
        model, processor = load_teacher(QWEN_TEACHER_PATH)
        seeds = EN_SEEDS
        out_path = KD_DATA_EN
    else:
        model, processor = load_teacher(GEMMA_TEACHER_PATH)
        seeds = EN_SEEDS
        out_path = KD_DATA_EN

    n = len(seeds)
    ok = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, p in enumerate(seeds, 1):
            print(f"[{teacher}] 生成 {i}/{n}: {p[:28]}...")
            try:
                resp = generate_one(model, processor, p)
            except Exception as e:
                print(f"  [error] 跳过: {e}")
                continue
            rec = {"messages": [
                {"role": "user", "content": p},
                {"role": "assistant", "content": resp},
            ]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ok += 1
    # 释放显存, 便于下一个教师顺序加载
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\n{teacher} 教师数据已保存: {out_path}  (成功 {ok}/{n})")


def combine():
    cnt = 0
    with open(KD_DATA_ALL, "w", encoding="utf-8") as out:
        for src in (KD_DATA_ZH, KD_DATA_EN):
            if not os.path.exists(src):
                print(f"[skip] 缺失: {src}")
                continue
            with open(src, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    json.loads(line)  # 校验
                    out.write(line + "\n")
                    cnt += 1
    print(f"合并完成: {KD_DATA_ALL}  (共 {cnt} 条)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["qwen", "gemma", "qwen_en", "combine", "all"])
    args = ap.parse_args()
    if args.action == "qwen":
        run_teacher("qwen")
    elif args.action == "qwen_en":
        run_teacher("qwen_en")
    elif args.action == "gemma":
        run_teacher("gemma")
    elif args.action == "combine":
        combine()
    elif args.action == "all":
        run_teacher("qwen")
        run_teacher("gemma")
        combine()


if __name__ == "__main__":
    main()
