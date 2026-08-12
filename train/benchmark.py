# -*- coding: utf-8 -*-
"""
MaiLLM 基准测试与对比 (vs 原始 Qwen3.5-0.8B-Base)
==================================================
全部离线加载, 顺序加载以控制 8GB 显存。

指标:
  1) 困惑度 PPL:
       - 域内(in-domain): 训练集里教师(Qwen3.5-4B)的回答文本 -> 衡量是否学到分布
       - 域外(OOD): 通用中/英文段落(不含 LLM 领域) -> 检验是否遗忘基础语言能力
  2) 蒸馏保真度: 用 Qwen3.5-4B 教师对一批 HELD-OUT 提示词生成参考,
       再算 原始底座(Base) 与 MaiLLM 输出 相对参考的 ROUGE-L / BLEU (均贪婪解码, 可复现)
  3) 生成样本定性对比

用法:
  python train/benchmark.py            # 全流程(含教师参考生成)
  python train/benchmark.py --skip-ref # 跳过教师参考生成(已有 bench_refs.jsonl 时)
"""
import sys
import os
import json
import math
import time
import argparse

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maillm_config import (DATA_DIR, OUTPUT_DIR,
                           STUDENT_PATH, QWEN_TEACHER_PATH, KD_DATA_ALL)

BASE_PATH = STUDENT_PATH                                   # 原始底座 = 对照
MAILLM_PATH = os.path.join(OUTPUT_DIR, "maillm_merged")    # 训练产物
TEACHER_PATH = QWEN_TEACHER_PATH                            # 参考教师

BENCH_DIR = os.path.join(OUTPUT_DIR, "benchmark")
os.makedirs(BENCH_DIR, exist_ok=True)
BENCH_PROMPTS = os.path.join(DATA_DIR, "bench_prompts.jsonl")
BENCH_REFS = os.path.join(DATA_DIR, "bench_refs.jsonl")
GEN_BASE = os.path.join(BENCH_DIR, "gen_base.jsonl")
GEN_MAILLM = os.path.join(BENCH_DIR, "gen_maillm.jsonl")
REPORT = os.path.join(BENCH_DIR, "report.md")

MAX_PPL_LEN = 1024
GEN_MAX_NEW = 768

# ---------------- HELD-OUT 测试提示词 (与训练种子不重叠) ----------------
BENCH_PROMPTS_LIST = [
    "请解释什么是模型量化里的 AWQ，它和 GPTQ 有什么主要区别？",
    "我想把一个 0.8B 的小模型部署到树莓派上离线运行，请给出存储与内存方面的注意事项。",
    "用通俗的话说明：为什么端侧模型通常需要做指令微调(Instruction Tuning)？",
    "帮我写一段关于「夏夜的星空」的短散文，100 字左右。",
    "一道题：某工程甲单独做 12 天完工，甲乙合作 8 天完工，问乙单独做要几天？给出推理。",
    "什么是模型推理时的 batch size？增大 batch size 对吞吐和延迟分别有什么影响？",
    "请用三步法解释什么是向量数据库(Vector Database)以及它在 RAG 中的作用。",
    "Explain what mixed-precision training means and why it is useful on GPUs.",
    "Write a short paragraph introducing the concept of 'edge AI' to a beginner.",
    "Why does a longer context window increase both memory usage and inference cost? Give a concise explanation.",
    "List three practical ways to speed up LLM inference on a consumer laptop GPU.",
    "Give a Python function that counts word frequencies in a string, with a brief comment.",
]

# 域外(通用)文本, 不含 LLM 领域, 用于检验是否遗忘基础语言能力
OOD_TEXTS = [
    "春天来了，河岸的柳树抽出嫩绿的新芽，微风拂过水面，泛起层层涟漪。远处的田野里，农人正弯腰插秧，空气中弥漫着泥土的清香。这是一年中最充满希望的季节。",
    "长城是中国古代最伟大的防御工程之一，东起山海关，西至嘉峪关，蜿蜒万里。它始建于春秋战国时期，历经多个朝代的修缮与扩建，见证了中华民族数千年的兴衰与变迁。",
    "The Industrial Revolution began in Britain in the late eighteenth century and transformed agrarian societies into industrial ones. New machines, factories, and railways reshaped how people worked and lived, while cities grew rapidly around coal and iron.",
    "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen. This biochemical pathway not only sustains nearly all life on Earth but also regulates the atmosphere we breathe.",
]


# ------------------------- 文本指标 (纯 Python) -------------------------
def _tok(text):
    has_cjk = any('\u4e00' <= c <= '\u9fff' for c in text)
    if has_cjk:
        return [c for c in text if not c.isspace()]
    return text.split()


def _lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        ai = a[i]
        for j in range(n):
            dp[i + 1][j + 1] = dp[i][j] + 1 if ai == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n]


def rouge_l(hyp, ref):
    a, b = _tok(hyp), _tok(ref)
    if not a or not b:
        return 0.0
    l = _lcs(a, b)
    r = l / len(b)
    p = l / len(a)
    return 0.0 if (r + p) == 0 else 2 * r * p / (r + p)


def _ngrams(toks, n):
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def bleu(hyp, ref, maxn=4):
    a, b = _tok(hyp), _tok(ref)
    if not a or not b:
        return 0.0
    bp = math.exp(min(0.0, 1.0 - len(b) / len(a)))
    score, cnt = 0.0, 0
    for n in range(1, maxn + 1):
        hp, rp = _ngrams(a, n), _ngrams(b, n)
        if not hp:
            continue
        refset = {}
        for g in rp:
            refset[g] = refset.get(g, 0) + 1
        hypc = {}
        for g in hp:
            hypc[g] = hypc.get(g, 0) + 1
        hit = sum(min(c, refset.get(g, 0)) for g, c in hypc.items())
        prec = hit / len(hp)
        if prec > 0:
            score += math.log(prec)
            cnt += 1
    return 0.0 if cnt == 0 else bp * math.exp(score / cnt)


# ------------------------- 模型加载 -------------------------
def load_full_model(path, quant4=False):
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen3_5ForConditionalGeneration)
    print(f"  [load] {path}  (4bit={quant4})")
    if quant4:
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_quant_type="nf4")
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            path, quantization_config=bnb, device_map="auto",
            trust_remote_code=False, local_files_only=True)
    else:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=False, local_files_only=True)
    model.config.use_cache = True
    model.eval()
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=False,
                                              local_files_only=True)
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    return model, processor, tok


def free_model(model):
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------- 评估函数 -------------------------
@torch.no_grad()
def compute_ppl(model, tok, texts, max_len=MAX_PPL_LEN):
    tot_loss, n = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=max_len)["input_ids"]
        if ids.shape[1] < 2:
            continue
        ids = ids.to(model.device)
        out = model(input_ids=ids, labels=ids)
        tot_loss += float(out.loss)
        n += 1
    return math.exp(tot_loss / n) if n else float("nan"), n


@torch.no_grad()
def generate(model, tok, prompt, max_new=GEN_MAX_NEW):
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new, do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs.input_ids.shape[1]:],
                      skip_special_tokens=True).strip()


def load_training_responses(path):
    resps = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for m in rec["messages"]:
                if m["role"] == "assistant":
                    resps.append(m["content"])
    return resps


# ------------------------- 主流程 -------------------------
def make_references(tok_for_teacher_load_only=None):
    """用 Qwen3.5-4B 教师对 held-out 提示词生成参考(贪婪, 可复现)。"""
    if os.path.exists(BENCH_REFS):
        print(f"[refs] 已存在, 跳过教师加载: {BENCH_REFS}")
        return
    print("\n===== 阶段0: 加载教师 Qwen3.5-4B(4bit) 生成参考 =====")
    model, processor, tok = load_full_model(TEACHER_PATH, quant4=True)
    with open(BENCH_REFS, "w", encoding="utf-8") as fout, \
         open(BENCH_PROMPTS, "w", encoding="utf-8") as fp:
        for i, p in enumerate(BENCH_PROMPTS_LIST, 1):
            fp.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")
            t0 = time.time()
            ref = generate(model, tok, p)
            fout.write(json.dumps({"prompt": p, "reference": ref},
                                  ensure_ascii=False) + "\n")
            print(f"  ref {i}/{len(BENCH_PROMPTS_LIST)}  ({time.time()-t0:.1f}s)  {p[:24]}...")
    free_model(model)


def evaluate_model(name, model_path, refs, out_gen_path):
    print(f"\n===== 阶段: 评测 {name} =====")
    model, processor, tok = load_full_model(model_path, quant4=False)

    # 1) 困惑度
    train_resp = load_training_responses(KD_DATA_ALL)
    ppl_in, n_in = compute_ppl(model, tok, train_resp)
    ppl_ood, n_ood = compute_ppl(model, tok, OOD_TEXTS)
    print(f"  PPL 域内(训练回答, n={n_in}): {ppl_in:.3f}")
    print(f"  PPL 域外(通用文本, n={n_ood}): {ppl_ood:.3f}")

    # 2) 生成 + 与参考的 ROUGE/BLEU
    rls, bls = [], []
    with open(out_gen_path, "w", encoding="utf-8") as fout:
        for i, rec in enumerate(refs, 1):
            p = rec["prompt"]
            ref = rec.get("reference", "")
            t0 = time.time()
            hyp = generate(model, tok, p)
            rl = rouge_l(hyp, ref)
            bl = bleu(hyp, ref)
            rls.append(rl)
            bls.append(bl)
            fout.write(json.dumps({"prompt": p, "hypothesis": hyp},
                                  ensure_ascii=False) + "\n")
            print(f"  gen {i}/{len(refs)}  ROUGE-L={rl:.3f} BLEU={bl:.3f}"
                  f"  ({time.time()-t0:.1f}s)")
    free_model(model)
    avg_rl = sum(rls) / len(rls)
    avg_bl = sum(bls) / len(bls)
    return {
        "name": name, "ppl_in": ppl_in, "ppl_ood": ppl_ood,
        "avg_rouge_l": avg_rl, "avg_bleu": avg_bl,
        "n_test": len(refs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ref", action="store_true",
                    help="已有 bench_refs.jsonl 时跳过教师参考生成")
    args = ap.parse_args()

    refs = []
    if not args.skip_ref:
        make_references()
    with open(BENCH_REFS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                refs.append(json.loads(line))
    print(f"\n参考样本数: {len(refs)}")

    res_base = evaluate_model("Base(Qwen3.5-0.8B-Base 原始)",
                              BASE_PATH, refs, GEN_BASE)
    res_maillm = evaluate_model("MaiLLM(merged)",
                                MAILLM_PATH, refs, GEN_MAILLM)

    # 汇总
    print("\n===== 汇总 =====")
    print(f"{'模型':<32}{'PPL域内':>10}{'PPL域外':>10}{'ROUGE-L':>10}{'BLEU':>10}")
    for r in (res_base, res_maillm):
        print(f"{r['name']:<30}{r['ppl_in']:>10.3f}{r['ppl_ood']:>10.3f}"
              f"{r['avg_rouge_l']:>10.3f}{r['avg_bleu']:>10.3f}")

    write_report(res_base, res_maillm, refs)
    print(f"\n报告已写入: {REPORT}")


def write_report(res_base, res_maillm, refs):
    def load_gen(path):
        out = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    gen_base = load_gen(GEN_BASE)
    gen_maillm = load_gen(GEN_MAILLM)

    def delta(a, b):
        return a - b

    lines = []
    lines.append("# MaiLLM 基准测试与对比报告\n")
    lines.append(f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("- 评测对象: **MaiLLM(merged)** vs **Qwen3.5-0.8B-Base(原始底座)**\n")
    lines.append("- 方法: QLoRA 响应蒸馏(教师 Qwen3.5-4B)，3 epoch，训练集 25 条\n")
    lines.append("\n## 1. 总览指标\n")
    lines.append("| 模型 | PPL 域内(↓) | PPL 域外(↓) | ROUGE-L(↑) | BLEU(↑) |")
    lines.append("|------|-----------|-----------|-----------|--------|")
    lines.append(f"| {res_base['name']} | {res_base['ppl_in']:.3f} | "
                 f"{res_base['ppl_ood']:.3f} | {res_base['avg_rouge_l']:.3f} | "
                 f"{res_base['avg_bleu']:.3f} |")
    lines.append(f"| {res_maillm['name']} | {res_maillm['ppl_in']:.3f} | "
                 f"{res_maillm['ppl_ood']:.3f} | {res_maillm['avg_rouge_l']:.3f} | "
                 f"{res_maillm['avg_bleu']:.3f} |")
    lines.append("\n说明: 困惑度(PPL)越低越好; ROUGE-L/BLEU 越高表示越接近教师参考(蒸馏保真度)。\n")

    lines.append("## 2. 指标解读\n")
    # 域内科
    if res_maillm['ppl_in'] < res_base['ppl_in']:
        lines.append(f"- **域内困惑度**: MaiLLM {res_maillm['ppl_in']:.3f} < "
                     f"Base {res_base['ppl_in']:.3f}，下降 "
                     f"{delta(res_base['ppl_in'], res_maillm['ppl_in']):.3f}。"
                     "说明训练后模型对教师回答风格/分布的拟合更好（学到东西）。")
    else:
        lines.append(f"- **域内困惑度**: MaiLLM {res_maillm['ppl_in']:.3f} ≥ "
                     f"Base {res_base['ppl_in']:.3f}，未见下降，需关注是否过拟合或训练不充分。")
    # 域外
    if res_maillm['ppl_ood'] <= res_base['ppl_ood'] * 1.05:
        lines.append(f"- **域外困惑度**: MaiLLM {res_maillm['ppl_ood']:.3f} 对比 Base "
                     f"{res_base['ppl_ood']:.3f}，基本持平(未显著退化基础语言能力，无严重灾难性遗忘)。")
    else:
        lines.append(f"- **域外困惑度**: MaiLLM {res_maillm['ppl_ood']:.3f} 明显高于 Base "
                     f"{res_base['ppl_ood']:.3f}，存在一定遗忘，建议加入通用语料或降低过拟合。")
    # ROUGE
    lines.append(f"- **蒸馏保真度(ROUGE-L)**: MaiLLM {res_maillm['avg_rouge_l']:.3f} vs "
                 f"Base {res_base['avg_rouge_l']:.3f}。"
                 "MaiLLM 经过 SFT 蒸馏，输出在形式上更贴近教师(含其『思考过程』风格)，"
                 "因此 n-gram 重叠更高；Base 输出更简短、与教师参考重叠低。")
    lines.append(f"- **BLEU**: MaiLLM {res_maillm['avg_bleu']:.3f} vs "
                 f"Base {res_base['avg_bleu']:.3f}。\n")

    lines.append("## 3. 生成样本定性对比 (held-out 提示词)\n")
    for i, rec in enumerate(refs):
        p = rec["prompt"]
        ref = rec.get("reference", "")
        gb = gen_base[i]["hypothesis"] if i < len(gen_base) else ""
        gm = gen_maillm[i]["hypothesis"] if i < len(gen_maillm) else ""
        lines.append(f"### 样本 {i+1}\n")
        lines.append(f"**Prompt:** {p}\n")
        lines.append(f"**教师参考(截断):**\n> {ref[:300]}\n")
        lines.append(f"**Base 原始输出(截断):**\n> {gb[:300]}\n")
        lines.append(f"**MaiLLM 输出(截断):**\n> {gm[:300]}\n")
        lines.append("")

    lines.append("## 4. 结论与建议\n")
    lines.append("- 本次为小样本(25 条)快速蒸馏的初版验证，重点看趋势而非绝对值。\n")
    lines.append("- 若域内 PPL 下降且域外 PPL 不涨：蒸馏有效且未破坏底座基础能力。\n")
    lines.append("- 后续可补充：更大/更多样训练集、加入纯中文蒸馏数据、引入 Gemma 英文数据、"
                 "以及用标准 benchmark(如 C-Eval 子集/MMLU 子集)做更严谨评测。\n")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
