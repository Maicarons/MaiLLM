# -*- coding: utf-8 -*-
"""
MaiLLM 训练脚本 (QLoRA + 文本 SFT, 针对 RTX 4060 8GB)
- 底座: Qwen/Qwen3.5-0.8B-Base (多模态 VLM, 仅训练语言部分, 保留多模态能力)
- 方法: 4bit 量化底座 + LoRA(r=32) + 响应蒸馏数据(来自 Qwen3.5-4B 与 Gemma4-E4B 教师)
- 数据: data/maillm_train.jsonl (由 generate_data.py combine 生成)

关键约束 (已实测):
- 模型从本地目录离线加载, 禁止直连 HF。
- Qwen3.5 是新架构, transformers 5.14.1 原生支持, trust_remote_code=False。
- 4bit+LoRA 训练必须先 prepare_model_for_kbit_training。

用法:
  python train/train_maillm.py            # 全量训练
  python train/train_maillm.py --smoke    # 仅冒烟测试(前向+反向 1 步)后退出
"""
import sys
import os
import json
import argparse
import torch
from torch.utils.data import Dataset

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maillm_config import (
    STUDENT_PATH, KD_DATA_ALL, LORA_OUTPUT,
    MAX_SEQ_LEN, LORA_R, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES,
    LEARNING_RATE, NUM_EPOCHS, PER_DEVICE_BATCH, GRAD_ACCUM, WARMUP_STEPS,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SFTDataset(Dataset):
    """对话 JSONL -> input_ids/labels, 屏蔽 prompt 部分。兼容 base 模型缺 chat template。"""
    def __init__(self, path, tokenizer, max_len):
        self.examples = []
        has_tmpl = hasattr(tokenizer, "apply_chat_template")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                msgs = rec["messages"]
                try:
                    if has_tmpl:
                        full_text = tokenizer.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=False)
                        prompt_msgs = msgs[:-1] + [{"role": "assistant", "content": ""}]
                        prompt_text = tokenizer.apply_chat_template(
                            prompt_msgs, tokenize=False, add_generation_prompt=True)
                    else:
                        full_text, prompt_text = manual_template(msgs)
                except Exception:
                    full_text, prompt_text = manual_template(msgs)

                full_ids = tokenizer(full_text, return_tensors="pt",
                                    truncation=True, max_length=max_len)["input_ids"][0]
                prompt_ids = tokenizer(prompt_text, return_tensors="pt",
                                       truncation=True, max_length=max_len)["input_ids"][0]
                if len(full_ids) < 4:
                    continue
                labels = full_ids.clone()
                mask_len = min(len(prompt_ids), len(full_ids))
                labels[:mask_len] = -100
                if (labels == -100).all():
                    continue
                self.examples.append({
                    "input_ids": full_ids,
                    "attention_mask": torch.ones_like(full_ids),
                    "labels": labels,
                })
        print(f"数据集加载完成: {len(self.examples)} 条 (来自 {path})")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def manual_template(msgs):
    """base 模型缺 chat template 时的兜底格式 (Qwen 风格)。"""
    parts = []
    for m in msgs:
        role = m["role"]
        content = m["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    full_text = "".join(parts)
    # prompt 部分 = 除最后一条 assistant 回复外的全部
    prompt_text = "".join(parts[:-1]) + "<|im_start|>assistant\n"
    return full_text, prompt_text


def collate(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attn, labels = [], [], []
    for x in batch:
        L = len(x["input_ids"])
        input_ids.append(torch.cat([x["input_ids"],
                         torch.full((max_len - L,), pad_id, dtype=torch.long)]))
        attn.append(torch.cat([x["attention_mask"],
                      torch.zeros(max_len - L, dtype=torch.long)]))
        labels.append(torch.cat([x["labels"],
                      torch.full((max_len - L,), -100, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attn),
        "labels": torch.stack(labels),
    }


def load_student():
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen3_5ForConditionalGeneration)
    print(f"载入学生底座(4bit, 离线): {STUDENT_PATH}")
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4")
    processor = AutoProcessor.from_pretrained(STUDENT_PATH,
                                             trust_remote_code=False,
                                             local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        STUDENT_PATH, quantization_config=bnb, device_map="auto",
        trust_remote_code=False, local_files_only=True)
    model.config.use_cache = False
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    return model, processor, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="仅冒烟测试后退出")
    args = ap.parse_args()

    from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training)
    from transformers import Trainer, TrainingArguments

    model, processor, tok = load_student()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"pad_token_id={pad_id}, eos_token_id={tok.eos_token_id}")

    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = SFTDataset(KD_DATA_ALL, tok, MAX_SEQ_LEN)

    # ---------- 冒烟测试: 文本前向 + 反向 1 步 ----------
    print("\n===== 冒烟测试 (前向+反向) =====")
    small = torch.utils.data.Subset(ds, range(min(2, len(ds))))
    loader = torch.utils.data.DataLoader(
        small, batch_size=2, collate_fn=lambda b: collate(b, pad_id))
    batch = next(iter(loader))
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    model.train()
    out = model(**batch)
    loss = out.loss
    print("smoke loss =", float(loss))
    loss.backward()
    print("smoke backward OK")
    if args.smoke:
        print("冒烟测试通过, 退出 (--smoke)。")
        return

    # ---------- 全量训练 ----------
    training_args = TrainingArguments(
        output_dir=LORA_OUTPUT,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        bf16=True,
        logging_steps=2,
        save_strategy="epoch",
        optim="adamw_torch",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=ds,
        data_collator=lambda b: collate(b, pad_id))
    print("开始训练 MaiLLM ...")
    trainer.train()
    trainer.save_model(LORA_OUTPUT)
    processor.save_pretrained(LORA_OUTPUT)
    print(f"LoRA 权重已保存到: {LORA_OUTPUT}")


if __name__ == "__main__":
    main()
