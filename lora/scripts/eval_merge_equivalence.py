#!/usr/bin/env python
"""Evaluate adapter form vs merged form equivalence for tied-shared affine adapters.

Two modes:
  --quick : single-text logit diff check (fast sanity, no eval data needed)
  default : full eval-set PPL + greedy generation + ROUGE-L comparison
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from affine_vocab_lora import load_affine_vocab_adapter

PROMPT_TEMPLATE = """Question:
{question}

Answer:
"""

DTYPE_MAP = {
    "auto": "auto",
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-dtype", choices=list(DTYPE_MAP), default="bf16")

    # Quick mode
    parser.add_argument(
        "--quick", action="store_true",
        help="Single-text logit diff check (fast, no eval data).",
    )
    parser.add_argument("--text", default="Question:\n请简要解释低秩适配器的作用。\n\nAnswer:\n")
    parser.add_argument("--max-new-tokens", type=int, default=0)

    # Full mode (default)
    parser.add_argument("--eval-data", default=None)
    parser.add_argument("--ppl-samples", type=int, default=500)
    parser.add_argument("--gen-samples", type=int, default=100)
    parser.add_argument("--gen-max-new-tokens", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


# ── shared helpers ─────────────────────────────────────────────────────

def diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    d = (a.float() - b.float()).abs()
    return {
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "rms": float(torch.sqrt(torch.mean(d * d)).item()),
    }


def merged_embedding_weight(model: Any) -> torch.Tensor:
    emb = model.get_input_embeddings()
    if not hasattr(emb, "base_embedding") or not hasattr(emb, "affine"):
        raise TypeError("Expected input embeddings to be wrapped by AffineEmbedding.")
    with torch.no_grad():
        return emb.affine(emb.base_embedding.weight.detach())


def apply_merged_tied_weight(model: Any, merged_w: torch.Tensor) -> None:
    inp = model.get_input_embeddings()
    head = model.lm_head
    with torch.no_grad():
        inp.weight.copy_(merged_w.to(device=inp.weight.device, dtype=inp.weight.dtype))
        if head.weight.data_ptr() != inp.weight.data_ptr():
            head.weight.copy_(merged_w.to(device=head.weight.device, dtype=head.weight.dtype))


def load_base(model_path: str, dtype_arg: str, device: torch.device) -> Any:
    base_dtype = DTYPE_MAP[dtype_arg]
    load_kw = {} if base_dtype == "auto" else {"torch_dtype": base_dtype}
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kw).to(device)
    model.eval()
    return model


def load_adapter_model(model_path: str, adapter_dir: str, dtype_arg: str, device: torch.device) -> Any:
    model = load_base(model_path, dtype_arg, device)
    load_affine_vocab_adapter(model, adapter_dir)
    cfg = getattr(model, "affine_vocab_config", None)
    if cfg is None or not getattr(cfg, "tie_input_lm_head_adapters", False):
        raise ValueError("Adapter is not tied/mergeable.")
    return model


# ── quick mode ─────────────────────────────────────────────────────────

def transposed_tied_lm_head_logits(adapter_model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    """Compute logits for the mergeable tied form W' = affine(W).

    If input embeddings are merged as W' = W M + b, the matching output-side
    computation is h M^T W^T plus the common scalar h b^T for every vocab logit.
    That common scalar is softmax-invariant, but included here for raw-logit equality.
    """
    lm_head = getattr(adapter_model, "lm_head", None)
    if not hasattr(lm_head, "base_head") or not hasattr(lm_head, "affine"):
        raise TypeError("Expected lm_head to be wrapped by AffineLMHead.")
    affine = lm_head.affine
    backbone = getattr(adapter_model, "model", None)
    if backbone is None:
        raise TypeError("Expected a decoder backbone at model.model.")

    outputs = backbone(input_ids=input_ids)
    hidden = outputs.last_hidden_state
    x = hidden.to(dtype=affine.down.weight.dtype)
    low_rank = F.linear(F.linear(x, affine.up.weight.T), affine.down.weight.T) * affine.scale
    projected = (x + low_rank).to(dtype=lm_head.base_head.weight.dtype)
    logits = lm_head.base_head(projected)
    if affine.bias is not None:
        common_shift = (x * (affine.bias_scale * affine.bias.to(dtype=x.dtype))).sum(dim=-1, keepdim=True)
        logits = logits + common_shift.to(dtype=logits.dtype)
    return logits


def run_quick(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    encoded = tokenizer(args.text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    adapter_model = load_adapter_model(args.model_path, args.adapter_dir, args.base_dtype, device)
    adapter_model.eval()

    merged_model = load_base(args.model_path, args.base_dtype, device)
    apply_merged_tied_weight(merged_model, merged_embedding_weight(adapter_model))

    with torch.no_grad():
        current_logits = adapter_model(input_ids=input_ids).logits
        merged_logits = merged_model(input_ids=input_ids).logits
        transposed_logits = transposed_tied_lm_head_logits(adapter_model, input_ids)

    if args.max_new_tokens > 0:
        adapter_gen = adapter_model.generate(
            input_ids, max_new_tokens=args.max_new_tokens, do_sample=False
        )
        merged_gen = merged_model.generate(
            input_ids, max_new_tokens=args.max_new_tokens, do_sample=False
        )
        gen_texts = {
            "adapter": tokenizer.decode(adapter_gen[0], skip_special_tokens=True),
            "merged": tokenizer.decode(merged_gen[0], skip_special_tokens=True),
        }
    else:
        gen_texts = None

    result = {
        "mode": "quick",
        "model_path": args.model_path,
        "adapter_dir": args.adapter_dir,
        "input_shape": list(input_ids.shape),
        "current_wrapper_vs_merged": diff_stats(current_logits, merged_logits),
        "transposed_tied_form_vs_merged": diff_stats(transposed_logits, merged_logits),
    }
    if gen_texts is not None:
        result["generation"] = gen_texts

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


# ── full mode ──────────────────────────────────────────────────────────

def first_present(row: dict[str, Any], keys: list[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v)
    return ""


def normalize_example(row: dict[str, Any]) -> tuple[str, str]:
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        user_parts: list[str] = []
        assistant_parts: list[str] = []
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role")
            content = str(turn.get("content") or "")
            if role == "user" and content:
                user_parts.append(content)
            elif role == "assistant" and content:
                assistant_parts.append(content)
        if user_parts and assistant_parts:
            return "\n".join(user_parts), "\n".join(assistant_parts)
    q = first_present(row, ["query", "question", "instruction", "problem", "input", "prompt"])
    a = first_present(row, ["response", "answer", "output", "solution", "target", "completion"])
    if not q or not a:
        raise ValueError("eval row missing question/answer fields")
    return q, a


def load_eval(eval_data: str) -> Dataset:
    p = Path(eval_data)
    if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}:
        return load_dataset("json", data_files=str(p), split="train")
    if p.is_dir():
        loaded = load_from_disk(str(p))
        if isinstance(loaded, DatasetDict):
            return loaded["test"]
        return loaded
    return load_dataset(eval_data, split="test")


@torch.no_grad()
def compute_ppl(model: Any, tokenizer: Any, examples: list[dict], device: torch.device, max_seq_len: int) -> dict:
    total_loss = 0.0
    total_tokens = 0
    for ex in examples:
        q, a = normalize_example(ex)
        prompt = PROMPT_TEMPLATE.format(question=q)
        full = prompt + a + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        enc = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_seq_len, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        labels = input_ids.clone()
        plen = min(len(prompt_ids), labels.shape[1])
        labels[:, :plen] = -100
        n_lbl = int((labels != -100).sum().item())
        if n_lbl == 0:
            continue
        out = model(input_ids=input_ids, labels=labels)
        total_loss += float(out.loss.item()) * n_lbl
        total_tokens += n_lbl
    return {"avg_ce": total_loss / max(total_tokens, 1), "tokens": total_tokens}


@torch.no_grad()
def greedy_generate(model: Any, tokenizer: Any, prompts: list[str], device: torch.device, max_new_tokens: int) -> list[list[int]]:
    outs: list[list[int]] = []
    eos_id = tokenizer.eos_token_id
    for p in prompts:
        prompt_ids = tokenizer(p, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        cur = prompt_ids
        past = None
        next_in = cur
        for _ in range(max_new_tokens):
            out = model(input_ids=next_in, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_id = int(out.logits[0, -1].argmax().item())
            next_in = torch.tensor([[next_id]], device=device)
            cur = torch.cat([cur, next_in], dim=1)
            if eos_id is not None and next_id == eos_id:
                break
        gen = cur[0, prompt_ids.shape[1]:].cpu().tolist()
        outs.append(gen)
    return outs


def rouge_l_lcs(a: list[int], b: list[int]) -> float:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[m]
    p = lcs / m
    r = lcs / n
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def run_full(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    if not args.eval_data:
        raise ValueError("--eval-data is required for full mode (omit --quick).")

    eval_ds = load_eval(args.eval_data)
    n_eval = len(eval_ds)
    ppl_examples = [eval_ds[i] for i in range(min(args.ppl_samples, n_eval))]
    gen_examples = [eval_ds[i] for i in range(min(args.gen_samples, n_eval))]

    print(f"[{time.strftime('%H:%M:%S')}] loading adapter model from {args.model_path}", flush=True)
    adapter_model = load_adapter_model(args.model_path, args.adapter_dir, args.base_dtype, device)

    print(f"[{time.strftime('%H:%M:%S')}] computing merged tied weight + loading merged model", flush=True)
    merged_w = merged_embedding_weight(adapter_model)
    merged_model = load_base(args.model_path, args.base_dtype, device)
    apply_merged_tied_weight(merged_model, merged_w)

    print(f"[{time.strftime('%H:%M:%S')}] computing PPL on {len(ppl_examples)} examples", flush=True)
    t = time.time()
    ppl_a = compute_ppl(adapter_model, tokenizer, ppl_examples, device, args.max_seq_len)
    ppl_m = compute_ppl(merged_model, tokenizer, ppl_examples, device, args.max_seq_len)
    t_ppl = time.time() - t

    print(f"[{time.strftime('%H:%M:%S')}] greedy generation on {len(gen_examples)} examples", flush=True)
    prompts = [PROMPT_TEMPLATE.format(question=normalize_example(ex)[0]) for ex in gen_examples]
    t = time.time()
    gen_a = greedy_generate(adapter_model, tokenizer, prompts, device, args.gen_max_new_tokens)
    gen_m = greedy_generate(merged_model, tokenizer, prompts, device, args.gen_max_new_tokens)
    t_gen = time.time() - t

    total_pos = 0
    match_pos = 0
    full_match = 0
    rouge_sum = 0.0
    first_diff_positions: list[int] = []
    for a, m in zip(gen_a, gen_m):
        Lcommon = min(len(a), len(m))
        total_pos += Lcommon
        fd = -1
        for k in range(Lcommon):
            if a[k] != m[k]:
                fd = k
                break
        first_diff_positions.append(fd)
        match_pos += sum(1 for k in range(Lcommon) if a[k] == m[k])
        if a == m:
            full_match += 1
        rouge_sum += rouge_l_lcs(a, m)

    summary = {
        "mode": "full",
        "model_path": args.model_path,
        "adapter_dir": args.adapter_dir,
        "eval_data": args.eval_data,
        "base_dtype": args.base_dtype,
        "ppl": {
            "adapter_avg_ce": ppl_a["avg_ce"],
            "merged_avg_ce": ppl_m["avg_ce"],
            "adapter_ppl": float(torch.tensor(ppl_a["avg_ce"]).exp().item()),
            "merged_ppl": float(torch.tensor(ppl_m["avg_ce"]).exp().item()),
            "abs_delta_ce": abs(ppl_a["avg_ce"] - ppl_m["avg_ce"]),
            "rel_delta_ce": abs(ppl_a["avg_ce"] - ppl_m["avg_ce"]) / max(abs(ppl_a["avg_ce"]), 1e-9),
            "tokens": ppl_a["tokens"],
            "samples": len(ppl_examples),
            "seconds": t_ppl,
        },
        "gen": {
            "samples": len(gen_examples),
            "token_position_match_rate": match_pos / max(total_pos, 1),
            "full_sequence_match_rate": full_match / max(len(gen_examples), 1),
            "avg_rouge_l": rouge_sum / max(len(gen_examples), 1),
            "first_diff_min_position": min(p for p in first_diff_positions if p >= 0) if any(p >= 0 for p in first_diff_positions) else -1,
            "n_sequences_identical": sum(1 for p in first_diff_positions if p == -1),
            "seconds": t_gen,
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# ── main ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    if args.quick:
        run_quick(args)
    else:
        run_full(args)


if __name__ == "__main__":
    main()
