#!/usr/bin/env python
"""Train hidden LoRA and/or affine vocab adapters for the main SFT task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from affine_vocab_lora import AffineVocabConfig, apply_affine_vocab_adapters, save_affine_vocab_adapter


MASTER_DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


PROMPT_TEMPLATE = """Question:
{question}

Answer:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--variant",
        choices=[
            "full_finetune",
            "hidden_lora",
            "affine_input",
            "affine_lm_head",
            "affine_input_lm_head",
            "affine_input_plus_hidden_lora",
            "affine_lm_head_plus_hidden_lora",
            "affine_input_lm_head_plus_hidden_lora",
        ],
        required=True,
    )
    parser.add_argument("--hidden-lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj")
    parser.add_argument(
        "--hidden-lora-layers-to-transform",
        default=None,
        help=(
            "Comma-separated layer indices to apply LoRA to (e.g. '14' or '0,13,27')."
            " If omitted, LoRA is applied to every decoder layer (default behavior)."
            " Useful for matched-capacity comparisons against affine adapters."
        ),
    )
    parser.add_argument("--affine-rank", type=int, default=16)
    parser.add_argument("--affine-alpha", type=float, default=32.0)
    parser.add_argument(
        "--affine-bias-scale",
        type=float,
        default=1.0,
        help="s2 in W' = W (I + s1 AB) + s2 b. Default 1.0 keeps prior behavior.",
    )
    parser.add_argument("--affine-dropout", type=float, default=0.0)
    parser.add_argument("--no-affine-input-bias", action="store_true")
    parser.add_argument("--affine-lm-head-bias", action="store_true")
    parser.add_argument("--vocab-logit-bias", action="store_true")
    parser.add_argument(
        "--tie-affine-input-lm-head-adapters",
        action="store_true",
        help=(
            "For tied-embedding models, share one affine adapter between input"
            " embeddings and lm_head so the learned adapter can be merged back"
            " into the single tied embedding matrix."
        ),
    )
    parser.add_argument(
        "--mergeable-tied-affine-output",
        action="store_true",
        help=(
            "When sharing tied input/lm_head affine adapters, apply the transpose"
            " affine on the lm_head side so raw logits match a model with the"
            " adapter merged into the tied embedding matrix."
        ),
    )
    parser.add_argument(
        "--affine-intermediate-layer-idx",
        type=int,
        default=None,
        help=(
            "If set, place the affine adapter at the output of decoder layer idx"
            " instead of the input-embedding lookup. Used for position ablation."
        ),
    )
    parser.add_argument(
        "--affine-use-after-norm",
        action="store_true",
        help="Place the affine adapter after the final RMSNorm (just before lm_head).",
    )
    parser.add_argument("--hidden-lora-rank", type=int, default=16)
    parser.add_argument("--hidden-lora-alpha", type=int, default=32)
    parser.add_argument("--hidden-lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-dora", action="store_true", help="Enable PEFT DoRA on the hidden LoRA.")
    parser.add_argument("--use-rslora", action="store_true", help="Enable PEFT rsLoRA on the hidden LoRA.")
    parser.add_argument(
        "--include-emb-lmh-lora-rank",
        type=int,
        default=0,
        help=(
            "Claim 1b control: if > 0, additionally place vocab-dim LoRA on"
            " embed_tokens and lm_head at the given rank, on top of the normal"
            " hidden_lora variant. Default 0 disables this. Most useful at"
            " rank=1 since vocab-dim LoRA cost scales with vocab_size * rank."
        ),
    )
    parser.add_argument(
        "--emb-lmh-lora-alpha",
        type=int,
        default=None,
        help="Alpha for the emb/lm_head LoRA. Defaults to 2*include_emb_lmh_lora_rank.",
    )
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--master-dtype",
        choices=list(MASTER_DTYPE_MAP),
        default="fp32",
        help=(
            "Storage dtype for *trainable* adapter parameters. Defaults to fp32 to avoid"
            " the bf16-master-weight pitfall documented in cpt/实验记录.md §9: with bf16"
            " params + bf16 AdamW state, lr=2e-4 updates get rounded away and training"
            " silently stalls."
        ),
    )
    parser.add_argument(
        "--base-dtype",
        choices=["auto", "fp32", "bf16", "fp16"],
        default="auto",
        help="Storage dtype for the *frozen* base model. 'auto' uses config.torch_dtype.",
    )
    parser.add_argument("--save-strategy", default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--skip-final-model-save", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--eval-data", default=None, help="Optional held-out JSONL/HF path for eval loss.")
    parser.add_argument("--eval-samples", type=int, default=0)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    return parser.parse_args()


def load_any_dataset(data_ref: str, split: str) -> Dataset:
    path = Path(data_ref)
    if path.exists() and path.is_dir():
        loaded = load_from_disk(str(path))
        if isinstance(loaded, DatasetDict):
            return loaded[split]
        return loaded
    if path.exists():
        suffix = path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            return load_dataset("json", data_files=str(path), split="train")
        raise ValueError(f"Unsupported local data suffix: {suffix}")
    return load_dataset(data_ref, split=split)


def first_present(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def normalize_example(row: dict[str, Any]) -> tuple[str, str]:
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        user_parts = []
        assistant_parts = []
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

    question = first_present(row, ["query", "question", "instruction", "problem", "input", "prompt"])
    answer = first_present(row, ["response", "answer", "output", "solution", "target", "completion"])
    if not question or not answer:
        raise ValueError(
            "Each training row must contain a question-like field and an answer-like field."
        )
    return question, answer


def tokenize_row(row: dict[str, Any], tokenizer: Any, max_seq_len: int) -> dict[str, list[int]]:
    question, answer = normalize_example(row)
    prompt = PROMPT_TEMPLATE.format(question=question)
    full_text = prompt + answer + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_seq_len)
    input_ids = full["input_ids"]
    labels = input_ids.copy()
    labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
    return {"input_ids": input_ids, "attention_mask": full["attention_mask"], "labels": labels}


def variant_uses_affine(variant: str) -> tuple[bool, bool]:
    return "affine_input" in variant, "lm_head" in variant


def variant_uses_hidden_lora(variant: str) -> bool:
    return variant == "hidden_lora" or variant.endswith("plus_hidden_lora")


def variant_uses_full_finetune(variant: str) -> bool:
    return variant == "full_finetune"


def trainable_summary(model: Any) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "pct": trainable / total * 100}


def enable_affine_trainable(model: Any) -> int:
    count = 0
    for name, param in model.named_parameters():
        if ".affine." in name or "vocab_logit_bias" in name:
            param.requires_grad_(True)
            count += param.numel()
    return count


def cast_trainable_to_master_dtype(model: Any, master_dtype: torch.dtype) -> dict[str, int]:
    """Force every trainable param to ``master_dtype`` (typically fp32).

    PEFT 0.19's ``_move_adapter_to_device_of_base_layer`` casts LoRA params to the base
    layer's dtype. If the base is loaded in bf16, LoRA + AdamW state end up bf16, which
    silently kills small lr updates (cpt/实验记录.md §9). Call this after PEFT wrapping
    and after enable_affine_trainable to make sure all trainable storage is fp32.
    """

    cast = 0
    already = 0
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dtype == master_dtype:
            already += 1
            continue
        param.data = param.data.to(dtype=master_dtype)
        cast += 1
    return {"cast": cast, "already": already}


def precision_sanity_report(model: Any, master_dtype_arg: str) -> dict[str, Any]:
    trainable_dtypes: dict[str, int] = {}
    sample = None
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        key = str(param.dtype)
        trainable_dtypes[key] = trainable_dtypes.get(key, 0) + 1
        if sample is None:
            sample = (name, key)
    frozen_dtypes: dict[str, int] = {}
    for _, param in model.named_parameters():
        if param.requires_grad:
            continue
        key = str(param.dtype)
        frozen_dtypes[key] = frozen_dtypes.get(key, 0) + 1
    return {
        "master_dtype_arg": master_dtype_arg,
        "trainable_param_dtypes": trainable_dtypes,
        "frozen_param_dtypes": frozen_dtypes,
        "sample_trainable_param": sample,
    }


class SaveAffineAdapterCallback(TrainerCallback):
    """Mirror Trainer's intermediate checkpoint saves for the affine vocab adapter.

    Trainer's ``save_model`` only serialises PEFT adapter files. Without this callback,
    a job that dies mid-training would lose every affine vocab update.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def _resolve_base(self) -> Any:
        return self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model

    def on_save(self, args, state, control, **kwargs):  # noqa: D401, ANN001
        if not state.is_world_process_zero:
            return
        if getattr(self._resolve_base(), "affine_vocab_config", None) is None:
            return
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_affine_vocab_adapter(self._resolve_base(), ckpt_dir)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_dtype_arg = args.base_dtype
    base_dtype: Any
    if base_dtype_arg == "auto":
        base_dtype = "auto"
    else:
        base_dtype = MASTER_DTYPE_MAP[base_dtype_arg]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=base_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    use_full_finetune = variant_uses_full_finetune(args.variant)
    use_input, use_lm_head = variant_uses_affine(args.variant)
    if use_input or use_lm_head:
        cfg = AffineVocabConfig(
            hidden_size=model.get_input_embeddings().embedding_dim,
            rank=args.affine_rank,
            alpha=args.affine_alpha,
            bias_scale=args.affine_bias_scale,
            dropout=args.affine_dropout,
            use_input=use_input,
            use_lm_head=use_lm_head,
            use_input_bias=not args.no_affine_input_bias,
            use_lm_head_bias=args.affine_lm_head_bias,
            use_vocab_logit_bias=args.vocab_logit_bias,
            intermediate_layer_idx=args.affine_intermediate_layer_idx,
            use_after_norm=args.affine_use_after_norm,
            tie_input_lm_head_adapters=args.tie_affine_input_lm_head_adapters,
            use_tied_lm_head_transpose=args.mergeable_tied_affine_output,
        )
        apply_affine_vocab_adapters(model, cfg)

    if use_full_finetune:
        for param in model.parameters():
            param.requires_grad_(True)
        print(json.dumps(trainable_summary(model), indent=2))
    elif variant_uses_hidden_lora(args.variant):
        layers_to_transform: list[int] | None = None
        if args.hidden_lora_layers_to_transform:
            layers_to_transform = [
                int(idx.strip())
                for idx in args.hidden_lora_layers_to_transform.split(",")
                if idx.strip()
            ]
        target_modules = [
            m.strip() for m in args.hidden_lora_target_modules.split(",") if m.strip()
        ]
        rank_pattern: dict[str, int] = {}
        alpha_pattern: dict[str, int] = {}
        if args.include_emb_lmh_lora_rank > 0:
            emb_rank = args.include_emb_lmh_lora_rank
            emb_alpha = args.emb_lmh_lora_alpha if args.emb_lmh_lora_alpha is not None else (2 * emb_rank)
            for m in ("embed_tokens", "lm_head"):
                if m not in target_modules:
                    target_modules.append(m)
                rank_pattern[m] = emb_rank
                alpha_pattern[m] = emb_alpha
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.hidden_lora_rank,
            lora_alpha=args.hidden_lora_alpha,
            lora_dropout=args.hidden_lora_dropout,
            target_modules=target_modules,
            layers_to_transform=layers_to_transform,
            rank_pattern=rank_pattern,
            alpha_pattern=alpha_pattern,
            use_dora=args.use_dora,
            use_rslora=args.use_rslora,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        if use_input or use_lm_head:
            affine_count = enable_affine_trainable(model)
            print(f"re-enabled affine vocab trainable params: {affine_count}")
        model.print_trainable_parameters()
    else:
        print(json.dumps(trainable_summary(model), indent=2))

    master_dtype = MASTER_DTYPE_MAP[args.master_dtype]
    cast_stats = cast_trainable_to_master_dtype(model, master_dtype)
    print(f"[precision] cast_to_master_dtype={args.master_dtype} stats={cast_stats}")
    print(json.dumps({"precision_sanity": precision_sanity_report(model, args.master_dtype)}, indent=2))

    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    train_ds = load_any_dataset(args.train_data, args.dataset_split)
    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))

    tokenized = train_ds.map(
        lambda row: tokenize_row(row, tokenizer, args.max_seq_len),
        remove_columns=train_ds.column_names,
        desc="Tokenizing",
    )

    eval_tokenized = None
    eval_strategy = "no"
    if args.eval_data and args.eval_samples > 0:
        eval_ds = load_any_dataset(args.eval_data, "test")
        eval_ds = eval_ds.select(range(min(args.eval_samples, len(eval_ds))))
        eval_tokenized = eval_ds.map(
            lambda row: tokenize_row(row, tokenizer, args.max_seq_len),
            remove_columns=eval_ds.column_names,
            desc="Tokenizing eval",
        )
        eval_strategy = "steps"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps if eval_strategy != "no" else None,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    callbacks: list[TrainerCallback] = []
    if (use_input or use_lm_head) and args.save_strategy != "no":
        callbacks.append(SaveAffineAdapterCallback(model))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=eval_tokenized,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
        ),
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if trainer.is_world_process_zero() and trainer.optimizer is not None:
        adam_state_dtypes: dict[str, int] = {}
        for state in trainer.optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                t = state.get(key)
                if isinstance(t, torch.Tensor):
                    name = f"{key}:{t.dtype}"
                    adam_state_dtypes[name] = adam_state_dtypes.get(name, 0) + 1
        print(f"[precision] post_train adam_state_dtypes={adam_state_dtypes}")

    if trainer.is_world_process_zero():
        if args.skip_final_model_save:
            pass
        elif use_input or use_lm_head:
            target = model.get_base_model() if hasattr(model, "get_base_model") else model
            save_affine_vocab_adapter(target, output_dir)
        if use_full_finetune:
            model.save_pretrained(str(output_dir))
        elif variant_uses_hidden_lora(args.variant):
            model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        with (output_dir / "run_args.json").open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        with (output_dir / "trainable_summary.json").open("w", encoding="utf-8") as f:
            target = model.get_base_model() if hasattr(model, "get_base_model") else model
            json.dump(trainable_summary(target), f, indent=2)


if __name__ == "__main__":
    main()
