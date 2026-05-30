import os
import sys
import json

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


UNTIED_VARIANTS = {"s2", "s8", "s9", "s10"}
VALID_VARIANTS = {f"s{i}" for i in range(1, 14)}


def normalize_variant_args(args):
    args.embedding_variant = args.embedding_variant.lower()
    if args.embedding_variant not in VALID_VARIANTS:
        raise ValueError(f"--embedding_variant 必须是 s1-s13，当前为 {args.embedding_variant}")
    if args.embedding_variant_rank <= 0:
        raise ValueError("--embedding_variant_rank 必须为正整数")
    required_tie = 0 if args.embedding_variant in UNTIED_VARIANTS else 1
    if args.tie_word_embeddings != required_tie:
        Logger(f"[variant] {args.embedding_variant} 自动设置 tie_word_embeddings={required_tie}（原值 {args.tie_word_embeddings}）")
        args.tie_word_embeddings = required_tie
    return args


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    target_iters = min(iters, args.max_steps) if args.max_steps > 0 else iters
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        if args.max_steps > 0 and step > args.max_steps:
            break
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        probe_grad = (
            args.grad_log_interval > 0
            and step % args.grad_log_interval == 0
            and step % args.accumulation_steps == 0
        )
        if probe_grad:
            res.logits.retain_grad()
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            if probe_grad:
                log_embedding_grad_stats(epoch, step, res, labels)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == target_iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (target_iters - step) // 60
            mem_info = ""
            if "cuda" in args.device:
                allocated = torch.cuda.max_memory_allocated() / 1024**3
                reserved = torch.cuda.max_memory_reserved() / 1024**3
                mem_info = f", peak_alloc={allocated:.2f}GiB, peak_reserved={reserved:.2f}GiB"
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{target_iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min{mem_info}')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == target_iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir=args.checkpoint_dir)
            model.train()
            del state_dict

        del input_ids, labels, res, loss

        if args.max_steps > 0 and step >= args.max_steps:
            break

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


def _raw_model():
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(raw_model, '_orig_mod', raw_model)


def _grad_stats(tensor):
    if tensor is None:
        return {"norm": None, "rms": None}
    tensor = tensor.detach().float()
    return {
        "norm": float(torch.linalg.vector_norm(tensor).item()),
        "rms": float(torch.sqrt(torch.mean(tensor * tensor)).item()),
    }


def _all_reduce_mean(tensor):
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor


def _save_grad_tensor(tensor):
    if tensor is None:
        return None
    return tensor.detach().to(dtype=torch.float16, device="cpu")


def _untie_module_grads(raw):
    grads = {}
    for name, param in raw.named_parameters():
        if any(key in name for key in ("embedding_beta", "embedding_lora", "embedding_mul", "output_lora", "output_mul", "output_beta")):
            grads[name] = param.grad
    return grads


def log_embedding_grad_stats(epoch, step, res, labels):
    raw = _raw_model()
    embed_weight = raw.model.embed_tokens.weight
    unemb_weight = raw.lm_head.weight
    embed_grad = embed_weight.grad
    unemb_grad = unemb_weight.grad
    is_tied = bool(args.tie_word_embeddings)
    untie_grads = _untie_module_grads(raw)
    row = {
        "epoch": epoch + 1,
        "step": step,
        "variant": args.embedding_variant,
        "seed": args.seed,
        "tie_word_embeddings": is_tied,
        "embed_weight_norm": float(torch.linalg.vector_norm(embed_weight.detach().float()).item()),
        "unemb_weight_norm": float(torch.linalg.vector_norm(unemb_weight.detach().float()).item()),
        "untie_module_grads": {name: _grad_stats(grad) for name, grad in untie_grads.items()},
    }
    tensor_payload = None

    if is_tied and res.logits.grad is not None:
        logits_grad = res.logits.grad[..., :-1, :].detach().float()
        hidden = res.hidden_states[:, :-1, :].detach().float()
        direct_grad = torch.matmul(
            logits_grad.reshape(-1, logits_grad.size(-1)).t(),
            hidden.reshape(-1, hidden.size(-1)),
        )
        direct_grad = _all_reduce_mean(direct_grad)
        total_grad = embed_grad.detach().float()
        input_path_grad = total_grad - direct_grad
        row["embed_grad"] = _grad_stats(input_path_grad)
        row["unemb_grad"] = _grad_stats(direct_grad)
        row["shared_total_grad"] = _grad_stats(total_grad)
        row["tied_split"] = {
            "embed_unemb_grad_cosine": float(F.cosine_similarity(
                input_path_grad.flatten(),
                direct_grad.flatten(),
                dim=0,
                eps=1e-12,
            ).item()),
        }
        if args.grad_save_tensors == 1 and is_main_process():
            tensor_payload = {
                "metadata": {k: row[k] for k in ("epoch", "step", "variant", "seed", "tie_word_embeddings")},
                "embed_grad": _save_grad_tensor(input_path_grad),
                "unemb_grad": _save_grad_tensor(direct_grad),
                "shared_total_grad": _save_grad_tensor(total_grad),
                "untie_module_grads": {name: _save_grad_tensor(grad) for name, grad in untie_grads.items()},
            }
        del direct_grad, input_path_grad
    else:
        row["embed_grad"] = _grad_stats(embed_grad)
        row["unemb_grad"] = _grad_stats(unemb_grad)
        if args.grad_save_tensors == 1 and is_main_process():
            tensor_payload = {
                "metadata": {k: row[k] for k in ("epoch", "step", "variant", "seed", "tie_word_embeddings")},
                "embed_grad": _save_grad_tensor(embed_grad),
                "unemb_grad": _save_grad_tensor(unemb_grad),
                "untie_module_grads": {name: _save_grad_tensor(grad) for name, grad in untie_grads.items()},
            }

    if is_main_process():
        os.makedirs(os.path.dirname(args.grad_log_path) or ".", exist_ok=True)
        with open(args.grad_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if tensor_payload is not None:
            os.makedirs(args.grad_tensor_dir, exist_ok=True)
            tensor_path = os.path.join(
                args.grad_tensor_dir,
                f"{args.embedding_variant}_epoch{epoch + 1}_step{step}.pt",
            )
            torch.save(tensor_payload, tensor_path)
            Logger(f"[grad] wrote gradient tensor snapshot: {tensor_path}")
        Logger(f"[grad] wrote embedding grad stats: {args.grad_log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="weights/final", help="模型保存目录")
    parser.add_argument("--checkpoint_dir", type=str, default="weights/resume", help="续训快照保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--tie_word_embeddings', default=1, type=int, choices=[0, 1], help="是否绑定embed_tokens与lm_head（0=untied，参数+vocab_size*hidden_size）")
    parser.add_argument('--lm_head_bias', default=0, type=int, choices=[0, 1], help="lm_head是否使用vocab维度bias")
    parser.add_argument('--embedding_variant', default='s1', type=str, choices=sorted(VALID_VARIANTS), help="S1-S13 embedding/head 变体")
    parser.add_argument('--embedding_variant_rank', default=32, type=int, help="S4/S5/S6/S7/S9/S10/S12/S13 低秩rank")
    parser.add_argument("--data_path", type=str, default="../dataset/minimind/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument("--train_split_ratio", default=1.0, type=float, help="使用数据前多少比例训练，1.0表示全量")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--max_steps", default=0, type=int, help="每个epoch最多训练step数，0表示不限制")
    parser.add_argument("--seed", default=42, type=int, help="随机种子基值")
    parser.add_argument("--tokenizer_path", default="../model", type=str, help="Tokenizer路径或Hugging Face名称")
    parser.add_argument("--grad_log_interval", default=0, type=int, help="每隔多少个optimizer step记录embedding/unembedding梯度，0表示关闭")
    parser.add_argument("--grad_log_path", default="../logs/grad_stats.jsonl", type=str, help="梯度统计JSONL输出路径")
    parser.add_argument("--grad_save_tensors", default=0, type=int, choices=[0, 1], help="是否保存梯度tensor快照")
    parser.add_argument("--grad_tensor_dir", default="../logs/grad_tensors", type=str, help="梯度tensor快照输出目录")
    args = parser.parse_args()
    args = normalize_variant_args(args)

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        tie_word_embeddings=bool(args.tie_word_embeddings),
        lm_head_bias=bool(args.lm_head_bias),
        embedding_variant=args.embedding_variant,
        embedding_variant_rank=args.embedding_variant_rank
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=args.checkpoint_dir) if args.from_resume==1 else None
    Logger(f"[variant] embedding_variant={args.embedding_variant}, rank={args.embedding_variant_rank}, tie_word_embeddings={args.tie_word_embeddings}, lm_head_bias={args.lm_head_bias}, seed={args.seed}")
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-{args.embedding_variant}-Seed-{args.seed}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, tokenizer_path=args.tokenizer_path, device=args.device)
    train_end_index = None
    if args.train_split_ratio < 1.0:
        if not 0 < args.train_split_ratio < 1:
            raise ValueError("--train_split_ratio 必须在 (0, 1] 范围内")
        total_samples = len(PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len))
        train_end_index = int(total_samples * args.train_split_ratio)
        Logger(f"[data] train split: [0, {train_end_index}) / total {total_samples} ({args.train_split_ratio:.2%})")
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len, end_index=train_end_index)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()