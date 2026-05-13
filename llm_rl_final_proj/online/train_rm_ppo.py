from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn

from llm_rl_final_proj.data.ultrafeedback import GenerationExample, build_generation_examples, dataset_overview
from llm_rl_final_proj.models.load import (
    load_lora_policy_model_and_tokenizer,
    load_reward_model_and_tokenizer,
)
from llm_rl_final_proj.offline.evaluation import generate_samples, summarize_generation_rows
from llm_rl_final_proj.reward_model.evaluation import score_prompt_response_pairs
from llm_rl_final_proj.rl.base import AlgoConfig
from llm_rl_final_proj.rl.grpo import GRPO
from llm_rl_final_proj.rollout.hf_sampler import HFSampler, SamplingConfig
from llm_rl_final_proj.rollout.rollout_buffer import RolloutBatch
from llm_rl_final_proj.utils.hardware import (
    get_cuda_memory_metrics,
    get_hardware_metrics,
    get_model_device_metrics,
    require_cuda_if_requested,
    resolve_device_and_dtype,
)
from llm_rl_final_proj.utils.seed import set_seed
from llm_rl_final_proj.utils.wandb_utils import WandBLogger


@dataclass
class OnlineRMPPOConfig:
    algo: str = "ppo"
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    reward_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    reward_adapter_path: str = ""
    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"
    train_split: str = "train_gen"
    eval_split: str = "test_gen"
    output_dir: str = "runs/rm_ppo_default"

    seed: int = 0
    steps: int = 101
    batch_size: int = 8
    group_size: int = 4

    min_new_tokens: int = 8
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0

    lr: float = 3e-5
    weight_decay: float = 0.0
    betas1: float = 0.9
    betas2: float = 0.95
    warmup_steps: int = 20
    grad_accum_steps: int = 1
    max_grad_norm: float = 0.5

    ppo_epochs: int = 2
    minibatch_size: int = 8
    clip_eps: float = 0.1
    clip_eps_high: float = 0.0
    kl_coef: float = 0.02
    adv_clip: float = 5.0

    max_prompt_tokens: int = 700
    max_response_tokens: int = 256
    train_limit: int = 0
    eval_limit: int = 64
    reward_batch_size: int = 16

    eval_interval: int = 25
    save_interval: int = 50
    eval_max_new_tokens: int = 256
    eval_temperature: float = 0.0
    eval_top_p: float = 1.0
    eval_batch_size: int = 8

    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    lora_bias: str = "none"
    grad_checkpointing: bool = True

    wandb_project: str = "llm-rl-final-project"
    wandb_name: str = "rm_ppo"
    wandb_enabled: bool = True
    sample_log_n: int = 8
    sample_log_max_chars: int = 2500

    # PPO-specific for value-head
    vf_lr: float = 1e-4
    vf_coef: float = 0.5
    vf_clip_eps: float = 0.2
    vf_epochs: int = 1


def parse_args() -> OnlineRMPPOConfig:
    ap = argparse.ArgumentParser(description="Train a policy online with PPO (value-baseline advantages + GRPO clipped surrogate).")
    ap.add_argument("--algo", type=str, default=OnlineRMPPOConfig.algo)
    ap.add_argument("--model_name", type=str, default=OnlineRMPPOConfig.model_name)
    ap.add_argument("--reward_model_name", type=str, default=OnlineRMPPOConfig.reward_model_name)
    ap.add_argument("--reward_adapter_path", type=str, required=True)
    ap.add_argument("--dataset_name", type=str, default=OnlineRMPPOConfig.dataset_name)
    ap.add_argument("--train_split", type=str, default=OnlineRMPPOConfig.train_split)
    ap.add_argument("--eval_split", type=str, default=OnlineRMPPOConfig.eval_split)
    ap.add_argument("--output_dir", type=str, default=OnlineRMPPOConfig.output_dir)

    ap.add_argument("--seed", type=int, default=OnlineRMPPOConfig.seed)
    ap.add_argument("--steps", type=int, default=OnlineRMPPOConfig.steps)
    ap.add_argument("--batch_size", type=int, default=OnlineRMPPOConfig.batch_size)
    ap.add_argument("--group_size", type=int, default=OnlineRMPPOConfig.group_size)

    ap.add_argument("--min_new_tokens", type=int, default=OnlineRMPPOConfig.min_new_tokens)
    ap.add_argument("--max_new_tokens", type=int, default=OnlineRMPPOConfig.max_new_tokens)
    ap.add_argument("--temperature", type=float, default=OnlineRMPPOConfig.temperature)
    ap.add_argument("--top_p", type=float, default=OnlineRMPPOConfig.top_p)
    ap.add_argument("--top_k", type=int, default=OnlineRMPPOConfig.top_k)
    ap.add_argument("--repetition_penalty", type=float, default=OnlineRMPPOConfig.repetition_penalty)

    ap.add_argument("--lr", type=float, default=OnlineRMPPOConfig.lr)
    ap.add_argument("--weight_decay", type=float, default=OnlineRMPPOConfig.weight_decay)
    ap.add_argument("--betas1", type=float, default=OnlineRMPPOConfig.betas1)
    ap.add_argument("--betas2", type=float, default=OnlineRMPPOConfig.betas2)
    ap.add_argument("--warmup_steps", type=int, default=OnlineRMPPOConfig.warmup_steps)
    ap.add_argument("--grad_accum_steps", type=int, default=OnlineRMPPOConfig.grad_accum_steps)
    ap.add_argument("--max_grad_norm", type=float, default=OnlineRMPPOConfig.max_grad_norm)

    ap.add_argument("--ppo_epochs", type=int, default=OnlineRMPPOConfig.ppo_epochs)
    ap.add_argument("--minibatch_size", type=int, default=OnlineRMPPOConfig.minibatch_size)
    ap.add_argument("--clip_eps", type=float, default=OnlineRMPPOConfig.clip_eps)
    ap.add_argument("--clip_eps_high", type=float, default=OnlineRMPPOConfig.clip_eps_high)
    ap.add_argument("--kl_coef", type=float, default=OnlineRMPPOConfig.kl_coef)
    ap.add_argument("--adv_clip", type=float, default=OnlineRMPPOConfig.adv_clip)

    ap.add_argument("--max_prompt_tokens", type=int, default=OnlineRMPPOConfig.max_prompt_tokens)
    ap.add_argument("--max_response_tokens", type=int, default=OnlineRMPPOConfig.max_response_tokens)
    ap.add_argument("--train_limit", type=int, default=OnlineRMPPOConfig.train_limit)
    ap.add_argument("--eval_limit", type=int, default=OnlineRMPPOConfig.eval_limit)
    ap.add_argument("--reward_batch_size", type=int, default=OnlineRMPPOConfig.reward_batch_size)

    ap.add_argument("--eval_interval", type=int, default=OnlineRMPPOConfig.eval_interval)
    ap.add_argument("--save_interval", type=int, default=OnlineRMPPOConfig.save_interval)
    ap.add_argument("--eval_max_new_tokens", type=int, default=OnlineRMPPOConfig.eval_max_new_tokens)
    ap.add_argument("--eval_temperature", type=float, default=OnlineRMPPOConfig.eval_temperature)
    ap.add_argument("--eval_top_p", type=float, default=OnlineRMPPOConfig.eval_top_p)
    ap.add_argument("--eval_batch_size", type=int, default=OnlineRMPPOConfig.eval_batch_size)

    ap.add_argument("--lora_r", type=int, default=OnlineRMPPOConfig.lora_r)
    ap.add_argument("--lora_alpha", type=int, default=OnlineRMPPOConfig.lora_alpha)
    ap.add_argument("--lora_dropout", type=float, default=OnlineRMPPOConfig.lora_dropout)
    ap.add_argument("--lora_target_modules", type=str, default=OnlineRMPPOConfig.lora_target_modules)
    ap.add_argument("--lora_bias", type=str, default=OnlineRMPPOConfig.lora_bias)
    ap.add_argument(
        "--grad_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=OnlineRMPPOConfig.grad_checkpointing,
    )

    ap.add_argument("--wandb_project", type=str, default=OnlineRMPPOConfig.wandb_project)
    ap.add_argument("--wandb_name", type=str, default=OnlineRMPPOConfig.wandb_name)
    ap.add_argument(
        "--wandb_enabled",
        action=argparse.BooleanOptionalAction,
        default=OnlineRMPPOConfig.wandb_enabled,
    )
    ap.add_argument("--sample_log_n", type=int, default=OnlineRMPPOConfig.sample_log_n)
    ap.add_argument("--sample_log_max_chars", type=int, default=OnlineRMPPOConfig.sample_log_max_chars)

    ap.add_argument("--vf_lr", type=float, default=OnlineRMPPOConfig.vf_lr)
    ap.add_argument("--vf_coef", type=float, default=OnlineRMPPOConfig.vf_coef)
    ap.add_argument("--vf_clip_eps", type=float, default=OnlineRMPPOConfig.vf_clip_eps)
    ap.add_argument("--vf_epochs", type=int, default=OnlineRMPPOConfig.vf_epochs)

    args = ap.parse_args()
    return OnlineRMPPOConfig(**vars(args))


def maybe_update_warmup_lr(optimizer: torch.optim.Optimizer, base_lr: float, step: int, warmup_steps: int) -> None:
    if warmup_steps <= 0:
        scale = 1.0
    else:
        scale = min(1.0, float(step + 1) / float(warmup_steps))
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * scale


def _normalize_lora_target_modules(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _sample_prompt_batch(examples: Sequence[GenerationExample], batch_size: int, rng: random.Random) -> List[GenerationExample]:
    return [examples[rng.randrange(len(examples))] for _ in range(batch_size)]


def _get_model_hidden_size(model: torch.nn.Module) -> int:
    if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
        return int(model.config.hidden_size)
    if hasattr(model, "base_model"):
        return _get_model_hidden_size(model.base_model)
    raise ValueError("Can't determine hidden_size from model config.")


@torch.no_grad()
def _compute_sequence_values(
    model: torch.nn.Module,
    value_head: nn.Linear,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the policy in eval mode; return scalar value estimates from the last non-padded hidden state."""
    was_training = model.training
    model.eval()
    model.config.use_cache = False
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    last_pos = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
    last_hidden = outputs.hidden_states[-1][batch_idx, last_pos].float()
    values = value_head(last_hidden).squeeze(-1)
    if was_training:
        model.train()
        model.config.use_cache = False
    return values


def _compute_value_advantages(rewards: torch.Tensor, old_values: torch.Tensor) -> torch.Tensor:
    """Advantage = reward - value baseline, then globally z-scored."""
    adv = rewards - old_values.detach()
    return (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)


def _train_value_head(
    model: torch.nn.Module,
    value_head: nn.Linear,
    vf_optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    returns: torch.Tensor,
    cfg: OnlineRMPPOConfig,
) -> Dict[str, float]:
    """Train the value head with PPO-clipped MSE. Policy model parameters are frozen."""
    device = next(value_head.parameters()).device
    N = rollout.input_ids.size(0)
    model.eval()
    model.config.use_cache = False
    with torch.no_grad():
        outputs = model(
            input_ids=rollout.input_ids.to(device),
            attention_mask=rollout.attention_mask.to(device),
            output_hidden_states=True,
            use_cache=False,
        )
        last_pos = rollout.attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(N, device=device)
        all_hidden = outputs.hidden_states[-1][batch_idx, last_pos].float().detach()
    model.train()
    model.config.use_cache = False

    returns_dev = returns.to(device).float().detach()
    old_values_dev = value_head(all_hidden).squeeze(-1).detach()

    total_vf_loss = 0.0
    n_mb = 0
    for _epoch in range(max(1, cfg.vf_epochs)):
        perm = torch.randperm(N, device=device)
        for start in range(0, N, cfg.minibatch_size):
            idx = perm[start : start + cfg.minibatch_size]
            new_v = value_head(all_hidden[idx]).squeeze(-1)
            ret = returns_dev[idx]
            old_v = old_values_dev[idx]

            vf_optimizer.zero_grad(set_to_none=True)
            vf_unclipped = (new_v - ret)**2
            vf_clipped = (old_v + (new_v - old_v).clamp(-cfg.vf_clip_eps, cfg.vf_clip_eps) - ret)**2
            vf_loss = torch.max(vf_unclipped, vf_clipped).mean()
            if not torch.isfinite(vf_loss):
                continue
            vf_loss.backward()
            vf_optimizer.step()
            total_vf_loss += float(vf_loss.detach().item())
            n_mb += 1

    return {"train/value_head_loss": total_vf_loss / max(1, n_mb)}


def _build_online_algo(cfg: OnlineRMPPOConfig) -> GRPO:
    algo_cfg = AlgoConfig(
        ppo_epochs=cfg.ppo_epochs,
        minibatch_size=cfg.minibatch_size,
        clip_eps=cfg.clip_eps,
        clip_eps_high=cfg.clip_eps_high,
        kl_coef=cfg.kl_coef,
        max_grad_norm=cfg.max_grad_norm,
        adv_clip=cfg.adv_clip,
        seed=cfg.seed,
    )
    return GRPO(algo_cfg)


def _normalize_completion_for_reward_scoring(text: str) -> str:
    if text.strip():
        return text
    return "[no response]"


def _truncate(text: str | None, max_chars: int) -> str | None:
    if text is None:
        return None
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + " ...[truncated]"


def _sample_rows_for_logging(
    examples: Sequence[GenerationExample],
    rows: Sequence[Dict[str, Any]],
    rm_scores: Sequence[float],
    *,
    sample_log_n: int,
    max_chars: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ex, row, score in list(zip(examples, rows, rm_scores))[: max(0, sample_log_n)]:
        out.append(
            {
                "row_id": ex.row_id,
                "prompt": _truncate(ex.prompt_text, max_chars),
                "reference_response": _truncate(ex.reference_response_text, max_chars),
                "model_response": _truncate(str(row.get("model_response", "")), max_chars),
                "reward_model_score": float(score),
            }
        )
    return out


def save_checkpoint(model: torch.nn.Module, value_head: nn.Linear, cfg: OnlineRMPPOConfig, step: int) -> None:
    ckpt_dir = Path(cfg.output_dir) / "checkpoints" / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = ckpt_dir / "adapter"
    model.save_pretrained(adapter_dir)
    torch.save(value_head.state_dict(), ckpt_dir / "value_head.pt")
    meta = {
        "step": step,
        "model_type": "online_policy_rm_ppo",
        "algo": cfg.algo,
        "model_name": cfg.model_name,
        "reward_model_name": cfg.reward_model_name,
        "reward_adapter_path": cfg.reward_adapter_path,
        "dataset_name": cfg.dataset_name,
        "train_split": cfg.train_split,
        "eval_split": cfg.eval_split,
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


@torch.no_grad()
def evaluate_policy_with_reward_model(
    *,
    policy_model: torch.nn.Module,
    policy_tokenizer,
    reward_model: torch.nn.Module,
    reward_tokenizer,
    examples: Sequence[GenerationExample],
    device: torch.device,
    max_prompt_tokens: int,
    max_response_tokens: int,
    generation_max_new_tokens: int,
    temperature: float,
    top_p: float,
    generation_batch_size: int,
) -> tuple[Dict[str, float], List[Dict[str, Any]], List[float]]:
    rows = generate_samples(
        policy_model,
        policy_tokenizer,
        examples,
        device=device,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=generation_max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        batch_size=generation_batch_size,
    )
    metrics = summarize_generation_rows(rows)
    scoring_rows = []
    reference_rows = []
    has_reference = True
    for ex, row in zip(examples, rows):
        scoring_rows.append(
            {
                "row_id": ex.row_id,
                "prompt_messages": ex.prompt_messages,
                "prompt_text": ex.prompt_text,
                "response_text": _normalize_completion_for_reward_scoring(str(row["model_response"])),
            }
        )
        if ex.reference_response_text:
            reference_rows.append(
                {
                    "row_id": ex.row_id,
                    "prompt_messages": ex.prompt_messages,
                    "prompt_text": ex.prompt_text,
                    "response_text": ex.reference_response_text,
                }
            )
        else:
            has_reference = False
    rm_scores = score_prompt_response_pairs(
        reward_model,
        reward_tokenizer,
        scoring_rows,
        max_prompt_tokens=max_prompt_tokens,
        max_response_tokens=max_response_tokens,
        per_device_batch_size=generation_batch_size,
        device=device,
    )
    score_tensor = torch.tensor(rm_scores, dtype=torch.float32)
    metrics["eval/rm_score_mean_on_policy_generations"] = float(score_tensor.mean().item())
    metrics["eval/rm_score_std_on_policy_generations"] = float(score_tensor.std(unbiased=False).item())
    if has_reference and reference_rows:
        ref_scores = score_prompt_response_pairs(
            reward_model,
            reward_tokenizer,
            reference_rows,
            max_prompt_tokens=max_prompt_tokens,
            max_response_tokens=max_response_tokens,
            per_device_batch_size=generation_batch_size,
            device=device,
        )
        ref_tensor = torch.tensor(ref_scores, dtype=torch.float32)
        margin = score_tensor - ref_tensor
        metrics["eval/rm_reference_score_mean_on_dataset_reference_responses"] = float(ref_tensor.mean().item())
        metrics["eval/rm_fraction_policy_scores_above_reference"] = float((margin > 0).float().mean().item())
        metrics["eval/rm_margin_policy_minus_reference_mean"] = float(margin.mean().item())
    return metrics, rows, rm_scores


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    require_cuda_if_requested()
    if cfg.steps <= 0:
        raise ValueError(f"--steps must be >= 1, got {cfg.steps}")
    if cfg.batch_size <= 0:
        raise ValueError(f"--batch_size must be >= 1, got {cfg.batch_size}")
    if cfg.group_size <= 0:
        raise ValueError(f"--group_size must be >= 1, got {cfg.group_size}")
    if not cfg.reward_adapter_path:
        raise ValueError("--reward_adapter_path is required")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_online_rm_ppo_config.json").write_text(
        json.dumps(vars(cfg), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rng = random.Random(cfg.seed)
    device, dtype = resolve_device_and_dtype()
    print(
        f"[setup] device={device} dtype={dtype} algo={cfg.algo} "
        f"policy={cfg.model_name} reward_model={cfg.reward_model_name}"
    )
    print("[setup][hardware]", json.dumps(get_hardware_metrics(device), indent=2, sort_keys=True))

    dataset_info = dataset_overview(cfg.dataset_name)
    train_examples = build_generation_examples(cfg.dataset_name, cfg.train_split, limit=cfg.train_limit)
    eval_examples = build_generation_examples(cfg.dataset_name, cfg.eval_split, limit=cfg.eval_limit)
    if not train_examples:
        raise RuntimeError("Training generation split produced zero examples.")
    if not eval_examples:
        raise RuntimeError("Evaluation generation split produced zero examples.")

    loaded_policy = load_lora_policy_model_and_tokenizer(
        cfg.model_name,
        device=device,
        dtype=dtype,
        grad_checkpointing=cfg.grad_checkpointing,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_target_modules=_normalize_lora_target_modules(cfg.lora_target_modules),
        lora_bias=cfg.lora_bias,
    )
    policy_model = loaded_policy.model
    policy_tokenizer = loaded_policy.tokenizer

    loaded_reward = load_reward_model_and_tokenizer(
        cfg.reward_model_name,
        device=device,
        dtype=dtype,
        adapter_path=cfg.reward_adapter_path,
    )
    reward_model = loaded_reward.model
    reward_tokenizer = loaded_reward.tokenizer
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=cfg.lr,
        betas=(cfg.betas1, cfg.betas2),
        weight_decay=cfg.weight_decay,
    )
    algo = _build_online_algo(cfg)

    hidden_size = _get_model_hidden_size(policy_model)
    value_head = nn.Linear(hidden_size, 1, bias=True).to(device).float()
    nn.init.zeros_(value_head.weight)
    nn.init.zeros_(value_head.bias)
    vf_optimizer = torch.optim.AdamW(
        value_head.parameters(),
        lr=cfg.vf_lr,
        betas=(cfg.betas1, cfg.betas2),
        weight_decay=cfg.weight_decay,
    )

    sampler = HFSampler(policy_tokenizer, device=device)
    sampling_cfg = SamplingConfig(
        min_new_tokens=cfg.min_new_tokens,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        do_sample=cfg.temperature > 0.0,
    )

    logger = WandBLogger(
        project=cfg.wandb_project,
        run_name=cfg.wandb_name,
        config=vars(cfg),
        enabled=cfg.wandb_enabled,
        local_dir=output_dir,
    )
    logger.log(
        {
            "setup/trainable_params": float(loaded_policy.trainable_params),
            "setup/total_params": float(loaded_policy.total_params),
            "setup/trainable_fraction": float(loaded_policy.trainable_params / max(1, loaded_policy.total_params)),
            "dataset/train_examples": float(len(train_examples)),
            "dataset/eval_examples": float(len(eval_examples)),
            **{f"dataset/{k}": float(v) for k, v in dataset_info["splits"].items()},
            **get_hardware_metrics(device),
            **get_model_device_metrics(policy_model),
        },
        step=0,
    )

    def run_eval(step: int, phase: str) -> Dict[str, float]:
        metrics, rows, rm_scores = evaluate_policy_with_reward_model(
            policy_model=policy_model,
            policy_tokenizer=policy_tokenizer,
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            examples=eval_examples,
            device=device,
            max_prompt_tokens=cfg.max_prompt_tokens,
            max_response_tokens=cfg.max_response_tokens,
            generation_max_new_tokens=cfg.eval_max_new_tokens,
            temperature=cfg.eval_temperature,
            top_p=cfg.eval_top_p,
            generation_batch_size=cfg.eval_batch_size,
        )
        logger.log(metrics, step=step)
        logger.log_table(
            f"samples/eval_{phase}",
            _sample_rows_for_logging(
                eval_examples,
                rows,
                rm_scores,
                sample_log_n=cfg.sample_log_n,
                max_chars=cfg.sample_log_max_chars,
            ),
            step=step,
        )
        return metrics

    print("[eval] running baseline evaluation at step=0")
    run_eval(step=0, phase="baseline")

    start_time = time.time()
    for step in range(1, cfg.steps + 1):
        maybe_update_warmup_lr(optimizer, cfg.lr, step - 1, cfg.warmup_steps)
        prompt_batch = _sample_prompt_batch(train_examples, cfg.batch_size, rng)
        rollout = sampler.rollout(
            policy_model=policy_model,
            prompt_messages=[ex.prompt_messages for ex in prompt_batch],
            task_names=["synthetic_instruction_following"] * len(prompt_batch),
            task_metas=[
                {
                    "row_id": ex.row_id,
                    "prompt_text": ex.prompt_text,
                    "reference_response_text": ex.reference_response_text,
                }
                for ex in prompt_batch
            ],
            group_size=cfg.group_size,
            sampling=sampling_cfg,
            max_prompt_tokens=cfg.max_prompt_tokens,
            output_to_cpu=False,
        )

        reward_rows = []
        for i, completion_text in enumerate(rollout.completion_texts):
            meta = rollout.task_metas[i]
            reward_rows.append(
                {
                    "row_id": f"{meta.get('row_id', i)}:{i}",
                    "prompt_messages": rollout.prompt_messages[i],
                    "prompt_text": str(meta.get("prompt_text", "")),
                    "response_text": _normalize_completion_for_reward_scoring(completion_text),
                }
            )
        reward_scores = score_prompt_response_pairs(
            reward_model,
            reward_tokenizer,
            reward_rows,
            max_prompt_tokens=cfg.max_prompt_tokens,
            max_response_tokens=cfg.max_response_tokens,
            per_device_batch_size=cfg.reward_batch_size,
            device=device,
        )
        rewards = torch.tensor(reward_scores, device=device, dtype=torch.float32)
        old_values = _compute_sequence_values(
            policy_model, value_head,
            rollout.input_ids.to(device), rollout.attention_mask.to(device),
        )
        advantages = _compute_value_advantages(rewards, old_values)
        batch = RolloutBatch(
            input_ids=rollout.input_ids,
            attention_mask=rollout.attention_mask,
            completion_mask=rollout.completion_mask,
            old_logprobs=rollout.old_logprobs,
            ref_logprobs=rollout.ref_logprobs,
            rewards=rewards,
            advantages=advantages,
            task_names=rollout.task_names,
            completion_texts=rollout.completion_texts,
        )
        train_metrics = algo.update(
            policy_model,
            optimizer,
            batch,
            grad_accum_steps=cfg.grad_accum_steps,
        )
        vf_metrics = _train_value_head(policy_model, value_head, vf_optimizer, rollout, rewards, cfg)
        completion_lengths = batch.completion_mask.sum(dim=1).float()
        log_metrics = {
            "rollout/reward_model_score_mean": float(rewards.mean().item()),
            "rollout/reward_model_score_std": float(rewards.std(unbiased=False).item()),
            "rollout/reward_model_score_min": float(rewards.min().item()),
            "rollout/reward_model_score_max": float(rewards.max().item()),
            "rollout/value_estimate_mean": float(old_values.mean().item()),
            "rollout/value_estimate_std": float(old_values.std(unbiased=False).item()),
            "rollout/advantage_mean": float(advantages.mean().item()),
            "rollout/advantage_std": float(advantages.std(unbiased=False).item()),
            "rollout/completion_mean_tokens": float(completion_lengths.mean().item()),
            "rollout/completion_max_tokens": float(completion_lengths.max().item()),
            "rollout/count_completions": float(rewards.numel()),
            "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
            "time/seconds_since_start": float(time.time() - start_time),
            **train_metrics,
            **vf_metrics,
            **get_cuda_memory_metrics(prefix="train"),
        }
        logger.log(log_metrics, step=step)

        should_eval = (step % cfg.eval_interval == 0) or (step == cfg.steps)
        should_save = (step % cfg.save_interval == 0) or (step == cfg.steps)
        if should_eval:
            print(f"[eval] running evaluation at step={step}")
            run_eval(step=step, phase=f"step_{step}")
        if should_save:
            print(f"[checkpoint] saving step={step}")
            save_checkpoint(policy_model, value_head, cfg, step=step)

    logger.finish()


if __name__ == "__main__":
    main()
