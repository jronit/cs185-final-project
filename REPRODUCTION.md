# Reproduction Commands

All training was run on Modal with H100 GPUs. The dataset lives on the Modal volume at
`/vol/synthetic_datasets/wildchat_min4_judged_5k_v1`. Submissions are built with
`build_policy_submission_remote` and downloaded with `modal volume get`.

The public evaluation prompts file used for **all** policy submissions is:
`/root/project/public_eval/public_test_gen_prompts_128.jsonl` (128 prompts, inside the
project image — not the volume).

---

## Reward Model

```bash
uv run modal run scripts/modal_train.py::reward_model_train_remote -- \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --output_dir runs/wildchat_min4_judged_5k_reward_model_v1 \
  --num_train_epochs 3.0 \
  --lr 3e-5 \
  --per_device_train_batch_size 8 \
  --lora_r 32 --lora_alpha 64 \
  --save_interval 50 \
  --wandb_name wildchat_min4_judged_5k_reward_model_v1
```

**Submitted checkpoint:** `step_000445` (final, 3 epochs)
**Reward adapter path (used by all online methods):**
`/vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter`

---

## Part 1: Offline Methods

### DPO (β = 0.1)

```bash
uv run modal run scripts/modal_train.py::train_remote -- \
  --algo dpo --beta 0.1 \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --output_dir runs/wildchat_min4_judged_5k_dpo_beta01_v1 \
  --num_train_epochs 3.0 \
  --lr 5e-5 \
  --lora_r 32 --lora_alpha 64 \
  --save_interval 100 \
  --wandb_name wildchat_min4_judged_5k_dpo_beta01_v1
```

**Submitted checkpoint:** `step_000100`

### IPO (β = 0.1)

```bash
uv run modal run scripts/modal_train.py::train_remote -- \
  --algo ipo --beta 0.1 \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --output_dir runs/wildchat_min4_judged_5k_ipo_v1 \
  --num_train_epochs 3.0 \
  --lr 5e-5 \
  --lora_r 32 --lora_alpha 64 \
  --save_interval 100 \
  --wandb_name wildchat_min4_judged_5k_ipo_v1
```

**Submitted checkpoint:** `step_000300`

### AOT (β = 0.2)

```bash
uv run modal run scripts/modal_train.py::train_remote -- \
  --algo aot --beta 0.2 \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --output_dir runs/wildchat_min4_judged_5k_aot_beta02_v1 \
  --num_train_epochs 3.0 \
  --lr 5e-5 \
  --lora_r 32 --lora_alpha 64 \
  --save_interval 50 \
  --wandb_name wildchat_min4_judged_5k_aot_beta02_v1
```

**Submitted checkpoint:** `step_000550`

---

## Part 1: Online Methods

All online methods use the reward model adapter at:
`/vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter`

All online methods were trained for **25 steps** with group size 4, batch size 16.

### GRPO

```bash
uv run modal run scripts/modal_train.py::rm_grpo_train_remote -- \
  --algo grpo \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter \
  --output_dir runs/wildchat_min4_judged_5k_grpo_rm445_v1 \
  --steps 25 --batch_size 16 --group_size 4 \
  --lr 1e-5 --kl_coef 0.01 --clip_eps 0.2 \
  --wandb_name wildchat_min4_judged_5k_grpo_rm445_v1
```

**Submitted checkpoint:** `step_000025`

### DrGRPO

```bash
uv run modal run scripts/modal_train.py::rm_grpo_train_remote -- \
  --algo dr_grpo \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter \
  --output_dir runs/wildchat_min4_judged_5k_drgrpo_rm445_v1 \
  --steps 25 --batch_size 16 --group_size 4 \
  --lr 1e-5 --kl_coef 0.01 --clip_eps 0.2 \
  --wandb_name wildchat_min4_judged_5k_drgrpo_rm445_v1
```

**Submitted checkpoint:** `step_000025`

### GSPO

```bash
uv run modal run scripts/modal_train.py::rm_grpo_train_remote -- \
  --algo gspo \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter \
  --output_dir runs/wildchat_min4_judged_5k_gspo_rm445_v1 \
  --steps 25 --batch_size 16 --group_size 4 \
  --lr 1e-5 --kl_coef 0.01 --clip_eps 0.2 \
  --wandb_name wildchat_min4_judged_5k_gspo_rm445_v1
```

**Submitted checkpoint:** `step_000025`

---

## Part 2: Experiments

### PPO with Learned Value Baseline

```bash
uv run modal run scripts/modal_train.py::rm_ppo_train_remote -- \
  --algo ppo \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter \
  --output_dir runs/wildchat_min4_judged_5k_ppo_rm445_v1 \
  --steps 100 --batch_size 16 --group_size 4 \
  --lr 1e-5 --kl_coef 0.01 --clip_eps 0.2 \
  --vf_lr 1e-4 --vf_clip_eps 0.2 --warmup_steps 20 \
  --wandb_name wildchat_min4_judged_5k_ppo_rm445_v1
```

**Result:** ~66% win rate. Best checkpoint: `step_000025` (evaluated at intervals).

### Extended DrGRPO — 100 Steps (Reward Hacking Ablation)

```bash
uv run modal run scripts/modal_train.py::rm_grpo_train_remote -- \
  --algo dr_grpo \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter \
  --output_dir runs/wildchat_min4_judged_5k_drgrpo_rm445_100steps_v2 \
  --steps 100 --batch_size 8 --group_size 4 \
  --lr 3e-5 \
  --wandb_name wildchat_min4_judged_5k_drgrpo_rm445_100steps_v2
```

**Result:** ~5% win rate at step 100 (catastrophic reward hacking collapse).

### Confidence-Weighted DPO (DPO-Conf) — Part 2 Offline Best

```bash
uv run modal run scripts/modal_train.py::train_remote -- \
  --algo dpo_conf --beta 0.1 \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --output_dir runs/wildchat_min4_judged_5k_dpo_conf_v1 \
  --num_train_epochs 3.0 \
  --lr 5e-5 \
  --lora_r 32 --lora_alpha 64 \
  --save_interval 100 \
  --wandb_name wildchat_min4_judged_5k_dpo_conf_v1
```

**Submitted checkpoint:** `step_000100`
**Result:** ~79% win rate, beating vanilla DPO (76%).

---

## Building Policy Submissions

Replace `<adapter_path>` with the volume-relative path to the adapter directory.

```bash
uv run modal run scripts/modal_train.py::build_policy_submission_remote -- \
  --adapter_path <adapter_path> \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl <output_path_on_volume>
```

**Example — DPO step 100:**
```bash
uv run modal run scripts/modal_train.py::build_policy_submission_remote -- \
  --adapter_path runs/wildchat_min4_judged_5k_dpo_beta01_v1/checkpoints/step_000100/adapter \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl runs/wildchat_min4_judged_5k_dpo_beta01_v1/submissions/step_000100.jsonl
```

**Example — DPO-Conf step 100:**
```bash
uv run modal run scripts/modal_train.py::build_policy_submission_remote -- \
  --adapter_path runs/wildchat_min4_judged_5k_dpo_conf_v1/checkpoints/step_000100/adapter \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl runs/wildchat_min4_judged_5k_dpo_conf_v1/submissions/step_000100.jsonl
```

---

## Building the Reward Model Submission

```bash
uv run modal run scripts/modal_train.py::build_reward_model_submission_remote -- \
  --reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter \
  --prefs_jsonl /vol/synthetic_datasets/wildchat_min4_judged_5k_v1/test_prefs.jsonl \
  --output_jsonl runs/wildchat_min4_judged_5k_reward_model_v1/submissions/public_test_pref_scores.jsonl
```

---

## Downloading Submission Files from Volume

```bash
uv run modal volume get llm-rl-final-project-volume \
  /runs/<run_name>/submissions/<file>.jsonl \
  <local_destination>.jsonl --force
```

---

## Running the Local Autograder

```bash
OPENAI_API_KEY=<your_key> uv run python student_autograder/run_local_autograder.py \
  --submission_dir llm_rl_final_proj_public_submission
```

The submission directory must contain:
```
llm_rl_final_proj_public_submission/
  policy_generations/{dpo,ipo,aot,grpo,drgrpo,gspo}.jsonl
  reward_model/public_test_pref_scores.jsonl
  part2/offline_best.jsonl
  part2/online_best.jsonl
```

---

## Submitted Checkpoint Summary

| Method | Run Directory | Checkpoint | Win Rate |
|--------|--------------|------------|----------|
| Reward Model | `wildchat_min4_judged_5k_reward_model_v1` | `step_000445` | 82.4% pair acc |
| DPO | `wildchat_min4_judged_5k_dpo_beta01_v1` | `step_000100` | 76% |
| IPO | `wildchat_min4_judged_5k_ipo_v1` | `step_000300` | 71% |
| AOT | `wildchat_min4_judged_5k_aot_beta02_v1` | `step_000550` | 68% |
| GRPO | `wildchat_min4_judged_5k_grpo_rm445_v1` | `step_000025` | 75% |
| DrGRPO | `wildchat_min4_judged_5k_drgrpo_rm445_v1` | `step_000025` | 73% |
| GSPO | `wildchat_min4_judged_5k_gspo_rm445_v1` | `step_000025` | 63% |
| PPO (Part 2) | `wildchat_min4_judged_5k_ppo_rm445_v1` | `step_000025` | ~66% |
| DrGRPO 100-step (Part 2) | `wildchat_min4_judged_5k_drgrpo_rm445_100steps_v2` | `step_000100` | ~5% |
| DPO-Conf (Part 2) | `wildchat_min4_judged_5k_dpo_conf_v1` | `step_000100` | ~79% |
