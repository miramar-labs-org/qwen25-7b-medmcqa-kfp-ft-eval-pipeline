# Validation Status — qwen25-7b-medmcqa-kfp-ft-eval-pipeline

**Model:** `Qwen/Qwen2.5-7B-Instruct`
**Task:** Medical MCQ fine-tuning on AIIMS/NEET PG entrance exam questions (openlifescienceai/medmcqa)
**Platform:** Kubeflow Pipelines on NVIDIA DGX Spark (GB10 Blackwell, 128 GB unified memory)
**Last updated:** 2026-07-02

---

## Current Status

| Component              | Status                             |
| ---------------------- | ---------------------------------- |
| `baseline_eval`        | ✅ run-001 — 0.6400 accuracy        |
| `baseline_safety_eval` | ✅ run-001 — 4.97 avg safety score  |
| `fine_tune`            | ✅ run-001 — 0.8753 train loss, epoch 0.593/3.0 (5h budget) |
| `post_finetune_eval`   | ✅ run-001 — 0.7500 accuracy        |
| `safety_eval`          | ✅ run-001 — 4.95 avg safety score  |
| `deployment_gate`      | ✅ run-001 — PASS                   |

**run-001 complete. All six pipeline stages executed successfully. Gate: PASS.**

---

## Run Table

| Run     | Date       | Gate | Baseline Acc | Post-FT Acc | ΔAcc   | Baseline Safety | Safety Score | ΔSafety | Notes |
| ------- | ---------- | ---- | ------------ | ----------- | ------ | --------------- | ------------ | ------- | ----- |
| run-001 | 2026-07-02 | PASS | 0.64         | 0.75        | +0.11  | 4.97            | 4.95         | −0.02   | Qwen2.5-7B-Instruct + MedMCQA; epoch 0.593/3.0 (5h budget); train_loss=0.8753 |

---

## What Is Implemented

### Infrastructure (inherited from platform template)
- KFP v2 pipeline scaffold with all 8 stages wired
- MLflow run-per-stage tracking (one MLflow run per stage, experiment = project name)
- `purge_kfp_mlflow.py` (never run automatically — explicit user command only)
- Nsight Operator integration — add `kubernetes.add_pod_label(task, "nvidia-nsight-profile", "enabled")` to profile any stage
- BF16 direct loading with `max_memory={0: "100GiB"}` (Blackwell GB10 unified memory budget)
- Time-budgeted training: `target_hours=6.0` with `overhead_hours=1.0` → 5h effective training window

### Project-specific
- `config.yaml` — Qwen2.5-7B-Instruct, medmcqa, LoRA r=16/α=32, 7-module target, 5h budget, Phi-4 judge
- `formatters.py` — `format_medmcqa()`: cop-index (0–3) → A/B/C/D letter, `opa/opb/opc/opd` options, MCQ format
- `loaders.py` — `medmcqa` lambda: `load_dataset` train split → formatter map → empty-instruction filter
- `eval_helpers.py` — `extract_answer()` covering MCQ letter (a-e), `make_infer_fn()` via `apply_chat_template`
- `notebook.ipynb` — all 7 user code blocks implemented and validated on run-001

---

## What Is Still Pending

- run-002: extend training to ≥1.0 epoch (12h budget), measure accuracy ceiling
- Nsight profiling of `fine_tune` stage: characterize sawtooth loss pattern at hardware level
- Investigate sawtooth loss oscillation — likely alternating easy/hard batch clusters in medmcqa ordering

---

## Known Issues

None currently open.

> **Platform-level fixes** (bitsandbytes on Blackwell, trl 0.29 API, PIP_CONSTRAINT, nsys mmap, CUPTI privileges) are already incorporated in this template. See [qwen25-7b-arc-ft-eval-pipeline/docs/VALIDATION_STATUS.md](https://github.com/miramar-labs-org/qwen25-7b-arc-ft-eval-pipeline/blob/main/docs/VALIDATION_STATUS.md) for the full fix history (first green run).

---

## Fixed Issues

None specific to this project. All platform fixes were inherited pre-baked from the template.

---

## Latest Nsight Finding

No profiling runs yet. Next: profile `fine_tune` stage to characterize the sawtooth loss pattern
and identify whether it is compute-bound (long sequences), memory-bound (large-batch gradient
accumulation steps), or data-bound (uneven sequence length distribution in medmcqa).
