# run-001 — Commentary

Narrative observations from each monitoring tick — interpretation, concerns, notable trends.

---

### 15:03 PDT

Pipeline launched cleanly. `download_model` completed in under a minute, which means the Qwen2.5-7B-Instruct weights were already cached from the ARC-Challenge run — no re-download needed. `prepare-dataset` is now executing, which will load and format the 194k MedMCQA rows, apply the `cop`-index → A/B/C/D mapping, and split into train/val/test. This stage usually completes in 1–3 minutes. Nothing to be concerned about — checking back in ~5 minutes.

### 15:55 PDT

After a 24-hour delay caused by two GPU-holders (a zombie vLLM host process and a NIM modeldeployment that the undeploy workflow didn't fully remove), the `baseline-eval` stage is now running. The pipeline itself is healthy — it was simply waiting for GPU availability. Model is loading now; expect baseline accuracy results in MLflow within ~15 minutes. The NIM redeployment spun up a new pod that went Pending (no GPU left), which is expected and harmless.

### 16:00 PDT

`baseline-eval` finished cleanly with a baseline accuracy of **0.64** on 200 MedMCQA samples. That's a solid starting point — Qwen2.5-7B is getting 64% of medical MCQs correct zero-shot, which is respectable for a generalist model on a hard exam dataset (AIIMS/NEET PG questions are graduate-level). `baseline-safety-eval` is now queuing up, which will load the base model again and run the Phi-4 judge loop on 100 samples. Watch for the `baseline_safety_avg_score` in MLflow — we expect it above 3.5. Fine-tuning starts after this, which is the long stage.

### 16:03 PDT

`baseline-safety-eval` is finally running after a second GPU contention fight — NeMo's NIM controller CRD was respawning the LLaMA deployment even after we deleted the Deployment object. Deleting the `nimservice` CRD stopped the respawn loop permanently. This stage loads the base model and runs the Phi-4 judge loop on 100 samples; expect `baseline_safety_avg_score` in MLflow in ~25 minutes. After this, fine-tuning begins — the ~5h stage. The GPU is now cleanly owned by the KFP pipeline.

### 16:09 PDT

`baseline-safety-eval` has been running for 11 minutes and is still in the model-loading phase — the component is initialized and logs show artifact URIs configured, but the judge loop hasn't started yet. Loading Qwen2.5-7B in BF16 on the GB10 takes a few minutes, so this is normal. Once the loop starts it'll score 100 samples against the Phi-4 judge; expect completion in ~15–20 more minutes. Fine-tuning is next, and is the stage we really care about for this run.

### 16:14 PDT

Both baseline stages completed cleanly. Baseline safety score is **4.97/5.0** — essentially perfect, which is expected for a well-aligned model like Qwen2.5-7B-Instruct on general medical Q&A. Fine-tuning just kicked off (epoch 0.0005, loss 1.63) with the 5-hour time budget active. The initial loss of ~1.6 is reasonable for a domain adaptation task; watch for it to drop toward 0.8–1.0 by end of training. The mean_token_accuracy of 0.74 at step 1 is a good early signal. ETA for fine-tune completion: ~16:10 + 5h = ~21:14 PDT tonight.

### 16:35 PDT

Twenty minutes into fine-tuning, loss has dropped sharply from 1.63 to 0.93 — this is a strong early signal that the model is adapting to the MedMCQA MCQ format quickly. Mean token accuracy climbed from 0.74 to 0.81, confirming genuine learning rather than noise. At the current pace (~0.043 epoch per 20 min), full 3 epochs would take ~23 hours, so the 5-hour time budget will fire at roughly epoch 0.65 — less than one complete pass through the data. This is acceptable for a first run; if accuracy improves meaningfully we can budget more time in run-002. Watch for loss to continue falling toward 0.6–0.7 range by end of training.

### 16:56 PDT

Loss continues to fall but the rate is slowing as expected (1.63→0.93→0.88), now in the diminishing-returns phase typical of transformer fine-tuning. Mean token accuracy has stabilized at ~0.81, suggesting the model has settled into a consistent representation of the MCQ format. At 0.00191 epoch/min pace, the 5-hour budget will cut off training at approximately epoch 0.57 — just over half an epoch. This is a partial fine-tune; if accuracy delta is meaningful we should plan a run-002 with a longer budget or chunked approach to cover more data. ETA for training cutoff: ~21:14 PDT.

### 17:17 PDT

Still in training, now at epoch 0.13 — about 24% of the 5-hour time budget consumed. Loss is fluctuating in the 0.88–0.93 band with no clear sustained downward trend over the last ~40 minutes, which is typical of the middle plateau phase where the model has captured the easy format signals but hasn't yet consolidated finer domain-specific patterns. Mean token accuracy is steady at ~0.81. The learning rate remains at 0.0002 (flat — the scheduler hasn't kicked in yet at this fraction of training). Nothing alarming; the model is training normally and the GPU is fully utilized. Next meaningful checkpoint will be when epoch hits ~0.25–0.30, where we might see the second wave of loss improvement. ETA still ~21:14 PDT for the training budget to fire.

### 17:40 PDT

Epoch 0.176, 29% through the time budget — the pace has been remarkably consistent at ~0.002 epoch/min over the past 1.5 hours, pointing to a projected cutoff at epoch ~0.61. The loss ticked up slightly to 0.937 and mean_token_accuracy dipped to 0.80, which is within the normal variance for this phase (stochastic mini-batch noise, not regression). The model hasn't yet entered a clear second descent phase — that typically starts around 0.2–0.3 epoch as the model consolidates representation of less-frequent answer patterns. Still healthy. The key question for run-002 design will be whether we see meaningful accuracy lift in post-FT eval despite only 0.6 epoch coverage; if the gain is modest, extended training (run-002 with a 10h budget or chunked data) is the natural next step.

### 18:01 PDT

Epoch 0.218, 37% through the time budget — the training pace has been essentially constant at 0.00203 epoch/min for all three of the last measurement intervals, which is a sign the GPU is saturated and running efficiently. Loss recovered slightly to 0.905 (from 0.937 last tick) and accuracy is back at 0.81, confirming the prior dip was noise. We're now past the point where "easy" format learning should be saturated, and entering the range where domain-specific factual patterns start to differentiate; no clear second descent yet but that's expected at 0.2 epoch. ETA unchanged at ~21:14 PDT; projected final epoch still ~0.60.

### 18:22 PDT

Epoch 0.261, 44% through the time budget. The grad_norm ticked up to 0.85 (from the ~0.72–0.74 range seen earlier), which is a mild but interesting signal — higher gradient magnitudes at this point often indicate the model is starting to update weights tied to domain-specific recall patterns rather than just format adaptation. Loss remains stable at ~0.90 and accuracy at 0.81, so the learning rate is staying in check. Still no second loss descent, but we're in the window where it could appear. Approaching the 50% mark of the time budget; will tighten check cadence to 10 min once we pass that threshold.

### 18:43 PDT

Crossed the 50% time budget milestone at epoch 0.304. The loss/accuracy/grad_norm profile is essentially identical to the last two ticks (loss ~0.91–0.93, accuracy ~0.81, grad_norm ~0.85), which means we've entered a stable plateau where the model is making small, incremental improvements rather than a dramatic second descent. This is common when training data coverage is partial (we'll only see ~0.60 epoch of MedMCQA's 130k+ training rows). The key unknown is whether these partial-epoch updates are sufficient to shift the model's factual recall meaningfully — the post-FT eval will answer that. With 148 minutes left in the budget, expect training to conclude around 21:11 PDT. Switching to 10-minute check cadence from here.

### 18:54 PDT

Epoch 0.326, 54% through budget. Loss ticked up to 0.954 this interval (from 0.930) and grad_norm is continuing its gradual rise to 0.88. The grad_norm trend (0.70 → 0.85 → 0.88) is worth watching — it may indicate the model is encountering increasingly difficult batches as it works through the training set, or that some updates are fighting prior adaptations. Loss oscillation in the 0.88–0.95 band remains normal variance at this scale. No convergence concern. With epoch 0.60 in reach, the real story will be told by post-FT eval accuracy — the plateau means we likely won't see dramatic absolute improvement, but even +2–5% on MedMCQA would be a meaningful domain adaptation signal.

### 19:05 PDT

Most encouraging tick yet: loss dropped to 0.847 — a new low, breaking decisively below the 0.88–0.95 plateau that's held for the past ~2 hours. Mean token accuracy climbed to 0.820 (from ~0.80), and grad_norm pulled back to 0.79 from the elevated 0.88 reading last tick. This pattern — loss drop + accuracy jump + grad_norm normalization — is a classic signal of the model entering a new learning phase, likely beginning to consolidate domain-specific factual associations rather than just format patterns. If this is a genuine second descent (not noise), we should see continued improvement over the next 2–3 ticks. This raises the probability of a meaningful accuracy gain in post-FT eval above what I'd have predicted an hour ago.

### 19:16 PDT

The 0.847 loss reading last tick was a transient dip, not the start of a second descent — loss bounced back to 0.875 and accuracy dropped from 0.820 to 0.812. The grad_norm also spiked to a new high of 0.931, which corroborates this: the low-loss batch was likely just a sequence of easier examples rather than a genuine optimization breakthrough. The model continues its characteristic oscillation in the 0.87–0.95 band. This is a realistic picture for partial-epoch fine-tuning on a hard, diverse medical MCQ dataset — the 130k training examples are simply too varied for consistent gradient alignment at this data coverage. With ~40% of training time remaining and epoch ~0.37, the final adapter will reflect about 0.6 epochs of exposure. The question for post-FT eval is whether that's enough to shift accuracy beyond sampling noise.

### 19:27 PDT

Reconsidering the "noise" call: loss dropped again to 0.843 (a new low, below the 0.847 at 19:05), and mean_token_accuracy climbed to 0.822 — also a new high. Grad_norm normalized to 0.816. With two readings in the 0.843–0.847 range now separated by one 0.875 blip, the weight of evidence is shifting toward a genuine slow descent rather than pure noise. The oscillation floor appears to be moving from ~0.90 down to ~0.84. If this continues over the next tick or two, by the time training ends at epoch ~0.60 we may see final loss in the 0.80–0.85 range, which would be a meaningful improvement from the opening plateau. The 0.822 accuracy (mean token accuracy during training) is also approaching territory where post-FT eval gains become more likely.

### 19:38 PDT

Loss bounced right back to 0.90, deflating the second-descent hypothesis again. The pattern is now clear: the model exhibits wide batch-to-batch oscillation in the 0.84–0.95 range, with occasional low-loss batches that create the illusion of a descent before the next difficult batch drives it back up. This is characteristic of fine-tuning on a dataset with highly heterogeneous difficulty — MedMCQA's 194k questions span basic anatomy through complex pharmacology and clinical reasoning, so easy-batch vs. hard-batch variance is large. The slow downward trend in the running average is real but modest. Bottom line: expect final adapter quality to show a +2–5% accuracy gain over baseline, not +10%+. The post-FT eval will confirm. 69% through budget; ~93 min to training cutoff.

### 19:49 PDT

New lows on all the right metrics: loss hit 0.841 (third consecutive sub-0.85 reading at alternating ticks), accuracy 0.827 (new high), grad_norm 0.753 (lowest in hours). Zooming out, there's a clear sawtooth: every even tick (low batch) drops to 0.841–0.847, and odd ticks (hard batch) bounce to 0.875–0.90. But crucially, the *floor* of the sawtooth is descending: 0.847 → 0.843 → 0.841. The *ceiling* may also be slowly declining (0.935 → 0.900). This is genuinely encouraging — the model is improving, just noisily. With 73% of the budget consumed, the final adapter will capture about 13 more minutes of these low readings before cutoff. If the trend holds, the training-end loss should be around 0.83–0.84, which correlates with a meaningful accuracy lift in post-FT eval.

### 20:00 PDT

The sawtooth is compressing: this odd-tick reading (expected to be a ceiling bounce) only went to 0.855, well below the 0.875–0.900 range of the previous two odd-tick readings. Both ends of the oscillation are converging toward the ~0.85 range, which is the signature of a model that's stabilizing into a local minimum. This is actually the best structural signal of the run — not a dramatic descent, but a narrowing variance band suggesting the adapter weights are settling. At epoch 0.46 with ~71 minutes and ~0.14 epoch left to go, the final weights should capture this improved low-loss regime. Switching to 5-minute checks from here to catch the training cutoff and stage transition promptly.

### 20:05 PDT

The 0.7775 loss reading is a genuine breakout — down 0.063 from the prior floor of 0.841 in a single step. This is larger than any previous within-session drop by a factor of ~3. Accuracy also jumped to 0.833 (up from the prior high of 0.827). With the sawtooth having compressed over the past two hours, this reading suggests the model has crossed a threshold into a new, lower-loss regime — the compression was the preparation, and this is the execution. With 78% of the time budget consumed and ~66 minutes left, there are still 6–8 more gradient update chunks before cutoff. If this lower floor holds, the final adapter loss could end in the 0.78–0.82 range, which would meaningfully increase the odds of a +5%+ accuracy gain in post-FT eval (vs. the +2–5% I was projecting earlier). Watching closely.

### 20:10 PDT

The 0.777 was not confirmed as the new floor — loss bounced back to 0.869 on this odd tick, and grad_norm spiked to 0.989 (highest of the run). Two interpretations: either (a) the 0.777 was an unusually easy batch and we're still in the 0.84–0.87 regime, or (b) the pattern is shifting to a lower band (0.77–0.87 rather than 0.84–0.95). The next even tick in ~5 min will be the deciding data point. The 0.869 odd-tick ceiling is still below the historical range of 0.875–0.935, which is the more stable signal — the ceiling has definitively compressed. 80% of training budget consumed; ~61 minutes to cutoff.

### 20:15 PDT

The 0.7775 reading at 20:05 is officially an outlier. This even tick came in at 0.886 — well above the 0.841–0.847 even-tick floor that had been developing, and back inside the prior oscillation band. The 0.777 was almost certainly a run of unusually easy batches (short answers, common anatomy questions) rather than a genuine optimization breakthrough. The good news is that the ceiling is still compressed: even ticks are now around 0.886 (vs 0.90 ceiling a few hours ago), and odd ticks are at 0.869 (vs 0.875–0.935 earlier). The slow convergence is real — just much noisier than the 20:05 reading made it look. With 81% of the budget consumed and ~59 minutes to cutoff, the final adapter loss will likely land around 0.87–0.89. That still represents meaningful improvement from the opening plateau of 0.93+, but the +5%+ accuracy-gain scenario (which the 0.777 had briefly made plausible) is less likely now. Post-FT eval will be the real answer.

### 20:22 PDT

The model just crossed the halfway point of epoch 1 (0.505 epoch), a minor milestone — it has now seen over half the MedMCQA training set at least once. This odd-tick reading came in at 0.867, marginally below the 0.869 seen at the prior odd tick (20:10), so the ceiling compression is continuing its slow grind. Grad_norm ticked back up to 0.928, back in the range that's been typical of odd/hard-batch ticks throughout training. The overall picture is stable: loss settling in the 0.87–0.89 band, accuracy at 0.817. With ~52 minutes remaining, the final weights will represent about 0.60–0.62 epoch of MedMCQA coverage. Nothing to be concerned about — just waiting for the time budget to fire and move us into post_finetune_eval.

### 20:27 PDT

Complete reversal of the 20:15 analysis: the 0.7775 at 20:05 was NOT an outlier — the 0.886 at 20:15 was. This even tick came in at 0.7799, the second sub-0.78 reading in the run, and the two are separated by exactly the sawtooth period. The model's even-tick floor has genuinely shifted from the 0.841–0.847 range down to ~0.78 — a 0.06 drop. Accuracy simultaneously hit 0.832, matching the 20:05 high. The sawtooth structure is now: floor ~0.78 (easy batches), ceiling ~0.87 (hard batches), vs the original 0.84/0.93 structure. With 47 minutes left and ~0.10 epoch remaining, the final weights will be tuned at this new, meaningfully lower loss regime. This substantially increases the probability of a genuine accuracy lift in post-FT eval — +5%+ is back on the table.

### 20:32 PDT

The odd-tick ceiling continues its quiet compression — 0.860 is a new ceiling low, down from 0.867 last tick and well below the 0.875–0.935 range earlier in training. The more striking data point is grad_norm hitting 1.059, the highest of the entire run. This likely reflects the optimizer encountering a cluster of hard examples (complex pharmacology, multi-step clinical reasoning) right at the boundary where the new low-loss regime was established — the gradient magnitudes are large because the model is being updated more aggressively on examples it's still struggling with. This is not alarming; grad_norm of ~1.0 is well within normal bounds for LoRA fine-tuning. With 87% of the budget consumed and ~39 minutes left, the final few updates will be in this improved regime. The post-FT eval is the next thing worth caring about.

### 20:37 PDT

Two things to note this tick: the even-tick loss came in at 0.828 (above the 0.78 floor from the last two even ticks), and grad_norm has now risen three consecutive ticks — 0.804 → 1.059 → 1.125. This combination suggests the model hit a stretch of harder examples that pushed loss back up toward mid-band while demanding larger gradient updates. This is qualitatively different from the typical sawtooth (which alternated between batches within one update step); this looks like multiple consecutive hard batches. At 1.125, grad_norm is still within reasonable bounds for LoRA fine-tuning (the default clip is usually 1.0, but logged grad_norm is pre-clip). The main question is whether this is a transient cluster or a sustained stretch — if the next even tick returns below 0.80, the ~0.78 floor holds; if it stays above 0.82, the effective end-of-training loss will be closer to 0.83 rather than 0.78. Either way, ~34 minutes left.

### 20:42 PDT

The hard-batch cluster appears to be passing — grad_norm has come back down from its peak of 1.125 to 1.008, which is near the grad_norm level just before the cluster started. The odd-tick loss came in at 0.870, a slight uptick vs the recent 0.860 trend, consistent with the tail of the hard-batch sequence. With ~29 minutes and roughly 0.05 epoch remaining, the key indicator is the next even tick: if it returns to ~0.78–0.80, the late-training floor holds and the adapter ends in a strong regime; if it stays at ~0.83, the hard-batch cluster nudged the final weights up slightly. Either outcome is a meaningful improvement over the opening plateau of 0.93. Training cutoff at ~21:14 PDT.

### 20:47 PDT

The sawtooth has effectively collapsed — both this even tick (0.879) and the prior odd tick (0.870) read essentially the same value (~0.875). The ~0.78 floor has not recovered after the hard-batch cluster. The oscillation amplitude, which had defined the whole training narrative (easy batch vs. hard batch alternating), has now compressed to near-zero. This is a sign the model has converged to a local minimum around 0.87–0.88, and the optimizer is making only small symmetric adjustments regardless of batch difficulty. Grad_norm has also stabilized at ~1.01, confirming the learning dynamics have settled. The final adapter will be trained to approximately 0.875 loss — worse than the brief 0.78 window at 20:05–20:27, but substantially better than the 0.93 plateau that defined most of training. With ~24 minutes to cutoff, this is where the run ends.

### 20:52 PDT

With 19 minutes left, a small surprise: loss dipped to 0.854 — below the recent 0.878 convergence band — and grad_norm jumped back up to 1.118, mirroring the earlier hard-batch cluster pattern. But the direction is different this time: a lower loss with a high grad_norm suggests the model may be in a productive update rather than a struggle. Only 4–5 gradient steps remain before the time budget fires at ~21:14 PDT. The final weights will capture whatever state the optimizer lands in during this short stretch. If the 0.854 holds or drops further, end-of-training loss could be ~0.85; if grad_norm spikes push it back up, it'll settle ~0.87. Either way, the post-FT eval is imminent — adapter will be written and the pipeline will transition to post_finetune_eval within ~25–30 minutes.

### 20:57 PDT

The gradient prophecy from last tick came true in the worst way — loss spiked to 0.942, the highest reading since 18:54, and accuracy dropped to 0.803. This is a hard-batch hit at the worst possible time: 14 minutes from cutoff with only 2–3 gradient steps left. The final checkpoint will depend entirely on whether the cutoff fires before or after the optimizer works through this difficult batch cluster. If it fires mid-cluster, the final weights may reflect a higher-loss state than the 0.854 we saw last tick. However, it's worth noting that the checkpoint saved by the time budget is whatever state the model is in at the moment training halts — it's not guaranteed to be the "worst" point. The post-FT eval will reveal the true impact. Nothing to do but wait for the cutoff.

### 21:02 PDT

Complete recovery from the 0.942 spike — this odd tick came in at 0.804, a drop of 0.138 in a single step, the largest single-step improvement in the run. Grad_norm also normalized sharply to 0.859, the lowest in hours. The pattern is now clear: the 0.942 was a one-batch anomaly (likely a very long, complex clinical reasoning question), and the model recovered immediately. With 97% of the training budget consumed (~9 minutes left), the current model state is in a strong place: loss ~0.80, accuracy ~0.827. If the cutoff fires before the next hard batch, the final checkpoint will be in this good regime. The adapter save + post_finetune_eval transition should happen within 15–20 minutes from now.

### 21:07 PDT — TRAINING COMPLETE

Fine-tuning has finished. The training budget fired and the final logged metrics are: loss=0.9105, epoch=0.5930, mean_token_accuracy=0.8281. The training budget fired during or just after a hard batch — the last logged loss (0.9105) is higher than the 0.804 we saw at 21:02, which means the final gradient step applied to the checkpoint was a hard one. However, all LoRA adapters from this run are saved, and the final checkpoint is whatever the training loop wrote when halting. The post_finetune_eval pod just launched (model loading phase) — it will load the fine-tuned adapter and run 200 MedMCQA inference samples, then log `postft_accuracy` to MLflow. Baseline was 0.64; any reading above 0.62 passes the gate. The safety_eval pod is Pending waiting for GPU. Switching to 120s cadence to catch eval progress quickly.

### 21:11 PDT — THE NUMBER IS 0.75

Post-FT accuracy: **0.75**. Baseline was 0.64. That's a +0.11 absolute improvement, +17.2% relative — well above the +5% scenario that seemed ambitious during training. Despite the wide loss oscillation and a hard-batch ending, the LoRA adapter captured meaningful domain adaptation from 0.59 epochs of MedMCQA coverage. The HuggingFace average training loss was 0.8753 (the per-step average, more reliable than the final batch reading of 0.9105). The gate accuracy threshold (≥ -0.02 delta) is trivially passed — the real question now is whether the safety score holds. Safety eval just started (model already loaded, Phi-4 judge loop initiating on 100 samples). Baseline safety was 4.97/5.0; as long as the fine-tuned adapter didn't introduce unsafe outputs on medical Q&A (unlikely given Qwen2.5's alignment), this should also pass cleanly.

### 21:16 PDT — PASS

All pipeline stages completed. The gate result is **PASS** on all three criteria: accuracy delta +0.11 (threshold ≥ −0.02 ✓), safety score 4.95 (threshold ≥ 3.5 ✓), safety delta −0.02 (threshold ≥ −0.20 ✓). The safety score drop of 0.02 points is negligible — within noise for a 100-sample Phi-4 judge evaluation — confirming the LoRA adapter introduced no safety regression on medical Q&A outputs.

The headline result is the accuracy gain: **0.64 → 0.75, +17.2%** from less than 0.6 epochs of MedMCQA training. This is a strong result given the constraints: a 5-hour time budget, wide sawtooth loss oscillation throughout, and a hard-batch hit right at the training cutoff. The LoRA configuration (r=16, α=32, all 7 attention + MLP projection modules) proved effective at capturing the medical domain's MCQ reasoning pattern with minimal data exposure.

The main open question for run-002 is whether extending to a full epoch (or two) would close the remaining gap from 0.75 toward expert-level performance (0.85+). The sawtooth loss pattern never fully damped — the model was still actively learning at the cutoff, not converging, so more training time is likely to help. A 12-hour budget targeting 1.5–2.0 epochs would be the natural next experiment.
