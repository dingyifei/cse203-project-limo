# CK-5 Codex Review — LIMO 2G76 Phase 4 Decoder Finetune

> Independent code review by `codex:codex-rescue`, run 2026-05-29.
> Codex sandbox was read-only; this file was hand-saved by team-lead from Codex's response.

## Verdict

**PASS** — all seven requested checks are satisfied by the cited implementation/config lines below; no BLOCKER or WARNING findings.

## Critical Issues

1. **[CHECK 1 — ENCODER + EMBEDDING FROZEN]** PASS. Encoder and embedding params are set `requires_grad = False` at `limo/vae_finetune.py:269-272`; only decoder params are re-enabled at `limo/vae_finetune.py:273-274`. The optimizer is built only from `p.requires_grad` params at `limo/vae_finetune.py:328-329`, and the sole training `optimizer.step()` is at `limo/vae_finetune.py:399` after a no-grad encoder/embedding pass at `limo/vae_finetune.py:370-376`. Severity: **INFO**.

2. **[CHECK 2 — MPS BATCHNORM IN EVAL MODE]** PASS. Each epoch calls `vae.train()` at `limo/vae_finetune.py:349` and immediately reasserts `vae.encoder.eval()` at `limo/vae_finetune.py:352`; encoder forward is wrapped in `with torch.no_grad():` at `limo/vae_finetune.py:370-376`. Severity: **INFO**.

3. **[CHECK 3 — VALIDITY GATE RESTORATION]** PASS. Baseline model and Adam states are saved at `limo/vae_finetune.py:336-339`; failed validity restores both via `vae.load_state_dict(prev_state)` and `optimizer.load_state_dict(prev_opt)` at `limo/vae_finetune.py:433-434`; successful epochs refresh both snapshots at `limo/vae_finetune.py:483-485`. Severity: **INFO**.

4. **[CHECK 4 — MAINTENANCE-SET WEIGHT FIX]** PASS. The full tensor mean is computed before quartile slicing: `full_w_mean = float(w.mean().item())` at `limo/vae_finetune.py:144`, while the quartile slice is only formed later at `limo/vae_finetune.py:147-150`; maintenance weight uses `maint_per = maint_w_mult * full_w_mean` at `limo/vae_finetune.py:300`, with multiplier `0.1` at `configs/finetune/04_decoder_init.yaml:22`. (This is the dba33a5 fix Codex's stop-hook caught earlier.) Severity: **INFO**.

5. **[CHECK 5 — LOSS NORMALIZATION SCALE]** PASS. Finetune divides per-sample KLD by `vocab_len` at `limo/vae_finetune.py:390-394` and averages the weighted samples at `limo/vae_finetune.py:395`; pretraining divides by `(batch_size * pred.shape[1])` at `limo/models.py:54`, with `pred` passed as `out.reshape((-1, self.vocab_len))` at `limo/models.py:60`. The denominators match the scale `B * vocab_len`. Severity: **INFO**.

6. **[CHECK 6 — WEIGHTEDRANDOMSAMPLER REPLACEMENT]** PASS. The sampler is `WeightedRandomSampler(weights=w_all.tolist(), num_samples=len(X_all), replacement=True)` at `limo/vae_finetune.py:313-315`. Severity: **INFO**.

7. **[CHECK 7 — OUTPUT PATH SAFETY]** PASS. The config writes `out_ckpt: "limo/vae_2g76.pt"` at `configs/finetune/04_decoder_init.yaml:4`, while `init_ckpt` remains `limo/vae.pt` at `configs/finetune/04_decoder_init.yaml:3`; code refuses equal init/output paths at `limo/vae_finetune.py:238-242`, saves epoch checkpoints under `runs/decoder` at `limo/vae_finetune.py:331` and `limo/vae_finetune.py:418-419`, and saves final state only to `out_ckpt` at `limo/vae_finetune.py:492-494`. Severity: **INFO**.

## Suggested Fixes

1. CHECK 1: No required fix; optional hardening is to assert frozen `requires_grad` immediately after `optimizer.step()` at `limo/vae_finetune.py:399`.
2. CHECK 2: No required fix; `vae.encoder.eval()` is already reasserted after `vae.train()` at `limo/vae_finetune.py:352`.
3. CHECK 3: No required fix; model and optimizer restore are already paired at `limo/vae_finetune.py:433-434`.
4. CHECK 4: No required fix; full-weight mean is already taken before slicing at `limo/vae_finetune.py:144-150`.
5. CHECK 5: No required fix; finetune/pretraining KLD denominators match at `limo/vae_finetune.py:390-395` and `limo/models.py:54,60`.
6. CHECK 6: No required fix; `replacement=True` is set at `limo/vae_finetune.py:313-315`.
7. CHECK 7: No required fix; configured output is `limo/vae_2g76.pt` at `configs/finetune/04_decoder_init.yaml:4` and final save uses `out_ckpt` at `limo/vae_finetune.py:492-494`.

## Confidence

**High.** Codex read all of `limo/vae_finetune.py:1-532`, all of `configs/finetune/04_decoder_init.yaml:1-28`, and the relevant pretraining loss path at `limo/models.py:52-60`. These checks are static; runtime data/checkpoint behavior was not validated.
