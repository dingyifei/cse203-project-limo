# Codex Review Prompt Templates

> One template per checkpoint. Each Codex review is dispatched via the `codex:codex-rescue` agent with the corresponding prompt below. Output goes to `docs/finetune/reviews/CK-N-codex.md`.

Codex has zero context from our conversation, so each prompt must include:
- (1) the file paths to review,
- (2) the specific failure modes we're worried about,
- (3) the expected verdict format.

## CK-1 — After Phase 0 (scaffolding)

```
Independent code review of the LIMO 2G76 finetune scaffolding at /Users/yifeiding/projects/school/BENG203/limo.

Review these files for shell quoting bugs, sha256 atomicity, append-only log race conditions, and env-var checkpoint propagation correctness:
- scripts/log_step.sh
- scripts/hash_artifact.sh
- run.sh
- Makefile
- configs/finetune/*.yaml
- .gitignore

Specifically check:
1. Does log_step.sh correctly handle step_ids that contain shell metacharacters?
2. Does hash_artifact.sh produce stable sha256 for directories regardless of traversal order? (Note: it uses `find -print0 | sort -z | xargs -0 cat | shasum`.)
3. Are the CSV writes in hash_artifact.sh single-line atomic? Any partial-write risk?
4. Do the configs/finetune/*.yaml files have consistent key naming, no typos in protein_file paths, no inconsistent units?
5. Does run.sh's --phase / --iter resume logic actually work, or does it have a missing default case?
6. Does .gitignore inadvertently ignore artifacts we want tracked (like docs/finetune/data/tensors/*.pt)?

Output a single markdown file with sections:
**Verdict:** PASS / PASS WITH CAVEATS / FAIL
**Critical issues:** (numbered list; only blockers)
**Suggested fixes:** (specific line-level diffs)
**Confidence:** (low / medium / high)

Be terse. Under 500 words.
```

## CK-2 — After Phase 1 (data curation)

```
Independent code review of LIMO 2G76 Phase 1 data curation.

Review:
- limo/curation/*.py (recover_failed_run.py, augment_dock.py, fetch_priors.py, dock_priors.py, recover_prior_latents.py, build_tensors.py, _log.py)
- The vectorized smiles_to_z patch in limo/utils.py (around line 86)
- docs/finetune/data/{run_failed_735.csv, augment_6000.csv, priors_docked.csv, tensors/*.pt, splits.json} sample rows

Specifically check:
1. Vocab drift: is dm.pkl sha256 verified in every script that loads vocab? Look for `assert sha256(dm.pkl) == ...` or equivalent.
2. The vectorized smiles_to_z must be numerically equivalent to the original utils.py:86 serial version. Verify by reading both, then propose a 4-mol sanity test.
3. utils.smiles_to_affinity does `rm -rf ligands/* outs/*`. Is there a guard at the top of each script that calls it asserting the prior-run archive (limo/outs.run_failed_735.tar) exists?
4. ΔG=0 placeholder rows from failed docks (utils.py:227 clips with `min(affin, 0)`). Are we dropping these from the predictor training corpus, or only down-weighting them?
5. Stratified split leakage: are any of the held-out reference inhibitors (eval/holdout_priors.csv) accidentally present in tensors/X.pt or splits.json's train_idx?

Output: docs/finetune/reviews/CK-2-codex.md with Verdict / Critical issues / Suggested fixes / Confidence. Under 600 words.
```

## CK-3 — After Phase 2 (chemistry filters)

```
Independent code review of LIMO 2G76 Phase 2 chemistry filters.

Review:
- limo/utils.py new functions (one_hots_to_ring_ok, one_hots_to_passes_all, _max_consec_bond) and the rewritten one_hots_to_filter
- limo/generate_molecules.py filter-wiring patch (line ~51 area; argparse additions ~65-74; JSONL dump 83-97)
- tests/test_filters.py
- configs/finetune/filter_thresholds.yaml, multiobj_weights.yaml

Specifically check:
1. Is the RDKit FilterCatalog built at module-level (one-time, ~200ms) or per call (would be slow)?
2. Does _max_consec_bond's graph DFS terminate on cyclic graphs? Does it double-count edges?
3. Adam state corruption: does the in-loop filter modify the z tensor in-place between optimizer steps? Adam tracks per-row exp_avg/exp_avg_sq; row deletion silently corrupts it. Verify the filter is applied AFTER the optim loop, OR that filtered rows have their gradients zeroed without resizing z.
4. YAML config load: what happens if filter_thresholds.yaml is missing a key (e.g., qed_min)? Default fallback or KeyError?
5. Test coverage: do tests cover empty SMILES, MolFromSmiles returning None, single-atom mols, deuterated SMILES?

Output: docs/finetune/reviews/CK-3-codex.md with Verdict / Critical issues / Suggested fixes / Confidence. Under 600 words.
```

## CK-4 — After Phase 3 (predictor head; MOST CRITICAL REVIEW)

```
Independent code review of LIMO 2G76 Phase 3 predictor head — the most critical phase because the trained head is used for ALL subsequent gradient steering.

Review:
- limo/train_property_predictor.py (modified): MPS device branch, warm-start, WeightedHuberPropertyPredictor subclass, stratified split
- limo/property_models/2g76_binding_affinity.pt (state_dict, can sha256 only)
- limo/property_models/2g76_binding_affinity.yaml (provenance)
- docs/finetune/eval/per_phase_metrics.csv row labeled `C_predictor`

Specifically check:
1. In_dim mismatch detection on warm-start: when loading property_models/2iik_binding_affinity.pt into a fresh PropertyPredictor, is `model.fc[0].weight.shape == sd['fc.0.weight'].shape` asserted? If not, strict=False would silently skip the first layer.
2. PL 1.9 accelerator='mps': verify this actually runs on MPS, not silent-fallback to CPU. Check by looking for a `print(trainer.accelerator)` or equivalent assertion. Worst case: training takes 30 min on CPU vs 2 min on MPS — silent CPU fallback would be invisible until inspecting wall time.
3. Huber loss with per-sample weights: F.huber_loss(reduction='none') × weights → .mean(). Verify the weight tensor is on the correct device and broadcasts correctly with the (N, 1) ΔG tensor.
4. Stratified split honors the seed: same input data + same seed = same val/test indices across runs.
5. predictor_r_holdout: the held-out priors (docs/finetune/eval/holdout_priors.csv) must NOT appear in tensors/splits.json['train_idx']. Hash-compare the holdout SMILES to the train SMILES.
6. OOD sanity check sign: priors should predict LOW ΔG (strong binders). Is the prediction sign consistent with the docking convention (more negative = stronger binding)?

Output: docs/finetune/reviews/CK-4-codex.md with Verdict / Critical issues / Suggested fixes / Confidence. Under 700 words.
```

## CK-5 — After Phase 4 (decoder finetune; MOST CRITICAL REVIEW for NEW code)

```
Independent code review of LIMO 2G76 Phase 4 decoder finetune — fully NEW code (limo/vae_finetune.py, ~250 lines), no upstream LIMO precedent.

Review:
- limo/vae_finetune.py (the entire file)
- configs/finetune/04_decoder_init.yaml
- limo/vae_2g76.yaml (provenance)
- Validity gate output if triggered: docs/finetune/decoder_validity_failure.md (may not exist)

Specifically check:
1. Encoder + embedding actually frozen: after every optimizer.step(), verify `all(not p.requires_grad for p in vae.encoder.parameters())` and same for vae.embedding. Are any leaked gradients accidentally updating these?
2. MPS BatchNorm in eval mode for the frozen encoder: `vae.encoder.eval()` should be called every epoch. Check by reading the training loop.
3. Validity gate restoration: when validity drops <80% and we restore the previous epoch's state_dict, are we ALSO restoring the Adam optimizer state? Or will the next epoch resume with stale Adam momentum (exp_avg, exp_avg_sq) that no longer matches the parameters?
4. Maintenance-set sampling from zinc250k.smi: uses fixed seed=0. Does it shadow or compete with the 2G76 corpus's WeightedRandomSampler weights? Concat weight = `0.1 * mean(weights_2g76)` — verify this is computed correctly.
5. Loss numerical scale: `(1-p) * weighted_NLL + p * weighted_KLD` with p=0.05. The pretraining convention at limo/models.py:54 divides KLD by `batch_size * pred.shape[1]`. Does the finetune loss match this convention, or does it use a different normalization?
6. Is the encoder forward pass wrapped in `torch.no_grad()`? It should be — otherwise we waste memory on gradients we don't use.
7. WeightedRandomSampler: replacement=True is the default; if False, the maintenance set may be exhausted in 1 epoch.

Output: docs/finetune/reviews/CK-5-codex.md with Verdict / Critical issues / Suggested fixes / Confidence. Under 800 words.
```

## CK-6 — After each Phase 5 AL iter

```
Independent code review of LIMO 2G76 active-learning iteration {i}.

Review:
- scripts/run_active_learning.py
- runs/al_iter_{i}/docked.csv
- runs/al_iter_{i}/cumulative.pt (if it exists, hash-only)
- The corresponding row in docs/finetune/IMPROVEMENTS.md
- runs/al_iter_{i}/predictor.pt (sha256 only) — should differ from iter {i-1}/predictor.pt

Specifically check:
1. Cumulative corpus dedupe: when iter {i}'s docked set is unioned with iter {i-1}'s cumulative corpus, is dedupe done on canonical SMILES (via Chem.MolToSmiles)? Are conflicting ΔG values resolved by `keep min`?
2. Predictor warm-start chain: iter {i}'s predictor.pt must be warm-started from iter {i-1}'s checkpoint, NOT from the initial property_models/2g76_binding_affinity.pt. Verify by reading the orchestrator's checkpoint-chaining logic.
3. Decoder catastrophic forgetting: validity rate (256 random decodes) should not drop sharply across iters. Compare iter {i}'s validity to iter {i-1}'s in the IMPROVEMENTS.md row.
4. Abort criteria are actually EVALUATED, not just printed. Look for `if best_dg[i] >= best_dg[i-1] - 0.1: break` or similar — and confirm it executes BEFORE the next iter spawns.
5. The IMPROVEMENTS row for this iter must agree with the docked.csv ground truth: best_dg_actual, mean_top10_dg, polyene_rate (compute fresh, don't trust the report).

Output: docs/finetune/reviews/CK-6-iter-{i}-codex.md with Verdict / Critical issues / Suggested fixes / Confidence. Under 500 words.
```

## CK-7 — After Phase 6 (final report)

```
Final independent review of LIMO 2G76 finetune before declaring success.

Review:
- docs/finetune/06-final-report.md
- docs/finetune/data/final_top10_docked.csv
- The .dlg files for the top-10 redocked hits
- docs/finetune/IMPROVEMENTS.md end-to-end
- docs/finetune/eval/per_phase_metrics.csv
- tests/test_filters.py output (`pytest -v` log)

Specifically check:
1. The final report's claimed best ΔG must match the actual redock output. No off-by-one. No unit error (kcal/mol vs kJ/mol).
2. The ablation contribution stack adds up: sum of per-phase Δ_best_dg should ≈ total Δ_best_dg (baseline → final) within ±0.2 kcal/mol rounding tolerance.
3. The top-10 SMILES in the report must all pass `pytest tests/test_filters.py::test_passes_all` — none should contain polyene runs ≥3 consecutive C=C.
4. The RMSD-to-grid-center claim (top-1 centroid ≤ 4 Å of (14.547, 25.866, 14.134)) is correct. Parse the top-1 .dlg yourself.
5. Any phase that failed its gate is honestly represented in the report, not buried in a footnote.
6. The IMPROVEMENTS.md log has rows for every phase including any FAIL gates.

Output: docs/finetune/reviews/CK-7-codex.md with Verdict / Critical issues / Suggested fixes / Confidence. Under 600 words. End with: "**Recommendation: ship / do not ship.**"
```
