# Phase 0 — Overview & Baseline

**Status:** scaffolded
**Started:** 2026-05-28
**Goal:** finetune the LIMO generative model for PHGDH 2G76 HHTH groove (R162 contact), producing drug-like hits with ΔG ≤ −7.0 kcal/mol, with full per-step traceability.

## Failed-run baseline (input to this work)

| Metric | Value |
|---|---|
| Best ΔG | −5.61 kcal/mol (Kd ≈ 77 µM) |
| Mean top-10 ΔG | TBD (computed by Phase 0 script) |
| Pipeline funnel | 10,000 decoded → 1,457 passed cycle filter → 753 QED > 0.4 → 735 docked |
| Top-3 hits | all polyene scaffolds (`C=CC=CC=C...`) — chemically suspect |
| Validity (SELFIES) | ~100% |
| Wall time on M3 Max | ~12 min |

## Root-cause diagnosis

| Bug | Location | Effect |
|---|---|---|
| A | `utils.py:143-153` cycle filter | rejects only undesired-size rings, NOT absence of rings → polyenes pass `cycles==0` trivially |
| B | `utils.py:101-111` PAINS/Brenk wrapper | defined but never called; `rd_filters` not installed |
| C | `generate_molecules.py:23,46` | uses 1err/ESR1 binding head, gradient points at estrogen-receptor binders, not PHGDH |
| D | `train_property_predictor.py:10` | lacks MPS device branch → CPU fallback on M3 |
| E | `generate_molecules.py:92-100` | only `print()`s top-K; 10k SMILES lost |
| F | `generate_molecules.py:46,70` | QED ≥ 0.4 + weight −8 doesn't reject polyenes |
| G | repo-wide | no gradient-based decoder finetuning (`fine-tune.py` is single-SMILES atom substitution) |

## Six-phase plan

1. **Phase 0** — scaffolding (docs, configs, scripts, git branch).
2. **Phase 1** — data curation: recover 735 + augment 6000 random docks + 66 reference inhibitors.
3. **Phase 2** — chemistry filters: RDKit FilterCatalog + ring-OK + polyene cap; differentiable drug-likeness MLP head.
4. **Phase 3** — 2G76 predictor head, warm-started from 2iik (ACAA1).
5. **Phase 4** — decoder finetune (NEW `vae_finetune.py`): freeze encoder, weighted NLL+KLD on top-quartile + priors, 10% ZINC maintenance.
6. **Phase 5** — active-learning loop (2-3 iters of gen → dock → retrain → finetune).
7. **Phase 6** — final eval (top-10 redock + RMSD) and report.

Per-phase ablations measure the contribution of each lever; per-phase Codex review checkpoints (CK-1 through CK-7) provide independent code review.

## Success criteria (final gate)

- `best_dg_actual < −7.0` kcal/mol
- top-10 polyene rate = 0%
- top-1 redocked centroid ≤ 4 Å of grid center (14.547, 25.866, 14.134)
- Predictor `r_holdout ≥ 0.4`
- Validity ≥ 80%

## Links

- Approved plan: `/Users/yifeiding/.claude/plans/the-pre-trained-model-is-compiled-sunbeam.md`
- Improvements log: [IMPROVEMENTS.md](IMPROVEMENTS.md)
- Manifest of artifacts: [data/MANIFEST.csv](data/MANIFEST.csv)
- Codex reviews: [reviews/](reviews/)
- `2026-05-29T03:25:48Z` [ef7765b+812beb9] (Mac.localdomain) **start**
- `2026-05-29T03:25:48Z` [ef7765b+812beb9] (Mac.localdomain) **end** status=done
- `2026-05-29T03:32:31Z` [ef7765b+ba87432] (Yifeis-MacBook-Pro-16.local) **start**
- `2026-05-29T03:32:31Z` [ef7765b+ba87432] (Yifeis-MacBook-Pro-16.local) **end** status=done
- `2026-05-29T03:32:52Z` [ef7765b+ba87432] (Yifeis-MacBook-Pro-16.local) **start**
- `2026-05-29T03:32:52Z` [ef7765b+ba87432] (Yifeis-MacBook-Pro-16.local) **end** status=done
- `2026-05-29T05:57:02Z` [8426107+4b333cc] (Yifeis-MacBook-Pro-16.local) **start**
- `2026-05-29T05:57:02Z` [8426107+4b333cc] (Yifeis-MacBook-Pro-16.local) **end** status=done
