# LIMO 2G76 Finetune — Documentation Index

> Hand-curated index of every per-step doc, config, checkpoint, and dataset produced by the 2G76 finetuning workflow. Updated at the end of every phase.

**Plan:** [`/Users/yifeiding/.claude/plans/the-pre-trained-model-is-compiled-sunbeam.md`](/Users/yifeiding/.claude/plans/the-pre-trained-model-is-compiled-sunbeam.md)

**Branch:** `finetune-2g76`

## Per-phase logs

| Phase | Status | Doc | Config |
|---|---|---|---|
| 0 — Scaffolding | scaffolded | [00-overview.md](00-overview.md) | [00_overview.yaml](../../configs/finetune/00_overview.yaml) |
| 1 — Data curation | pending | [01-data-curation.md](01-data-curation.md) | [01_data.yaml](../../configs/finetune/01_data.yaml) |
| 2 — Chem filters | pending | [02-chem-filters.md](02-chem-filters.md) | [02_filters.yaml](../../configs/finetune/02_filters.yaml) |
| 3 — Predictor (initial) | pending | [03-predictor-initial.md](03-predictor-initial.md) | [03_predictor_init.yaml](../../configs/finetune/03_predictor_init.yaml) |
| 4 — Decoder finetune (initial) | pending | [04-decoder-initial.md](04-decoder-initial.md) | [04_decoder_init.yaml](../../configs/finetune/04_decoder_init.yaml) |
| 5 — AL iter 1 | pending | [05-al-iter-1.md](05-al-iter-1.md) | [05_al_iter.yaml](../../configs/finetune/05_al_iter.yaml) |
| 5 — AL iter 2 | pending | [05-al-iter-2.md](05-al-iter-2.md) | same |
| 5 — AL iter 3 | pending | [05-al-iter-3.md](05-al-iter-3.md) | same |
| 6 — Final report | pending | [06-final-report.md](06-final-report.md) | [06_final_eval.yaml](../../configs/finetune/06_final_eval.yaml) |

## Cross-cutting artifacts

- [IMPROVEMENTS.md](IMPROVEMENTS.md) — headline per-phase improvement log (append-only).
- [priors.md](priors.md) — reference inhibitor provenance.
- [data/MANIFEST.csv](data/MANIFEST.csv) — every artifact with sha256.
- [eval/per_phase_metrics.csv](eval/per_phase_metrics.csv) — structured per-phase metrics.
- [reviews/](reviews/) — Codex review verdicts (CK-1 through CK-7).

## Codex review checkpoints

| CK | After phase | Verdict | Doc |
|---|---|---|---|
| 1 | Phase 0 | pending | reviews/CK-1-codex.md |
| 2 | Phase 1 | pending | reviews/CK-2-codex.md |
| 3 | Phase 2 | pending | reviews/CK-3-codex.md |
| 4 | Phase 3 | pending | reviews/CK-4-codex.md |
| 5 | Phase 4 | pending | reviews/CK-5-codex.md |
| 6 | Phase 5 each iter | pending | reviews/CK-6-iter-{1,2,3}-codex.md |
| 7 | Phase 6 | pending | reviews/CK-7-codex.md |
