#!/usr/bin/env bash
# run.sh — top-level driver for the LIMO 2G76 finetune workflow.
#
# Usage:
#   bash run.sh                            # run phases 0..6 in order
#   bash run.sh --phase 5 --iter 2         # resume from a specific phase / AL iter
#   bash run.sh --phase 0                  # just scaffold (no compute)
#   bash run.sh --phase eval --phase_id B_filters    # run the ablation evaluator for a phase
#
# Each phase reads configs/finetune/NN_*.yaml, calls scripts/log_step.sh at start/end,
# and writes its results to docs/finetune/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

PHASE="all"
ITER=""
PHASE_ID=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --phase)    PHASE="$2"; shift 2 ;;
    --iter)     ITER="$2"; shift 2 ;;
    --phase_id) PHASE_ID="$2"; shift 2 ;;
    -h|--help)
      grep -E '^# ' "$0" | sed 's/^# //'
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

log_step() { scripts/log_step.sh "$@"; }

cd_limo() { cd "$REPO_ROOT/limo"; }
cd_root() { cd "$REPO_ROOT"; }

phase0() {
  log_step 00-overview start
  echo "phase 0 (scaffolding): docs/configs/scripts already present"
  log_step 00-overview end status=done
}

phase1() {
  log_step 01-data-curation start
  cd_limo
  python -m curation.recover_failed_run --config ../configs/finetune/01_data.yaml
  python -m curation.fetch_priors       --config ../configs/finetune/01_data.yaml
  python -m curation.dock_priors        --config ../configs/finetune/01_data.yaml
  python -m curation.augment_dock       --config ../configs/finetune/01_data.yaml
  python -m curation.recover_prior_latents --config ../configs/finetune/01_data.yaml
  python -m curation.build_tensors      --config ../configs/finetune/01_data.yaml
  cd_root
  log_step 01-data-curation end status=done
  scripts/eval_phase.py --phase A_baseline   --config configs/finetune/00_overview.yaml
}

phase2() {
  log_step 02-chem-filters start
  pytest tests/test_filters.py -v
  cd_limo
  # Smoke test: run the recovered SMILES through the new filter chain (engineering smoke, no compute)
  python -c "from utils import one_hots_to_passes_all; print('filter chain importable')"
  cd_root
  log_step 02-chem-filters end status=done
  scripts/eval_phase.py --phase B_filters    --config configs/finetune/02_filters.yaml
}

phase3() {
  log_step 03-predictor-initial start
  cd_limo
  python train_property_predictor.py --config ../configs/finetune/03_predictor_init.yaml
  cd_root
  log_step 03-predictor-initial end status=done
  scripts/eval_phase.py --phase C_predictor  --config configs/finetune/03_predictor_init.yaml
}

phase4() {
  log_step 04-decoder-initial start
  cd_limo
  python vae_finetune.py --config ../configs/finetune/04_decoder_init.yaml
  cd_root
  log_step 04-decoder-initial end status=done
  scripts/eval_phase.py --phase D_decoder    --config configs/finetune/04_decoder_init.yaml
}

phase5() {
  local i="${ITER:-1}"
  log_step "05-al-iter-${i}" start
  scripts/run_active_learning.py --iter "$i" --config configs/finetune/05_al_iter.yaml
  log_step "05-al-iter-${i}" end status=done
  scripts/eval_phase.py --phase "E_al_iter${i}" --config configs/finetune/05_al_iter.yaml
}

phase6() {
  log_step 06-final-report start
  scripts/build_final_report.py --config configs/finetune/06_final_eval.yaml
  log_step 06-final-report end status=done
}

phase_eval() {
  local pid="${PHASE_ID:?--phase_id required when --phase eval}"
  scripts/eval_phase.py --phase "$pid" --config configs/finetune/00_overview.yaml
}

case "$PHASE" in
  all)  phase0; phase1; phase2; phase3; phase4; ITER=1 phase5; ITER=2 phase5; ITER=3 phase5; phase6 ;;
  0)    phase0 ;;
  1)    phase1 ;;
  2)    phase2 ;;
  3)    phase3 ;;
  4)    phase4 ;;
  5)    phase5 ;;
  6)    phase6 ;;
  eval) phase_eval ;;
  *)    echo "unknown phase: $PHASE" >&2; exit 2 ;;
esac
