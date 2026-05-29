# LIMO 2G76 finetune — convenience targets.
# Prefer `bash run.sh --phase N` for the canonical entry point; these are shortcuts.

.PHONY: all scaffold data filters predictor decoder al1 al2 al3 final eval-baseline test clean-runs

all:
	bash run.sh

scaffold:
	bash run.sh --phase 0

data:
	bash run.sh --phase 1

filters:
	bash run.sh --phase 2

predictor:
	bash run.sh --phase 3

decoder:
	bash run.sh --phase 4

al1:
	bash run.sh --phase 5 --iter 1

al2:
	bash run.sh --phase 5 --iter 2

al3:
	bash run.sh --phase 5 --iter 3

final:
	bash run.sh --phase 6

eval-baseline:
	bash run.sh --phase eval --phase_id A_baseline

test:
	pytest tests/ -v

clean-runs:
	rm -rf runs/ limo/outs/ligand*.dlg limo/outs/ligand*.xml limo/ligands/*/ligand*.pdbqt
	@echo "Note: archived priors (limo/outs.run_failed_735.tar) preserved."
