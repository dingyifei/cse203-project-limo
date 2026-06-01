# runs/compare/ — LIMO vs Tengen vs PHGDH controls AutoDock comparison

## What this is

Head-to-head AutoDock-GPU docking of three sources of small molecules against the PHGDH 2G76 HHTH (R162) grid:

- **LIMO top-10** — output of our finetuned LIMO pipeline (`runs/final/top10.txt`); chemistry is hydrophobic drug-like scaffolds.
- **Tengen top-10** — top-10 by generative-model log-prob from `~/Downloads/2g76_nonvae_flatten.tsv` (Jason's model); chemistry is NAD+/cofactor mimics.
- **NCT-503 control** — published PHGDH inhibitor with reported (computational-only) HHTH engagement (Chen 2025).
- **Other prior PHGDH inhibitors** (BI-4924, PKUMDL-WQ-2101/2247, Disulfiram) re-used from `docs/finetune/data/priors_docked.csv`.

## Methods (apples-to-apples)

- **Grid**: `limo/2g76/2g76.maps.fld` — 60³ at 0.375 Å spacing, centered at (14.547, 25.866, 14.134) = the R162 sidechain.
- **Docking**: AutoDock-GPU 1.6 (binary `bin/autodock_gpu_128wi`).
- **Ligand prep**: OpenBabel `obabel -p 7.4 --partialcharge gasteiger --gen3d`.
- **Two modes**:
  - Single-seed (`-s 0`, default): 21 SMILES (10 LIMO + 10 tengen + 1 NCT-503) docked in one batch.
  - `-nrun 50`: top-3 from each source (by single-seed ΔG) + NCT-503 re-docked with 50 random seeds for mean ± std.
- **Driver**: `runs/compare/dock_comparison.py`. Reuses `limo/utils.py:smiles_to_affinity`.

## Outputs

- `single_seed_docked.csv` — 21 rows.
- `nrun50_docked.csv` — 7 rows.
- `comparison.csv` — full join with prior controls + computed chem properties.
- `comparison.md` — slide-ready markdown.
- `comparison.png` — bar chart.
- `dlgs_for_haddock/` — the 7 .dlg files for Jason's HADDOCK refinement.

## Sanity checks performed

1. NCT-503 ΔG matches between original priors run and this re-dock (within < 0.02 kcal/mol on single seed).
2. All 21 SMILES parse via RDKit MolFromSmiles before docking.
3. Phase 5 final-run dlgs/pdbqts archived to `runs/final/dlgs_phase5_final.tar` before docking overwrote them.

## Caveats for the slide

- **Docking ≠ binding measurement.** AutoDock-GPU is an empirical scoring function; the LIMO-vs-NCT-503 gap is real but the absolute numbers are screening-grade.
- **HHTH groove has no experimentally validated binder.** NCT-503's HHTH engagement is itself computational (Chen 2025, AlphaFold dock). No SPR/co-crystal/HDX evidence exists for *any* small molecule binding the PHGDH HHTH domain.
- **Tengen targeted a different pocket.** The cofactor-mimic chemistry suggests their grid was the NAD+ pocket; on our HHTH grid, most of tengen's compounds dock poorly. tengen_5 is an outlier (sulfonamide-naphthalene + morpholine, single best pose).
- **HADDOCK (Jason) is the next verification layer.** Different scoring function (CNS-derived), protein-flexible.
