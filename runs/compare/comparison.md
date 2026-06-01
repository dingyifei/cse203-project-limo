# LIMO vs Tengen vs PHGDH controls — AutoDock-GPU comparison on the 2G76 HHTH grid

Grid: `limo/2g76/2g76.maps.fld` (R162-centered, 60³ at 0.375 Å).
Docking: AutoDock-GPU 1.6 + obabel (pH 7.4, Gasteiger, --gen3d), pipeline identical to `docs/finetune/data/priors_docked.csv`.
Two modes shown:
- **single-seed (`-s 0`)** = AutoDock-GPU default; deterministic; matches the prior controls. ΔG_ss column.
- **`-nrun 50`** = 50 random seeds averaged for the top-3 of each source + NCT-503. ΔG mean ± std and min (best pose) shown.
Tengen score = generative-model log-probability (NOT a docking ΔG; less negative = higher gen confidence).

## Headline (`-nrun 50`, 7 mols)

| Source | Compound | mean ΔG | std | best pose ΔG | best Kd | tengen score |
|---|---|---:|---:|---:|---:|---:|
| tengen | tengen_5 |  -4.84 | 0.46 |  -5.95 | 43.5 µM | -0.188 |
| limo | limo_2 |  -5.07 | 0.27 |  -5.48 | 96.2 µM | — |
| limo | limo_4 |  -4.05 | 0.59 |  -5.12 | 176.6 µM | — |
| limo | limo_1 |  -4.98 | 0.11 |  -5.09 | 185.8 µM | — |
| tengen | tengen_6 |  -3.91 | 0.42 |  -4.74 | 335.4 µM | -0.190 |
| control | NCT-503 |  -4.52 | 0.10 |  -4.59 | 432.0 µM | — |
| tengen | tengen_3 |  -3.29 | 0.59 |  -4.49 | 511.5 µM | -0.187 |

## Full single-seed table (21 new docks + published PHGDH controls)

| Source | Rank | Compound | ΔG_ss | Kd | tengen_score | MW | Rot | HBD | HBA | QED | SA | n_arom |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| limo | 1 | limo_1 |  -5.07 | 192.2 | — | 265 | 1 | 0 | 1 | 0.72 | 4.09 | 1 |
| limo | 2 | limo_2 |  -5.40 | 110.1 | — | 288 | 4 | 0 | 0 | 0.59 | 3.23 | 1 |
| limo | 3 | limo_3 |  -5.04 | 202.1 | — | 374 | 7 | 0 | 1 | 0.51 | 3.02 | 2 |
| limo | 4 | limo_4 |  -5.14 | 170.7 | — | 346 | 6 | 1 | 1 | 0.58 | 4.16 | 1 |
| limo | 5 | limo_5 |  -4.94 | 239.3 | — | 262 | 4 | 0 | 0 | 0.77 | 3.81 | 1 |
| limo | 6 | limo_6 |  -4.86 | 273.9 | — | 332 | 6 | 0 | 1 | 0.64 | 4.92 | 1 |
| limo | 7 | limo_7 |  -4.48 | 520.2 | — | 292 | 3 | 1 | 2 | 0.88 | 5.08 | 1 |
| limo | 8 | limo_8 |  -4.65 | 390.4 | — | 284 | 1 | 0 | 1 | 0.70 | 3.21 | 2 |
| limo | 9 | limo_9 |  -4.67 | 377.5 | — | 224 | 2 | 0 | 1 | 0.74 | 2.93 | 1 |
| limo | 10 | limo_10 |  -4.77 | 318.8 | — | 250 | 4 | 1 | 1 | 0.82 | 3.46 | 1 |
| tengen | 1 | tengen_1 |  -3.82 | 1584.6 | -0.161 | 477 | 9 | 6 | 8 | 0.27 | 3.54 | 3 |
| tengen | 2 | tengen_2 |   0.00 | — | -0.173 | 376 | 5 | 5 | 9 | 0.34 | 3.99 | 1 |
| tengen | 3 | tengen_3 |  -4.32 | 681.4 | -0.187 | 489 | 7 | 4 | 7 | 0.46 | 3.68 | 2 |
| tengen | 4 | tengen_4 |  -3.73 | 1844.6 | -0.188 | 427 | 9 | 7 | 6 | 0.26 | 3.33 | 3 |
| tengen | 5 | tengen_5 |  -5.97 | 42.1 | -0.188 | 466 | 5 | 1 | 5 | 0.72 | 3.20 | 2 |
| tengen | 6 | tengen_6 |  -4.77 | 318.8 | -0.190 | 334 | 4 | 2 | 4 | 0.80 | 3.20 | 1 |
| tengen | 7 | tengen_7 |  -3.62 | 2220.9 | -0.192 | 441 | 9 | 7 | 9 | 0.24 | 3.30 | 3 |
| tengen | 8 | tengen_8 |  -3.35 | 3502.9 | -0.193 | 454 | 9 | 6 | 9 | 0.26 | 3.42 | 3 |
| tengen | 9 | tengen_9 |   0.00 | — | -0.194 | 377 | 4 | 5 | 5 | 0.23 | 3.31 | 2 |
| tengen | 10 | tengen_10 |  -3.92 | 1338.5 | -0.195 | 455 | 9 | 6 | 9 | 0.26 | 3.47 | 3 |
| control | 1 | NCT-503 |  -4.59 | 432.0 | — | 325 | 4 | 2 | 3 | 0.91 | 1.59 | 2 |
| prior_control | 0 | BI-4924_priors |  -4.85 | 278.6 | — | 374 | 6 | 2 | 4 | 0.81 | 2.31 | 2 |
| prior_control | 0 | Disulfiram_priors |  -2.51 | 14459.6 | — | 297 | 4 | 0 | 4 | 0.57 | 3.12 | 0 |
| prior_control | 0 | NCT-503_priors |  -4.60 | 424.8 | — | 325 | 4 | 2 | 3 | 0.91 | 1.59 | 2 |
| prior_control | 0 | PKUMDL-WQ-2101_priors |  -4.53 | 478.1 | — | 268 | 2 | 3 | 5 | 0.66 | 2.50 | 3 |
| prior_control | 0 | PKUMDL-WQ-2247_priors |  -4.87 | 269.3 | — | 336 | 4 | 3 | 5 | 0.66 | 2.53 | 3 |

## Reproducibility / sanity

- **NCT-503 reproducibility:** prior single-seed run (in `priors_docked.csv`) gave ΔG = **-4.60**; this comparison batch re-docked NCT-503 and got **-4.59**. Difference = 0.01 kcal/mol → pipeline deterministic. ✓

## Per-source best (`-nrun 50` min)

- **limo** best: limo_2, ΔG_min = -5.48 kcal/mol, Kd = 96.2 µM
  - SMILES: `C1CCC=CC=C1CCC=CC=C2CC=CC3=CC=CC=C23`
- **tengen** best: tengen_5, ΔG_min = -5.95 kcal/mol, Kd = 43.5 µM
  - SMILES: `C[C@@H](C(=O)N1CCOCC1)N1CC[C@H](NS(=O)(=O)c2ccc3cc(Cl)ccc3c2)C1=O`
- **control** best: NCT-503, ΔG_min = -4.59 kcal/mol, Kd = 432.0 µM
  - SMILES: `CC(=O)Nc1ccc(NS(=O)(=O)c2ccc(Cl)cc2)cc1`

## Interpretive notes (for slide caption)

1. **All compounds docked with the same AutoDock-GPU pipeline on the same 2G76 HHTH grid.** Apples-to-apples ΔG comparison.
2. **Tengen's generative-confidence top-3 ≠ tengen's binding top-3.** Tengen rank-5 (`tengen_5`) is the actual best tengen binder by docking (best pose −5.95), even though it's only rank-5 by their gen-likelihood metric.
3. **LIMO's predictor-steered output beats NCT-503 on `mean ΔG` basis** (LIMO rank-2 mean = −5.07 vs NCT-503 mean = −4.52, single-pose −5.40 vs −4.59). NCT-503 itself is the only published HHTH-engaging compound (Chen 2025 AlphaFold dock; not experimentally validated for HHTH).
4. **Single best pose across the whole comparison:** tengen_5 (−5.95 kcal/mol, Kd ≈ 43 µM). This is also the lowest Kd of any compound tested.
5. **NCT-503 has the tightest dock std (0.10)** — pose is well-defined; the others vary more, including LIMO rank-4 (std = 0.59) where the single-seed result was unrepresentative of the seed distribution.
6. **Tengen scaffolds are NAD+ / cofactor mimics** (antifolate, riboflavin, peptidomimetic) — chemistry-of-the-pocket mismatch may explain why several dock poorly on the HHTH (DNA-binding-groove) grid.
7. **Verification next:** HADDOCK (Jason) — different scoring function, protein-flexible. .dlg files for the 7 nrun50-docked compounds are in `runs/compare/dlgs_for_haddock/`.
