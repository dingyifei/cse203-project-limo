# Per-phase metrics — `A_baseline`

_Generated 2026-05-29T05:18:53Z — git ef7765b115c1_

## Metrics

| Metric | Value |
|---|---|
| `phase_id` | A_baseline |
| `vae_sha` | 30bd21969c87a4aa |
| `predictor_sha` | c8ed6e7c43b595de |
| `filter_cfg_sha` | c5c6c399bec0d3c0 |
| `n_decoded` | 5150 |
| `n_pass_filter` | 0 |
| `n_docked` | 5150 |
| `best_dg_actual` | 0.0 |
| `mean_top10_dg` | 0.0 |
| `mean_top100_dg` | 0.0 |
| `predictor_r_holdout` |  |
| `predictor_top_decile_recall` |  |
| `drug_likeness_pass_rate` |  |
| `polyene_rate` | 0.0 |
| `mean_qed_top100` | 0.4478363959831234 |
| `mean_sa_top100` | 7.068790198863235 |
| `n_distinct_scaffolds_top100` | 67 |
| `diversity_top100` | 0.9063922800222322 |
| `validity_rate` |  |
| `delta_best_dg_vs_baseline` | 0.0 |
| `delta_best_dg_vs_prev_phase` |  |
| `gate_passed` | True |
| `ts` | 2026-05-29T05:18:53Z |
| `git_sha` | ef7765b115c11878ce2d43c9c5d55e9f5b12a68b |

## Example SMILES per quartile

### top decile

- `[C][C]([C])[C]=[C]`
- `[C][C][C][C]1[C][C][C@@]1(Cl)/[C]=[C]/[C]=[C]/C(=O)[C]1[C][C]1[C]N[C][C]([NH])O[C][C]`
- `[C][C]1NN([C][C]=O)[C][C]N[C][C]OC(=O)O1`
- `[C][C]1[C]2O[C][C]N([C])[C]1[C][C]N[N@@]1[C][C]2NC(=O)[C]1`
- `[C]OC(=O)NN[C]N[C]/C([C])=C(F)/[C]=[C]/[C]=[C]`

### mid decile

- `[C][C][C][C]OC([C][C]O[C]([C][NH])/[C]=[C]/[C])=C=C1[C][C]([C])[C]N=[C][C]N1`
- `[C][C][C@]1([C]/N=[C]/[C])N[C][C][C][C][C][C]O[C]([C][C]([C])O)O1`
- `[C][C]/[C]=[C]/[C]=[C]/[C]C(=O)/[C]=[C]/[C]([C])O[C]([C])[C][S]1[C]C(=[C])[C]1`
- `[C][S@]12[C][C][C]NN1O[C][C][C]O[C][C]C(=O)O2`
- `[C][C]1[C][C]2[C][C]2/[C]=[C]/[C]=[C]/[C][C][C]([NH])C(=O)[C](N)/[C]=[C]/[N][C][C]N1`

### bottom decile

- `[C][C][C]1[C][C][C][C][C][C][C][C][C]N[C][C]1`
- `[C][C][C][C][C][C](/[C]=[C]/[C]=O)N[C][C][C][C][C]S[C]([C])C(=O)[N][C]=O`
- `[C][C]O/[C]=C(\N)C(=O)/N=[C]/[C][C][C][C]1S[C][C][C]1[C]`
- `[C][C]([C][C][C]C([C])=[C])[C][C]N([N][NH])C(=O)N/[C]=[C]/[C]=O`
- `[C][C]O[C]/[C]=[C]/[C][C]1[C][C]2O[C][C]=[C][C](S2)C(=O)[N][N]1`

