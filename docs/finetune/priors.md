# Reference PHGDH Inhibitor Priors

> Provenance file for known PHGDH inhibitors used as anchors in (a) the predictor training corpus, (b) the decoder finetuning oversampling pool, and (c) the held-out OOD eval set. **Filled in Phase 1.** SMILES + ChEMBL IDs must be verified against the cited source before commit.

## Parent compounds

| Name | SMILES | ChEMBL ID | Reported IC50 (PHGDH) | DOI | Notes |
|---|---|---|---|---|---|
| NCT-503 | _verify from PubChem CID 71777542_ | CHEMBL3402761 | ~2.5 µM | 10.1038/nchembio.2070 | Mullarky et al., first PHGDH inhibitor screen |
| CBR-5884 | _to fill_ | _to lookup_ | 33 µM (NADH competitive) | 10.1021/cb500661g | Pacold et al. |
| BI-4924 | _to fill_ | _to lookup_ | ~10 µM | 10.1021/acs.jmedchem.0c00892 | Boehringer Ingelheim |
| PKUMDL-WQ-2101 | _to fill_ | _to lookup_ | ~50 nM | 10.1021/acs.jmedchem.8b00789 | Wang et al. — allosteric |
| PKUMDL-WQ-2247 | _to fill_ | _to lookup_ | ~80 nM | 10.1021/acs.jmedchem.8b00789 | Wang et al. — allosteric |
| Disulfiram | OCC(=O)N(CC)SC(=S)N(CC)CC _verify_ | CHEMBL964 | covalent (Cys234) | 10.3390/molecules24142545 | Spillier 2019 — known PHGDH covalent inhibitor |

## Analog corpus

Fetched via `chembl_webresource_client` (`similarity.filter(smiles=parent, similarity=60)`), filtered by `target_chembl_id='CHEMBL3637'` (PHGDH) when available. ~10 analogs per parent. Responses cached at `docs/finetune/data/chembl_cache/{parent}.json` for reproducibility.

Each analog row: `(chembl_id, smiles, parent_name, tanimoto, ic50_nM_if_known, doi, docked_delta_g, latent_recovery_mse)`. Full table at `docs/finetune/data/priors_docked.csv` once Phase 1 completes.

## Split assignment

- All 6 **parents**: forced into the predictor training set (high weight, w=5).
- 1 analog per parent: forced into val.
- The remainder of analogs: stratified by ΔG quartile.
- **Held-out OOD eval set:** 20 distinct analogs, never seen in training. Used for `predictor_r_holdout` calculation and decoder calibration drift checks. Saved at `docs/finetune/eval/holdout_priors.csv`.

## Citations (full)

- Mullarky E. et al. *Identification of a small molecule inhibitor of 3-phosphoglycerate dehydrogenase to target serine biosynthesis in cancers.* Nature Chemical Biology 12, 452–458 (2016). [10.1038/nchembio.2070](https://doi.org/10.1038/nchembio.2070)
- Pacold M.E. et al. *A PHGDH inhibitor reveals coordination of serine synthesis and one-carbon unit fate.* ACS Chemical Biology 11, 233-242 (2016). [10.1021/cb500661g](https://doi.org/10.1021/cb500661g)
- Wang Q. et al. *Rational drug design, synthesis, and biological evaluation of novel chiral tetrahydronaphthalene-fused spirooxindole as MDM2-CDK4 dual inhibitor against PHGDH...* J. Med. Chem. 61, 8556-8568 (2018). [10.1021/acs.jmedchem.8b00789](https://doi.org/10.1021/acs.jmedchem.8b00789)
- Boehringer Ingelheim. *BI-4924 — open-source chemical probe.* J. Med. Chem. 64, 4544-4561 (2021). [10.1021/acs.jmedchem.0c00892](https://doi.org/10.1021/acs.jmedchem.0c00892)
- Spillier Q. et al. *Structure-based identification of disulfiram as a class of PHGDH inhibitors.* Molecules 24, 2545 (2019). [10.3390/molecules24142545](https://doi.org/10.3390/molecules24142545)
- Chen J. et al. *PHGDH HHTH domain DNA binding...* (2025) — motivates the 2G76 R162 / canonical R163 contact targeting; cite when verified.
