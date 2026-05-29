# CK-4 Codex Review

**Verdict**
The warm-start, trainer, weighted Huber, OOD sign, and CPU save mechanics mostly pass, but Phase 3 should not ship until split-schema consumption and holdout-leakage validation are fixed.

**Critical Issues**
1. Item 1 PASS - `limo/train_property_predictor.py:200-209`, `limo/models.py:81`: the warm-start path asserts `model.fc[0].weight.shape == sd['fc.0.weight'].shape` before `load_state_dict(..., strict=False)`, covering input-dimension drift before partial loading.
2. Item 2 PASS - `limo/train_property_predictor.py:23-24`, `limo/train_property_predictor.py:216-218`: `_PL_ACCELERATOR` selects CUDA first, otherwise MPS if `torch.backends.mps.is_available()`, otherwise CPU; the config-path `Trainer` passes `accelerator=_PL_ACCELERATOR` and `devices=1`.
3. Item 3 PASS - `limo/train_property_predictor.py:83-90`, `limo/curation/build_tensors.py:331-337`: weighted Huber computes unreduced Huber loss, multiplies by `w`, and reduces with `.mean()`. The tensor builder writes both `y` and `w` as `(N, 1)`, so broadcasting is aligned; device placement relies on Lightning moving the whole batch.
4. Item 4 PASS - `limo/train_property_predictor.py:214-222`, `limo/train_property_predictor.py:318-325`: the config/warm-start training path omits `auto_lr_find`; only the legacy non-config path sets `auto_lr_find=True`.
5. Item 5 FAIL - `limo/train_property_predictor.py:180-185`, `configs/finetune/01_data.yaml:37-42`, `limo/curation/build_tensors.py:353-358`: the trainer does load precomputed JSON rather than re-stratifying, and the split producer records seed `20260528`; however, the trainer expects `splits['train']`, `splits['val']`, `splits['test']`, while the produced schema is `train_idx`, `val_idx`, `test_idx`. This will not consume the intended `splits.json`.
6. Item 6 FAIL - `configs/finetune/03_predictor_init.yaml:17-18`, `limo/train_property_predictor.py:240-247`: the config names `splits.json` and `holdout_priors.csv`, but the training code only loads `holdout_priors_X.pt` for prediction. There is no canonical-SMILES disjointness check between holdout SMILES and `train_idx`; local generated artifacts were also absent, so leakage could not be independently disproved.
7. Item 7 PASS - `limo/train_property_predictor.py:237-247`, `configs/finetune/03_predictor_init.yaml:23-26`: the sign is consistent with more-negative-is-stronger docking convention: the gate is true when `median(pred_priors) < median(pred_test)`.
8. Item 8 PASS - `limo/train_property_predictor.py:254-259`: the save call materializes CPU tensors with `{k: v.cpu() for k, v in model.state_dict().items()}` before `torch.save`.

**Suggested Fixes**
1. Item 5: read `train_idx`/`val_idx`/`test_idx` from `splits.json`, optionally accepting legacy `train`/`val`/`test`; assert `metadata.seed == 20260528`.
2. Item 6: add a preflight that reads `holdout_priors.csv`, `tensors/corpus.csv`, and `splits.json`, canonicalizes SMILES with RDKit, and asserts holdout canonical SMILES are disjoint from train canonical SMILES.

**Confidence**
High for code-path findings at nested LIMO commit `b2c1e28`; medium for split/leakage artifact conclusions because `docs/finetune/data/tensors/*` and `docs/finetune/eval/holdout_priors.csv` were absent locally.
