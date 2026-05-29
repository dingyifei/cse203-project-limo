**Verdict:** FAIL

**Critical issues:**
- MISSING strict `smiles_to_z` equivalence for B>1: batch-wide mean loss scales each molecule gradient by `1/B` before Adam (`limo/utils.py:168-173`). B=1 does match the serial path: string input becomes one item, uses one-hot target, encoded `z`, Adam `lr=0.1`, 10000 steps, and `exp(decode(z))` MSE (`limo/utils.py:159-173`).
- Vocab guard caveat: `fetch_priors.py` and `__init__.py` do not import `utils` and therefore never call `assert_dm_loaded()` (`fetch_priors.py:36-45`, `__init__.py:8-15`). Direct `utils.py` import loads `dm.pkl` without hash guard (`limo/utils.py:499-500`). All other curation scripts call `assert_dm_loaded()` correctly.
- Holdout leakage cannot be artifact-verified: `X.pt` and `splits.json` are absent on disk. Source logic does sequester before tensor build (`build_tensors.py:217-254`, `build_tensors.py:283-331`) but has not been run yet.

**Non-issues confirmed:**
- Destructive docking guard: `archive_run_failed()` precedes every `smiles_to_affinity()` call (`dock_priors.py:62-72`, `augment_dock.py:97-109`) and is idempotent via existing-archive/source-missing skips (`_log.py:204-211`). PASS.
- ΔG zero placeholder rows: rows with `delta_g_kcal_mol >= 0` dropped before tensor build (`build_tensors.py:186-188`, `build_tensors.py:329-331`). PASS.
- Atomic CSV writes: `.tmp` write then `os.replace()` (`_log.py:92-110`). PASS.

**Suggested fixes:**
- Make vectorized loss serial-exact for any B: change the per-batch mean to per-molecule sum before backprop — `loss = ((torch.exp(vae.decode(z)) - targets) ** 2).mean(dim=1).sum()`.
- Add this 4-mol sanity test before merging:
```python
smis = ["CCO", "c1ccccc1", "CC(=O)NC", "CCN(CC)CC"]
vae.eval()
torch.manual_seed(20260528); z_serial = torch.cat([utils.smiles_to_z(s, vae) for s in smis], 0)
torch.manual_seed(20260528); z_batch = utils.smiles_to_z(smis, vae)
assert torch.allclose(z_batch, z_serial, atol=1e-6, rtol=1e-6), (z_batch - z_serial).abs().max().item()
torch.manual_seed(7); a = utils.smiles_to_z(smis[0], vae)
torch.manual_seed(7); b = utils.smiles_to_z([smis[0]], vae)
assert torch.equal(a, b)
```
- Run `build_tensors.py` and then verify holdout SMILES are disjoint from every index in `splits.json` before Phase 2.
- Add a `dm.pkl` sha256 assertion directly in `utils.py` at load time (line ~499) so vocab drift is caught even when scripts import `utils` directly.

**Confidence:** High — source flow and artifact directories were inspected line-by-line; leakage cannot be fully confirmed because generated outputs (`X.pt`, `splits.json`) are absent on disk.
