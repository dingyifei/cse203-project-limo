# CK-3 Codex Review — LIMO 2G76 Phase 2 Chemistry Filters

> Independent code review by `codex:codex-rescue`, run 2026-05-29.
> Codex sandbox was read-only; this file was hand-saved by team-lead from Codex's response.

## Verdict

**PASS WITH CAVEATS** — no ship blocker, but one config-key drift bug found that would silently ignore YAML ring-limit overrides, plus one test coverage gap (NIH/ZINC catalogs not separately tested; the nonempty-invalid-SMILES path through `one_hots_to_filter` is not exercised).

## Critical Issues

1. **[CHECK 1 — FilterCatalog construction]** PASS. `limo/utils.py:32-44` defines `_build_filter_catalog()` and assigns `_FILTER_CATALOG = _build_filter_catalog()` at module level. Built once at import, not per-call. Severity: **Low**.

2. **[CHECK 2 — DFS cycle termination]** PASS. `limo/utils.py:244-290` `_max_consec_bond` builds a finite bond-ID adjacency and uses a `visited_bonds` set (add at :284, skip at :282, remove at :286). Because each recursive path cannot reuse a bond, cyclic graphs terminate. Severity: **Low**.

3. **[CHECK 3 — Adam state corruption (`hard_log`)]** PASS. `limo/generate_molecules.py:147-158` calls `loss.backward()`, `optimizer.step()`, then probes under `torch.no_grad()` using `vae.decode(z.detach())`; no assignment to `z` occurs in that block. Later `z = z[keep...]` filtering is post-loop at lines 239-250 and 263-267. No in-place modification of `z` between optimizer steps. Severity: **Low**.

4. **[CHECK 4 — YAML key drift / threshold defaults]** **MEDIUM**. `limo/utils.py:363-365` copies `_DEFAULT_FILTER_THRESHOLDS` then calls `.update(thresh)`, and reads values by direct key access at :391. Missing YAML keys fall back to module defaults, so no KeyError. **But `configs/finetune/filter_thresholds.yaml:14-15` uses `ring_size_min`/`ring_size_max` while `limo/utils.py:398` reads `min_ring_size`/`max_ring_size`** — YAML edits to ring limits are silently ignored. Severity: **Medium** (fix shipped: outer repo commit that renames YAML keys to match code).

5. **[CHECK 5 — Test coverage gaps]** **MEDIUM**. `tests/test_filters.py:69-73` defines empty, `xxx`, single-atom, and `[2H]C` fixtures and parametrizes them at :151-173, but no test constructs a one-hot that decodes to a nonempty invalid SMILES hitting `MolFromSmiles(smile) is None` at `limo/utils.py:369-371`. PAINS and Brenk are separately tested at :253-264 and via composite violators at :285-299; NIH and ZINC catalogs are not tested separately. Severity: **Medium**.

6. **[CHECK 6 — Weight sign conventions]** PASS. `configs/finetune/multiobj_weights.yaml:5-8` sets `binding_affinity: 5`, `qed: -4`, `drug_likeness: -10`. `limo/generate_molecules.py:128-145` applies `loss += torch.sum(out) * weight` and minimizes at :147-148. Positive binding weight minimizes predicted ΔG; negative QED/drug-likeness weights maximize those outputs. Sign conventions are internally consistent. Severity: **Low**.

## Suggested Fixes

1. **(SHIPPED)** Rename YAML ring keys from `ring_size_min`/`ring_size_max` to `min_ring_size`/`max_ring_size` in both `configs/finetune/filter_thresholds.yaml` and `configs/finetune/02_filters.yaml`. Add a follow-up unit test that confirms YAML ring-limit overrides are actually applied (defer to a follow-up commit if not in current pass).
2. **(DEFERRED)** Add a test that monkeypatches `one_hot_to_smiles` to return a nonempty invalid SMILES (so `MolFromSmiles` returns `None`) and asserts `one_hots_to_filter` handles it correctly. Add at least one NIH- and one ZINC-specific violator to the test parametrization.

## Confidence

**High.** All five requested files existed and were readable; conclusions above are based on observed code paths only, with explicit [INFERRED] labels where reasoning goes beyond directly observed lines.
