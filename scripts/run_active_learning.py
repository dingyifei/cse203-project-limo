#!/usr/bin/env python3
"""Active-learning loop orchestrator (Phase 5).

Coordinates the per-iteration substeps documented in the plan:

  1. Generate B latents via ``limo/generate_molecules.py`` with the current VAE
     + predictor + filter config; dump the per-sample JSONL.
  2. Diversity-rank the decoded SMILES (Morgan fingerprint Tanimoto cluster,
     radius 0.4) → keep the top-N most-diverse predicted-best mols.
  3. Dock the top-N against 2G76 via ``limo.utils.smiles_to_affinity``.
  4. Cumulative corpus dedupe (canonical SMILES, keep min ΔG) — written to
     ``runs/al_iter_{i}/cumulative.csv``.
  5. Rebuild tensors from the cumulative corpus.
  6. Retrain the predictor head warm-started from iter ``i-1``.
  7. Finetune the decoder for 1 epoch on the top-quartile.
  8. Call ``scripts/eval_phase.py --phase E_al_iter{i}`` → appends a row to
     ``per_phase_metrics.csv`` + ``IMPROVEMENTS.md``.
  9. Evaluate abort criteria: improvement Δ < 0.1, validity < 0.80, polyene
     rate > 0.30 → break out of the loop.

The orchestrator is intentionally a thin shell over existing scripts. It never
re-implements docking, RDKit chemistry, or model architecture. Path handling
respects the ``smiles_to_affinity`` destructive-``rm`` quirk by archiving the
existing ``limo/outs/`` and ``limo/ligands/`` dirs before each invocation.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Path setup — same convention as eval_phase.py.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LIMO_PKG_DIR = _REPO_ROOT / "limo"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_LIMO_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_LIMO_PKG_DIR))


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# YAML loader — try PyYAML, fall back to a minimal scalar/dict reader.
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        with path.open() as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"failed to parse YAML {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Cumulative-corpus dedupe — canonical SMILES, keep min ΔG.
# ---------------------------------------------------------------------------


def _canonicalize(smiles: str) -> Optional[str]:
    try:
        from rdkit import Chem  # type: ignore

        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return smiles


def _dedupe_min_dg(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        smi = r.get("smiles") or r.get("SMILES")
        if not smi:
            continue
        canon = _canonicalize(smi) or smi
        try:
            dg = float(r.get("actual_dg", r.get("dg", 0.0)))
        except (TypeError, ValueError):
            dg = 0.0
        if canon not in best:
            best[canon] = {**r, "smiles": canon, "actual_dg": dg}
            continue
        cur = float(best[canon].get("actual_dg", 0.0))
        if dg < cur:
            best[canon] = {**r, "smiles": canon, "actual_dg": dg}
    return sorted(best.values(), key=lambda r: float(r.get("actual_dg", 0.0)))


# ---------------------------------------------------------------------------
# Diversity rank — Morgan fingerprint Tanimoto cluster.
# ---------------------------------------------------------------------------


def _diversity_rank(
    decoded_jsonl: Path,
    top_n: int,
    radius: float,
    morgan_radius: int,
    morgan_bits: int,
) -> List[Dict[str, Any]]:
    """Greedy-cluster decoded mols by Morgan-FP Tanimoto and keep top-N predictions.

    Sorts the input JSONL records by predicted ΔG (ascending = most-negative
    first), then iterates and accepts each candidate if its Tanimoto similarity
    to every already-accepted candidate is below ``1 - radius`` (interpreted as
    the cluster threshold — a candidate joins a new cluster). Stops at ``top_n``.
    """
    if not decoded_jsonl.exists():
        return []
    records: List[Dict[str, Any]] = []
    with decoded_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    # Sort by predicted ΔG (smaller = better binder).
    records.sort(
        key=lambda r: float(
            r.get("delta_g_pred", r.get("predicted_dg", r.get("dg_pred", 0.0)))
        )
    )

    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
        from rdkit import DataStructs  # type: ignore
    except Exception:
        # No RDKit — return the top-N by predicted ΔG without clustering.
        return records[:top_n]

    cluster_threshold = 1.0 - radius  # Tanimoto similarity above which two mols are in the same cluster
    accepted: List[Dict[str, Any]] = []
    accepted_fps: List[Any] = []
    for r in records:
        smi = r.get("smiles") or r.get("SMILES")
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, morgan_radius, nBits=morgan_bits)
        too_similar = False
        if accepted_fps:
            sims = DataStructs.BulkTanimotoSimilarity(fp, accepted_fps)
            if max(sims) >= cluster_threshold:
                too_similar = True
        if too_similar:
            continue
        accepted.append(r)
        accepted_fps.append(fp)
        if len(accepted) >= top_n:
            break
    return accepted


# ---------------------------------------------------------------------------
# Docking — wraps limo.utils.smiles_to_affinity with the cwd switch.
# ---------------------------------------------------------------------------


def _archive_outs(iter_dir: Path) -> None:
    """Tar-archive existing limo/outs and limo/ligands before each dock call.

    ``limo/utils.py:smiles_to_affinity`` destructively ``rm -rf``s ``outs/*``
    and ``ligands/*`` on every invocation, so we snapshot first. The archive
    lives under the iteration dir for traceability.
    """
    outs = _LIMO_PKG_DIR / "outs"
    ligands = _LIMO_PKG_DIR / "ligands"
    if not (outs.exists() or ligands.exists()):
        return
    iter_dir.mkdir(parents=True, exist_ok=True)
    archive = iter_dir / "pre_dock_outs_ligands.tar"
    if archive.exists():
        return
    with tarfile.open(archive, "w") as t:
        if outs.exists():
            t.add(outs, arcname="outs")
        if ligands.exists():
            t.add(ligands, arcname="ligands")


def _dock_smiles(
    smiles_list: Sequence[str], protein_file: str, autodock: str
) -> List[float]:
    if not smiles_list:
        return []
    import utils as limo_utils  # type: ignore

    cwd = os.getcwd()
    os.chdir(_LIMO_PKG_DIR)
    try:
        affins = limo_utils.smiles_to_affinity(list(smiles_list), autodock, protein_file)
    finally:
        os.chdir(cwd)
    return list(affins)


# ---------------------------------------------------------------------------
# Subprocess wrappers — generate, predictor retrain, decoder finetune.
# ---------------------------------------------------------------------------


def _run(cmd: Sequence[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> None:
    """Run a subprocess and stream output. Raise on non-zero exit."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    print(f"[al] $ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=full_env)
    if r.returncode != 0:
        raise RuntimeError(f"command failed (exit {r.returncode}): {' '.join(str(c) for c in cmd)}")


def _generate_molecules(
    iter_dir: Path,
    vae_ckpt: Path,
    predictor_ckpt: Path,
    gen_cfg: Dict[str, Any],
    multiobj_weights: Path,
    filter_config: Path,
) -> Path:
    save_dir = iter_dir / "gen"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = save_dir / "decoded.jsonl"
    cmd = [
        sys.executable,
        "generate_molecules.py",
        f"--vae_checkpoint={vae_ckpt.resolve()}",
        f"--binding_affinity_model={predictor_ckpt.resolve()}",
        f"--num_mols={gen_cfg.get('num_mols', 10000)}",
        f"--save_dir={save_dir.resolve()}",
        f"--out_jsonl={out_jsonl.resolve()}",
        f"--filter_config={filter_config.resolve()}",
        f"--multiobj_weights={multiobj_weights.resolve()}",
        f"--optim_steps={gen_cfg.get('optim_steps', 10)}",
    ]
    _run(cmd, cwd=_LIMO_PKG_DIR)
    return out_jsonl


def _retrain_predictor(
    iter_dir: Path,
    warm_start_ckpt: Path,
    tensors_dir: Path,
    config: Path,
    epochs: int,
) -> Path:
    out = iter_dir / "predictor.pt"
    cmd = [
        sys.executable,
        "train_property_predictor.py",
        f"--config={config.resolve()}",
        f"--warm_start_path={warm_start_ckpt.resolve()}",
        f"--tensors={tensors_dir.resolve()}",
        f"--out={out.resolve()}",
        f"--max_epochs={epochs}",
    ]
    _run(cmd, cwd=_LIMO_PKG_DIR)
    return out


def _finetune_decoder(
    iter_dir: Path,
    init_ckpt: Path,
    tensors_dir: Path,
    config: Path,
    epochs: int,
    top_quartile_only: bool,
) -> Path:
    out = iter_dir / "vae.pt"
    cmd = [
        sys.executable,
        "vae_finetune.py",
        f"--config={config.resolve()}",
        f"--init_ckpt={init_ckpt.resolve()}",
        f"--epochs={epochs}",
        f"--top_quartile_only={'true' if top_quartile_only else 'false'}",
        f"--tensors={tensors_dir.resolve()}",
        f"--out={out.resolve()}",
    ]
    _run(cmd, cwd=_LIMO_PKG_DIR)
    return out


def _build_tensors(corpus_csv: Path, out_dir: Path) -> Path:
    """Rebuild X/y/weights tensors from a cumulative corpus CSV.

    Tries the canonical ``limo.curation.build_tensors`` entry point; if absent,
    falls back to a simple loader using ``limo.utils.smiles_to_one_hot`` and
    ``torch.save``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    curation_script = _LIMO_PKG_DIR / "curation" / "build_tensors.py"
    if curation_script.exists():
        cmd = [
            sys.executable,
            str(curation_script.resolve()),
            f"--input={corpus_csv.resolve()}",
            f"--out_dir={out_dir.resolve()}",
        ]
        _run(cmd, cwd=_LIMO_PKG_DIR)
        return out_dir
    # Fallback minimal builder.
    import utils as limo_utils  # type: ignore
    import torch  # type: ignore

    xs: List[Any] = []
    ys: List[float] = []
    weights: List[float] = []
    with corpus_csv.open() as f:
        for r in csv.DictReader(f):
            smi = r.get("smiles") or r.get("SMILES")
            if not smi:
                continue
            try:
                hot = limo_utils.smiles_to_one_hot(smi)
            except Exception:
                continue
            xs.append(hot)
            ys.append(float(r.get("actual_dg", r.get("dg", 0.0))))
            weights.append(1.0)
    if not xs:
        return out_dir
    X = torch.stack(xs)
    y = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)
    w = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
    torch.save(X, out_dir / "X.pt")
    torch.save(y, out_dir / "y.pt")
    torch.save(w, out_dir / "weights.pt")
    return out_dir


# ---------------------------------------------------------------------------
# Per-iteration driver.
# ---------------------------------------------------------------------------


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv_rows(path: Path, rows: List[Dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


def _eval_phase(
    iter_idx: int,
    vae_ckpt: Path,
    predictor_ckpt: Path,
    filter_config: Path,
    docked_csv: Path,
    config_link: Path,
) -> Tuple[int, Optional[Dict[str, str]]]:
    """Call scripts/eval_phase.py for this iteration. Returns (exit_code, row)."""
    phase_id = f"E_al_iter{iter_idx}"
    metrics_csv = _REPO_ROOT / "docs/finetune/eval/per_phase_metrics.csv"
    cmd = [
        sys.executable,
        str(_SCRIPTS_DIR / "eval_phase.py"),
        f"--phase={phase_id}",
        f"--vae={vae_ckpt.resolve()}",
        f"--predictor={predictor_ckpt.resolve()}",
        f"--filter_config={filter_config.resolve()}",
        f"--candidates_csv={docked_csv.resolve()}",
        f"--out_row={metrics_csv}",
        f"--config_link={config_link}",
    ]
    print(f"[al] $ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd)
    rows = _read_csv_rows(metrics_csv)
    row = next((row for row in reversed(rows) if row.get("phase_id") == phase_id), None)
    return r.returncode, row


def _check_abort(row: Optional[Dict[str, str]], abort_cfg: Dict[str, Any]) -> Optional[str]:
    if row is None:
        return "no metric row produced — eval_phase failed"
    try:
        delta = float(row.get("delta_best_dg_vs_prev_phase") or "nan")
    except ValueError:
        delta = float("nan")
    try:
        validity = float(row.get("validity_rate") or "nan")
    except ValueError:
        validity = float("nan")
    try:
        polyene = float(row.get("polyene_rate") or "nan")
    except ValueError:
        polyene = float("nan")

    no_improve_threshold = -float(abort_cfg.get("best_dg_no_improvement_kcal_mol", 0.1))
    if not (delta != delta) and delta > no_improve_threshold:  # delta != delta catches NaN
        return f"best_dg failed to improve by ≥{abs(no_improve_threshold)} (delta={delta:.3f})"
    if not (validity != validity) and validity < float(abort_cfg.get("validity_floor", 0.80)):
        return f"validity below floor (validity={validity:.3f})"
    if not (polyene != polyene) and polyene > float(abort_cfg.get("polyene_rate_ceiling", 0.30)):
        return f"polyene rate above ceiling (polyene={polyene:.3f})"
    return None


def _append_improvements_iter_summary(
    iter_idx: int, row: Optional[Dict[str, str]], abort_reason: Optional[str]
) -> None:
    """Compact one-liner in IMPROVEMENTS.md — extra to the formal block written by eval_phase."""
    path = _REPO_ROOT / "docs/finetune/IMPROVEMENTS.md"
    if not path.exists():
        return
    if row is None:
        return
    line = (
        f"\n<!-- AL iter {iter_idx} side-summary: "
        f"best_dg={row.get('best_dg_actual','')} validity={row.get('validity_rate','')} "
        f"polyene={row.get('polyene_rate','')} delta_vs_prev={row.get('delta_best_dg_vs_prev_phase','')} "
        f"abort={abort_reason or 'none'} -->\n"
    )
    with path.open("a") as f:
        f.write(line)


def run_active_learning(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path)

    gen_cfg = cfg.get("gen", {})
    rank_cfg = cfg.get("rank", {})
    dock_cfg = cfg.get("dock", {})
    retrain_cfg = cfg.get("retrain", {})
    abort_cfg = cfg.get("abort", {})
    max_iters = int(cfg.get("max_iters", 3))

    multiobj_weights = (_REPO_ROOT / gen_cfg.get(
        "multiobj_weights", "configs/finetune/multiobj_weights.yaml"
    )).resolve()
    filter_config = (_REPO_ROOT / gen_cfg.get(
        "filter_config", "configs/finetune/filter_thresholds.yaml"
    )).resolve()
    protein_file = dock_cfg.get("protein_file", "2g76/2g76.maps.fld")
    autodock = dock_cfg.get("autodock_executable", "../bin/autodock_gpu_128wi")
    top_n = int(rank_cfg.get("top_n", 2000))
    diversity = rank_cfg.get("diversity", {}) or {}
    radius = float(diversity.get("tanimoto_radius", 0.4))
    morgan_radius = int(diversity.get("morgan_radius", 2))
    morgan_bits = int(diversity.get("morgan_bits", 2048))

    iter_template = cfg.get("output", {}).get("iter_dir_template", "runs/al_iter_{i}")
    predictor_cfg_path = (_REPO_ROOT / "configs/finetune/03_predictor_init.yaml").resolve()
    decoder_cfg_path = (_REPO_ROOT / "configs/finetune/04_decoder_init.yaml").resolve()

    # Initial checkpoints come from Phase 3/4 outputs.
    current_predictor = Path(args.init_predictor or "limo/property_models/2g76_binding_affinity.pt")
    current_vae = Path(args.init_vae or "limo/vae_2g76.pt")
    cumulative_corpus_path: Optional[Path] = None
    if args.init_corpus:
        cumulative_corpus_path = Path(args.init_corpus).resolve()

    start_iter = max(1, args.iter or 1)
    stop_after = args.iter if args.iter else max_iters

    for i in range(start_iter, stop_after + 1):
        iter_dir = (_REPO_ROOT / iter_template.format(i=i)).resolve()
        iter_dir.mkdir(parents=True, exist_ok=True)
        print(f"[al] === iter {i} → {iter_dir} ===")

        # 1. Generate.
        decoded_jsonl = _generate_molecules(
            iter_dir=iter_dir,
            vae_ckpt=current_vae.resolve(),
            predictor_ckpt=current_predictor.resolve(),
            gen_cfg=gen_cfg,
            multiobj_weights=multiobj_weights,
            filter_config=filter_config,
        )

        # 2. Diversity rank → top-N.
        accepted = _diversity_rank(
            decoded_jsonl=decoded_jsonl,
            top_n=top_n,
            radius=radius,
            morgan_radius=morgan_radius,
            morgan_bits=morgan_bits,
        )
        candidates_csv = iter_dir / "candidates.csv"
        _write_csv_rows(
            candidates_csv,
            [
                {
                    "smiles": r.get("smiles") or r.get("SMILES"),
                    "predicted_dg": r.get(
                        "delta_g_pred", r.get("predicted_dg", r.get("dg_pred", ""))
                    ),
                }
                for r in accepted
                if r.get("smiles") or r.get("SMILES")
            ],
            columns=("smiles", "predicted_dg"),
        )

        # 3. Dock the 2000 against 2G76.
        _archive_outs(iter_dir)
        smiles_to_dock = [r["smiles"] for r in _read_csv_rows(candidates_csv) if r.get("smiles")]
        affins = _dock_smiles(smiles_to_dock, protein_file, autodock)
        docked_csv = iter_dir / "docked.csv"
        from math import exp

        kd_const = 1.0 / (0.00198720425864083 * 298.15)
        with docked_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=("smiles", "predicted_dg", "actual_dg", "kd"))
            w.writeheader()
            cand_rows = _read_csv_rows(candidates_csv)
            for r, dg in zip(cand_rows, affins):
                try:
                    kd = exp(dg * kd_const) if dg < 0 else ""
                except OverflowError:
                    kd = ""
                w.writerow(
                    {
                        "smiles": r.get("smiles"),
                        "predicted_dg": r.get("predicted_dg", ""),
                        "actual_dg": dg,
                        "kd": kd,
                    }
                )

        # 4. Cumulative dedupe.
        all_rows: List[Dict[str, Any]] = []
        if cumulative_corpus_path and cumulative_corpus_path.exists():
            all_rows.extend(_read_csv_rows(cumulative_corpus_path))
        all_rows.extend(_read_csv_rows(docked_csv))
        deduped = _dedupe_min_dg(all_rows)
        cumulative_csv = iter_dir / "cumulative.csv"
        _write_csv_rows(
            cumulative_csv,
            deduped,
            columns=("smiles", "actual_dg", "predicted_dg", "kd"),
        )
        cumulative_corpus_path = cumulative_csv

        # 5. Rebuild tensors.
        tensors_dir = iter_dir / "tensors"
        _build_tensors(cumulative_csv, tensors_dir)

        # 6. Retrain predictor.
        new_predictor = _retrain_predictor(
            iter_dir=iter_dir,
            warm_start_ckpt=current_predictor.resolve(),
            tensors_dir=tensors_dir.resolve(),
            config=predictor_cfg_path,
            epochs=int(retrain_cfg.get("predictor", {}).get("epochs", 5)),
        )

        # 7. Decoder finetune.
        new_vae = _finetune_decoder(
            iter_dir=iter_dir,
            init_ckpt=current_vae.resolve(),
            tensors_dir=tensors_dir.resolve(),
            config=decoder_cfg_path,
            epochs=int(retrain_cfg.get("decoder", {}).get("epochs", 1)),
            top_quartile_only=bool(retrain_cfg.get("decoder", {}).get("top_quartile_only", True)),
        )

        # 8. Per-iter eval (writes IMPROVEMENTS row + per_phase_metrics row).
        exit_code, row = _eval_phase(
            iter_idx=i,
            vae_ckpt=new_vae.resolve(),
            predictor_ckpt=new_predictor.resolve(),
            filter_config=filter_config,
            docked_csv=docked_csv,
            config_link=cfg_path.resolve(),
        )

        # 9. Abort criteria.
        abort_reason = _check_abort(row, abort_cfg)
        _append_improvements_iter_summary(i, row, abort_reason)
        if abort_reason is not None:
            print(f"[al] aborting after iter {i}: {abort_reason}")
            return 1

        # Advance checkpoints for the next iter.
        current_predictor = new_predictor
        current_vae = new_vae

    print(f"[al] completed {stop_after - start_iter + 1} iter(s) without abort.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LIMO active-learning orchestrator (Phase 5)")
    p.add_argument("--config", default="configs/finetune/05_al_iter.yaml",
                   help="path to the AL YAML config")
    p.add_argument("--iter", type=int, default=None,
                   help="if set, run only this iteration (1-indexed); otherwise loop 1..max_iters")
    p.add_argument("--init_vae", default=None,
                   help="VAE checkpoint to start iter 1 from (default: limo/vae_2g76.pt)")
    p.add_argument("--init_predictor", default=None,
                   help="predictor checkpoint to start iter 1 from (default: limo/property_models/2g76_binding_affinity.pt)")
    p.add_argument("--init_corpus", default=None,
                   help="cumulative-corpus CSV to seed iter 1 (default: none — fresh from this iter's dock)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    return run_active_learning(args)


if __name__ == "__main__":
    sys.exit(main())
