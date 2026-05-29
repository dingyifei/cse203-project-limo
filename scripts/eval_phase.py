#!/usr/bin/env python3
"""Per-phase ablation evaluator + IMPROVEMENTS row generator.

Runs at every phase boundary of the LIMO 2G76 finetune workflow.

For the baseline phase (phase_id == "A_baseline"), this script computes metrics
directly from the previous failed run's existing AutoDock outputs under
``limo/outs/*.dlg`` and ``limo/ligands/0/*.pdbqt`` — no fresh generation or
docking is performed.

For every later phase, the script:

  1. Loads the frozen evaluation latents (``--frozen_z``) and decodes them
     through the provided VAE.
  2. Optionally generates ``--num_gen`` extra fresh latents (gradient-optimized
     by the provided predictor + filter chain via ``limo.utils``); these widen
     the funnel beyond the 100-mol frozen-z A/B set.
  3. Runs the composite filter (RDKit FilterCatalog + drug-likeness rules from
     ``--filter_config``).
  4. Predicts ΔG with the provided predictor head; docks the top-N decoded mols
     via ``limo.utils.smiles_to_affinity`` (top-N defaults to 100).
  5. Computes per-phase metrics (see ``COLUMNS`` below) and appends one row to
     ``docs/finetune/eval/per_phase_metrics.csv``.
  6. Writes a markdown summary block to
     ``docs/finetune/eval/per_phase_{phase_id}.md`` with a ΔG histogram, the
     polyene rate, and example SMILES per quartile.
  7. Appends a formatted entry to ``docs/finetune/IMPROVEMENTS.md`` per the
     schema documented in the plan.
  8. Evaluates the phase's decision gate (``GATES`` dict, keyed by phase id);
     exits 0 on PASS, 1 on FAIL. The row is still written either way — failed
     gates are part of the durable record.

The script is importable as a module (``from scripts.eval_phase import
run_eval``) so the active-learning orchestrator can call it without ``argparse``
gymnastics.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Path setup — the limo runtime expects to be importable with ``limo/`` on
# sys.path because its modules import each other unqualified (``from utils
# import *``). We add it lazily so importing this script alone does not
# require the full RDKit/torch stack.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LIMO_PKG_DIR = _REPO_ROOT / "limo"
if str(_LIMO_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_LIMO_PKG_DIR))

# ---------------------------------------------------------------------------
# Schema — per_phase_metrics.csv header (frozen).
# ---------------------------------------------------------------------------

COLUMNS: Tuple[str, ...] = (
    "phase_id",
    "vae_sha",
    "predictor_sha",
    "filter_cfg_sha",
    "n_decoded",
    "n_pass_filter",
    "n_docked",
    "best_dg_actual",
    "mean_top10_dg",
    "mean_top100_dg",
    "predictor_r_holdout",
    "predictor_top_decile_recall",
    "drug_likeness_pass_rate",
    "polyene_rate",
    "mean_qed_top100",
    "mean_sa_top100",
    "n_distinct_scaffolds_top100",
    "diversity_top100",
    "validity_rate",
    "delta_best_dg_vs_baseline",
    "delta_best_dg_vs_prev_phase",
    "gate_passed",
    "ts",
    "git_sha",
)

# ---------------------------------------------------------------------------
# Phase description table — used by the IMPROVEMENTS.md row builder.
# ---------------------------------------------------------------------------

PHASE_DESCRIPTIONS: Dict[str, str] = {
    "A_baseline": "Baseline metrics from the failed pre-finetune run.",
    "B_filters": "RDKit FilterCatalog + ring/polyene cap wired into optimization loop.",
    "C_predictor": "2G76-specific predictor head warm-started from 2iik.",
    "D_decoder": "Decoder finetuned on top-quartile 2G76 corpus.",
    "E_al_iter1": "Active-learning iteration 1.",
    "E_al_iter2": "Active-learning iteration 2.",
    "E_al_iter3": "Active-learning iteration 3.",
    "F_final": "Final eval — top-10 redock with -nrun 50 + structural overlay.",
}

# ---------------------------------------------------------------------------
# Decision gates — inline lookup table per the plan.
# Each gate takes the just-computed row dict and returns True for PASS.
# Some gates need to read the previous phase's row (e.g. polyene drop ≥ 50%
# rel. to ablation C); those gates inspect ``prev`` from the closure helper.
# ---------------------------------------------------------------------------


def _polyene_dropped_50pct(row: Dict[str, Any], prev_C: Optional[Dict[str, Any]]) -> bool:
    if prev_C is None:
        return False
    try:
        prev_rate = float(prev_C.get("polyene_rate") or 0.0)
        this_rate = float(row.get("polyene_rate") or 0.0)
    except (TypeError, ValueError):
        return False
    if prev_rate <= 0:
        # If C had no polyenes to begin with, treat as PASS — we cannot
        # meaningfully require a 50% drop from zero.
        return True
    return (prev_rate <= 0.5 * prev_rate) or (this_rate <= 0.5 * prev_rate)


def _gate_passes_threshold(row: Dict[str, Any], key: str, threshold: float, direction: str = "ge") -> bool:
    val = row.get(key)
    if val is None or val == "":
        return False
    try:
        v = float(val)
    except (TypeError, ValueError):
        return False
    if direction == "ge":
        return v >= threshold
    if direction == "le":
        return v <= threshold
    raise ValueError(f"unknown direction: {direction}")


GATES: Dict[str, Callable[[Dict[str, Any], Optional[Dict[str, Any]]], bool]] = {
    "A_baseline": lambda r, prev: True,  # baseline is informational
    "B_filters": lambda r, prev: _gate_passes_threshold(r, "drug_likeness_pass_rate", 0.20, "ge"),
    "C_predictor": lambda r, prev: (
        _gate_passes_threshold(r, "predictor_r_holdout", 0.40, "ge")
        and _gate_passes_threshold(r, "delta_best_dg_vs_baseline", -0.5, "le")
    ),
    "D_decoder": lambda r, prev: (
        _gate_passes_threshold(r, "validity_rate", 0.80, "ge")
        and _polyene_dropped_50pct(r, prev)
    ),
    "E_al_iter1": lambda r, prev: _gate_passes_threshold(r, "delta_best_dg_vs_prev_phase", -0.1, "le"),
    "E_al_iter2": lambda r, prev: _gate_passes_threshold(r, "delta_best_dg_vs_prev_phase", -0.1, "le"),
    "E_al_iter3": lambda r, prev: _gate_passes_threshold(r, "delta_best_dg_vs_prev_phase", -0.1, "le"),
    "F_final": lambda r, prev: (
        _gate_passes_threshold(r, "best_dg_actual", -7.0, "le")
        and _gate_passes_threshold(r, "polyene_rate", 0.0, "le")  # in top-10 must be 0 — caller passes top-10 metric
    ),
}


# ---------------------------------------------------------------------------
# Generic helpers.
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _sha256_file(path: Optional[str]) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _append_csv_row(path: Path, row: Dict[str, Any], columns: Sequence[str] = COLUMNS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        if new_file:
            w.writeheader()
        # Coerce values to strings to avoid CSV-quoting surprises on floats.
        out = {}
        for c in columns:
            v = row.get(c, "")
            if isinstance(v, float):
                out[c] = f"{v:.6g}"
            else:
                out[c] = "" if v is None else str(v)
        w.writerow(out)


# ---------------------------------------------------------------------------
# Polyene / drug-likeness fallbacks — used when the RDKit-based helpers in
# limo/utils.py are unavailable (e.g. during a dry-run in CI without RDKit).
# ---------------------------------------------------------------------------

_POLYENE_RE = re.compile(r"C=CC=CC=C")  # ≥3 consecutive C=C (4 carbons, 3 bonds)


def _smiles_is_polyene(smiles: str) -> bool:
    """Heuristic: SMILES contains a run of ≥3 consecutive C=C bonds.

    Used both as a regex fallback and as the canonical definition of "polyene"
    in the per-phase metric. The plan's polyene rate column is defined as the
    fraction of top-100 SMILES matching this pattern.
    """
    if not smiles:
        return False
    return bool(_POLYENE_RE.search(smiles))


# ---------------------------------------------------------------------------
# Baseline ingestion — parse limo/outs/*.dlg + limo/ligands/0/*.pdbqt.
# ---------------------------------------------------------------------------


def _parse_dlg_dg(dlg_path: Path) -> Optional[float]:
    try:
        text = dlg_path.read_text(errors="ignore")
    except OSError:
        return None
    # Skip files with the zero-pose sentinel (failed dock).
    if "0.000   0.000   0.000  0.00  0.00" in text:
        return None
    # Mirror the existing limo/utils.py:226 grep + awk pattern.
    for line in text.splitlines():
        if "RANKING" in line:
            parts = re.split(r"\s+", line.strip())
            # The fifth column carries the binding free energy.
            if len(parts) >= 5:
                try:
                    return float(parts[4])
                except ValueError:
                    continue
    return None


def _pdbqt_to_smiles(pdbqt_path: Path) -> Optional[str]:
    """Use obabel to reconvert pdbqt → SMILES. Falls back to None on failure."""
    if shutil.which("obabel") is None:
        return None
    try:
        out = subprocess.check_output(
            ["obabel", str(pdbqt_path), "-osmi", "-xn"],
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    text = out.decode("utf-8", errors="ignore").strip()
    if not text:
        return None
    # obabel prints "<SMILES>\tligandN" on the first line.
    first = text.splitlines()[0].split()
    return first[0] if first else None


def _canonicalize(smiles: str) -> Optional[str]:
    """Canonicalize via RDKit if available; otherwise return the raw string."""
    try:
        from rdkit import Chem  # type: ignore

        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return smiles


def _collect_baseline_pairs(limo_dir: Path) -> List[Tuple[str, float]]:
    """Walk limo/outs/*.dlg + limo/ligands/0/*.pdbqt and return [(smiles, dg)]."""
    outs = limo_dir / "outs"
    ligands = limo_dir / "ligands" / "0"
    pairs: List[Tuple[str, float]] = []
    if not outs.exists():
        return pairs
    for dlg in sorted(outs.glob("ligand*.dlg")):
        m = re.match(r"ligand(\d+)\.dlg", dlg.name)
        if not m:
            continue
        idx = m.group(1)
        dg = _parse_dlg_dg(dlg)
        if dg is None:
            continue
        pdbqt = ligands / f"ligand{idx}.pdbqt"
        if not pdbqt.exists():
            continue
        smi = _pdbqt_to_smiles(pdbqt)
        if not smi:
            continue
        canon = _canonicalize(smi)
        if not canon:
            continue
        pairs.append((canon, dg))
    return pairs


# ---------------------------------------------------------------------------
# Chemistry metrics (RDKit-backed; graceful fallback when RDKit missing).
# ---------------------------------------------------------------------------


def _compute_chem_metrics(smiles_list: Sequence[str]) -> Dict[str, Any]:
    """Compute QED, SA, scaffold count, and diversity for the given top-N list."""
    out: Dict[str, Any] = {
        "mean_qed": float("nan"),
        "mean_sa": float("nan"),
        "n_distinct_scaffolds": 0,
        "diversity": float("nan"),
    }
    if not smiles_list:
        return out
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem, QED  # type: ignore
        from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore
        from rdkit import DataStructs  # type: ignore
    except Exception:
        return out

    qeds: List[float] = []
    sas: List[float] = []
    scaffolds: set = set()
    fps = []
    try:
        from sascorer import calculateScore  # type: ignore
    except Exception:  # pragma: no cover — limo expects sascorer to be importable
        calculateScore = None  # type: ignore

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        try:
            qeds.append(float(QED.qed(mol)))
        except Exception:
            pass
        if calculateScore is not None:
            try:
                sas.append(float(calculateScore(mol)))
            except Exception:
                pass
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            if scaffold:
                scaffolds.add(scaffold)
        except Exception:
            pass
        try:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        except Exception:
            pass

    if qeds:
        out["mean_qed"] = sum(qeds) / len(qeds)
    if sas:
        out["mean_sa"] = sum(sas) / len(sas)
    out["n_distinct_scaffolds"] = len(scaffolds)

    if len(fps) >= 2:
        tani_sum = 0.0
        n_pairs = 0
        for i in range(len(fps)):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
            tani_sum += sum(sims)
            n_pairs += len(sims)
        if n_pairs > 0:
            out["diversity"] = 1.0 - (tani_sum / n_pairs)

    return out


# ---------------------------------------------------------------------------
# Predictor-on-holdout — Pearson r against pre-computed docked ΔG.
# ---------------------------------------------------------------------------


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = sum((xs[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ys[i] - my) ** 2 for i in range(n)) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _predictor_r_holdout(predictor_path: str, holdout_csv: Path) -> float:
    """Forward-pass holdout SMILES through the predictor; correlate with docked ΔG."""
    if not predictor_path or not holdout_csv.exists():
        return float("nan")
    rows = _read_csv_rows(holdout_csv)
    if not rows:
        return float("nan")
    smiles_col = "smiles" if "smiles" in rows[0] else ("SMILES" if "SMILES" in rows[0] else None)
    dg_col = next((c for c in ("delta_g", "docked_dg", "dg", "actual_dg") if c in rows[0]), None)
    if smiles_col is None or dg_col is None:
        return float("nan")
    smiles = [r[smiles_col] for r in rows if r.get(smiles_col)]
    y = [_safe_float(r[dg_col]) for r in rows if r.get(smiles_col)]

    try:
        import torch  # type: ignore
        import utils as limo_utils  # type: ignore — added to sys.path above
        from models import PropertyPredictor  # type: ignore
    except Exception:
        return float("nan")

    try:
        in_dim = limo_utils.dm.dataset.max_len * len(limo_utils.dm.dataset.symbol_to_idx)
        model = PropertyPredictor(in_dim)
        sd = torch.load(predictor_path, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=False)
        model.eval()
        preds: List[float] = []
        kept_y: List[float] = []
        for s, true_dg in zip(smiles, y):
            try:
                hot = limo_utils.smiles_to_one_hot(s).unsqueeze(0)
            except Exception:
                continue
            with torch.no_grad():
                pred = float(model(hot).flatten()[0].item())
            preds.append(pred)
            kept_y.append(true_dg)
        return _pearson(preds, kept_y)
    except Exception:
        return float("nan")


def _top_decile_recall(predicted: Sequence[float], actual: Sequence[float]) -> float:
    """Fraction of the actual-ΔG top decile that the predictor also ranks in its top decile."""
    n = len(predicted)
    if n < 10 or len(actual) != n:
        return float("nan")
    k = max(1, n // 10)
    # Lowest (most negative) ΔG = best binders.
    actual_top = set(sorted(range(n), key=lambda i: actual[i])[:k])
    pred_top = set(sorted(range(n), key=lambda i: predicted[i])[:k])
    return len(actual_top & pred_top) / k


# ---------------------------------------------------------------------------
# VAE generation & decoding helpers — used by the non-baseline path.
# ---------------------------------------------------------------------------


def _load_vae(vae_path: str):
    """Instantiate VAE in the same way generate_molecules.py does."""
    import torch  # type: ignore
    import utils as limo_utils  # type: ignore
    from models import VAE  # type: ignore

    device = limo_utils.device
    vae = VAE(
        max_len=limo_utils.dm.dataset.max_len,
        vocab_len=len(limo_utils.dm.dataset.symbol_to_idx),
        latent_dim=1024,
        embedding_dim=64,
    ).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
    vae.eval()
    return vae, device


def _decode_frozen_z(vae_path: str, frozen_z_path: str) -> List[str]:
    if not frozen_z_path or not Path(frozen_z_path).exists():
        return []
    try:
        import torch  # type: ignore
        import utils as limo_utils  # type: ignore
    except Exception:
        return []
    try:
        vae, device = _load_vae(vae_path)
        z = torch.load(frozen_z_path, map_location=device, weights_only=True)
        with torch.no_grad():
            probs = torch.exp(vae.decode(z))
        return [limo_utils.one_hot_to_smiles(h) for h in probs]
    except Exception as exc:  # pragma: no cover
        print(f"[eval_phase] decode_frozen_z failed: {exc}", file=sys.stderr)
        return []


def _validity_rate(vae_path: str, n: int = 256, seed: int = 42) -> float:
    try:
        import torch  # type: ignore
        import utils as limo_utils  # type: ignore
        from rdkit import Chem  # type: ignore
    except Exception:
        return float("nan")
    try:
        vae, device = _load_vae(vae_path)
        g = torch.Generator(device="cpu").manual_seed(seed)
        z = torch.randn((n, 1024), generator=g).to(device)
        with torch.no_grad():
            probs = torch.exp(vae.decode(z))
        smiles = [limo_utils.one_hot_to_smiles(h) for h in probs]
        ok = sum(1 for s in smiles if s and Chem.MolFromSmiles(s) is not None)
        return ok / max(1, n)
    except Exception:
        return float("nan")


def _filter_pass_rate(smiles_list: Sequence[str], filter_config: Optional[str]) -> Tuple[float, List[bool]]:
    """Apply the composite drug-likeness filter chain. Returns (rate, per-mol flags)."""
    if not smiles_list:
        return float("nan"), []
    try:
        import yaml  # type: ignore
        import utils as limo_utils  # type: ignore
        from rdkit import Chem  # type: ignore
    except Exception:
        return float("nan"), [False] * len(smiles_list)

    thresh: Dict[str, Any] = {}
    if filter_config and Path(filter_config).exists():
        try:
            with open(filter_config) as f:
                thresh = yaml.safe_load(f) or {}
        except Exception:
            thresh = {}

    flags: List[bool] = []
    has_passes_all = hasattr(limo_utils, "one_hots_to_passes_all")
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            flags.append(False)
            continue
        if has_passes_all:
            try:
                # The util operates on one-hot encodings; encode + reshape per call.
                hot = limo_utils.smiles_to_one_hot(smi).unsqueeze(0)
                ok = bool(limo_utils.one_hots_to_passes_all(hot, thresh)[0])
            except Exception:
                ok = False
        else:
            # Fallback heuristic: MW range + rot bonds + polyene check.
            from rdkit.Chem import Descriptors  # type: ignore

            try:
                mw = Descriptors.MolWt(mol)
                rot = Descriptors.NumRotatableBonds(mol)
                ok = (
                    thresh.get("mw_min", 200) <= mw <= thresh.get("mw_max", 500)
                    and rot <= thresh.get("rot_bonds_max", 8)
                    and not _smiles_is_polyene(smi)
                )
            except Exception:
                ok = False
        flags.append(ok)
    return (sum(flags) / len(flags)), flags


def _dock_smiles(smiles_list: Sequence[str], protein_file: str, autodock: str) -> List[float]:
    """Dispatch to ``limo.utils.smiles_to_affinity``; cwd switch to limo/ first."""
    if not smiles_list:
        return []
    try:
        import utils as limo_utils  # type: ignore
    except Exception:
        return [float("nan")] * len(smiles_list)
    cwd = os.getcwd()
    os.chdir(_LIMO_PKG_DIR)
    try:
        affins = limo_utils.smiles_to_affinity(list(smiles_list), autodock, protein_file)
    finally:
        os.chdir(cwd)
    return list(affins)


# ---------------------------------------------------------------------------
# Markdown helpers — per-phase summary + IMPROVEMENTS.md row.
# ---------------------------------------------------------------------------


def _write_phase_summary(phase_id: str, row: Dict[str, Any], smiles_examples: Dict[str, List[str]],
                         out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"per_phase_{phase_id}.md"
    lines: List[str] = []
    lines.append(f"# Per-phase metrics — `{phase_id}`")
    lines.append("")
    lines.append(f"_Generated {row.get('ts','')} — git {row.get('git_sha','')[:12]}_")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for col in COLUMNS:
        lines.append(f"| `{col}` | {row.get(col,'')} |")
    lines.append("")
    if smiles_examples:
        lines.append("## Example SMILES per quartile")
        lines.append("")
        for quartile_name, smiles in smiles_examples.items():
            lines.append(f"### {quartile_name}")
            lines.append("")
            for s in smiles[:5]:
                lines.append(f"- `{s}`")
            lines.append("")
    path.write_text("\n".join(lines) + "\n")
    return path


def _format_improvements_row(phase_id: str, row: Dict[str, Any],
                             baseline: Optional[Dict[str, Any]],
                             prev: Optional[Dict[str, Any]],
                             gate_passed: bool,
                             gate_action: str,
                             notes: str,
                             yaml_link: Optional[str],
                             phase_doc_link: Optional[str]) -> str:
    def _g(d: Optional[Dict[str, Any]], k: str) -> str:
        if d is None:
            return "n/a"
        v = d.get(k)
        if v is None or v == "":
            return "n/a"
        return str(v)

    def _delta(a: str, b: str) -> str:
        try:
            return f"{float(b) - float(a):+.3f}"
        except (TypeError, ValueError):
            return "n/a"

    baseline_dg = _g(baseline, "best_dg_actual")
    prev_dg = _g(prev, "best_dg_actual")
    this_dg = _g(row, "best_dg_actual")
    description = PHASE_DESCRIPTIONS.get(phase_id, phase_id)
    gate_str = "PASS" if gate_passed else "FAIL"

    lines: List[str] = []
    lines.append("")
    lines.append(f"## {phase_id} — {row.get('ts','')}   git: {row.get('git_sha','')[:12]}")
    lines.append(f"Description: {description}")
    lines.append("Primary metric: best_dg_actual")
    lines.append(
        f"  baseline ➜ this phase: {baseline_dg} ➜ {this_dg} kcal/mol   "
        f"(Δ = {_delta(baseline_dg, this_dg)}, target ≤ -7.0)"
    )
    lines.append(
        f"  prev    ➜ this phase: {prev_dg} ➜ {this_dg} kcal/mol   "
        f"(Δ = {_delta(prev_dg, this_dg)})"
    )
    lines.append("Secondary deltas vs prev:")
    for key in (
        "drug_likeness_pass_rate",
        "polyene_rate",
        "predictor_r_holdout",
        "validity_rate",
        "n_distinct_scaffolds_top100",
    ):
        prev_v = _g(prev, key)
        this_v = _g(row, key)
        lines.append(f"  {key}: {prev_v} ➜ {this_v} ({_delta(prev_v, this_v)})")
    lines.append(f"Gate: {gate_str} — {gate_action}.")
    if notes:
        lines.append(f"Notes: {notes}")
    link_parts = [p for p in (yaml_link, phase_doc_link) if p]
    if link_parts:
        lines.append(f"Links: {' · '.join(link_parts)}")
    return "\n".join(lines) + "\n"


def _append_improvements(path: Path, block: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# LIMO 2G76 Finetune — Improvements Log\n\n")
    with path.open("a") as f:
        f.write(block)


# ---------------------------------------------------------------------------
# Top-level driver.
# ---------------------------------------------------------------------------


def _baseline_row(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """Build the baseline row from existing limo/outs + limo/ligands."""
    pairs = _collect_baseline_pairs(_LIMO_PKG_DIR)
    pairs.sort(key=lambda p: p[1])  # most negative ΔG first
    smiles_all = [s for s, _ in pairs]
    dgs = [d for _, d in pairs]

    top100 = smiles_all[:100]
    chem = _compute_chem_metrics(top100)
    polyene_top100 = sum(1 for s in top100 if _smiles_is_polyene(s)) / max(1, len(top100))
    n_decoded = len(smiles_all)

    row: Dict[str, Any] = {
        "phase_id": args.phase,
        "vae_sha": _sha256_file(args.vae),
        "predictor_sha": _sha256_file(args.predictor),
        "filter_cfg_sha": _sha256_file(args.filter_config),
        "n_decoded": n_decoded,
        "n_pass_filter": 0,
        "n_docked": n_decoded,
        "best_dg_actual": min(dgs) if dgs else float("nan"),
        "mean_top10_dg": (sum(dgs[:10]) / max(1, len(dgs[:10]))) if dgs else float("nan"),
        "mean_top100_dg": (sum(dgs[:100]) / max(1, len(dgs[:100]))) if dgs else float("nan"),
        "predictor_r_holdout": "",
        "predictor_top_decile_recall": "",
        "drug_likeness_pass_rate": "",
        "polyene_rate": polyene_top100,
        "mean_qed_top100": chem["mean_qed"],
        "mean_sa_top100": chem["mean_sa"],
        "n_distinct_scaffolds_top100": chem["n_distinct_scaffolds"],
        "diversity_top100": chem["diversity"],
        "validity_rate": "",
        "delta_best_dg_vs_baseline": 0.0,  # baseline is its own reference
        "delta_best_dg_vs_prev_phase": "",
        "gate_passed": True,
        "ts": _now_iso(),
        "git_sha": _git_sha(),
    }
    examples = {
        "top decile": top100[: max(1, len(top100) // 10)],
        "mid decile": top100[len(top100) // 2 : len(top100) // 2 + 5],
        "bottom decile": smiles_all[-5:],
    }
    return row, examples


def _phase_row(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """Build a row for any non-baseline phase."""
    # 1. Decode frozen z (A/B comparison set).
    frozen_smiles = _decode_frozen_z(args.vae, args.frozen_z)
    # 2. Filter the union of frozen decodes (and optionally more — we keep the
    #    eval funnel simple and skip the fresh-gen optim loop, since the AL
    #    orchestrator already calls generate_molecules.py upstream and feeds
    #    its docked.csv into this script via --candidates_csv).
    candidates: List[str] = list(frozen_smiles)
    candidates_csv = getattr(args, "candidates_csv", None)
    candidates_dgs: Dict[str, float] = {}
    if candidates_csv:
        cand_path = Path(candidates_csv)
        if cand_path.exists():
            for r in _read_csv_rows(cand_path):
                smi = r.get("smiles") or r.get("SMILES")
                if not smi:
                    continue
                candidates.append(smi)
                if r.get("actual_dg"):
                    candidates_dgs[smi] = _safe_float(r["actual_dg"])
                elif r.get("dg"):
                    candidates_dgs[smi] = _safe_float(r["dg"])

    candidates = list(dict.fromkeys(candidates))  # dedupe, preserve order
    n_decoded = len(candidates)

    drug_pass_rate, drug_flags = _filter_pass_rate(candidates, args.filter_config)
    passing = [s for s, ok in zip(candidates, drug_flags) if ok]
    n_pass_filter = len(passing)

    # 3. Predict ΔG with the given predictor (only used to choose the top-N
    #    to dock when no candidates_csv was supplied).
    predicted: List[float] = []
    if not candidates_dgs:
        try:
            import torch  # type: ignore
            import utils as limo_utils  # type: ignore
            from models import PropertyPredictor  # type: ignore

            in_dim = limo_utils.dm.dataset.max_len * len(limo_utils.dm.dataset.symbol_to_idx)
            model = PropertyPredictor(in_dim)
            model.load_state_dict(
                torch.load(args.predictor, map_location="cpu", weights_only=True), strict=False
            )
            model.eval()
            for s in passing:
                try:
                    hot = limo_utils.smiles_to_one_hot(s).unsqueeze(0)
                    with torch.no_grad():
                        predicted.append(float(model(hot).flatten()[0].item()))
                except Exception:
                    predicted.append(float("inf"))
        except Exception:
            predicted = [float("inf")] * len(passing)

    # 4. Dock the top-N (by predicted ΔG, ascending — most-negative first).
    if candidates_dgs:
        # Caller already docked — just use their numbers.
        sorted_by_dg = sorted(
            ((s, candidates_dgs.get(s, float("nan"))) for s in candidates if s in candidates_dgs),
            key=lambda p: p[1],
        )
        top_smiles = [s for s, _ in sorted_by_dg]
        top_dgs = [d for _, d in sorted_by_dg]
    elif passing and predicted:
        order = sorted(range(len(passing)), key=lambda i: predicted[i])
        top_smiles = [passing[i] for i in order[: args.num_dock_top]]
        if top_smiles:
            top_dgs = _dock_smiles(top_smiles, args.protein_file, args.autodock_executable)
        else:
            top_dgs = []
    else:
        top_smiles = []
        top_dgs = []

    n_docked = sum(1 for d in top_dgs if d is not None and d < 0)
    actuals = [d for d in top_dgs if d is not None and d < 0]

    # 5. Per-phase chemistry / diversity metrics on top-100.
    top100_smiles = top_smiles[:100]
    chem = _compute_chem_metrics(top100_smiles)
    polyene_top100 = (
        sum(1 for s in top100_smiles if _smiles_is_polyene(s)) / max(1, len(top100_smiles))
    )

    # 6. Predictor metrics on holdout + top-decile recall on this batch.
    r_holdout = _predictor_r_holdout(args.predictor, Path(args.holdout))
    if candidates_dgs and predicted:
        # Recompute predicted alignment for the docked subset only.
        try:
            import torch  # type: ignore
            import utils as limo_utils  # type: ignore
            from models import PropertyPredictor  # type: ignore

            in_dim = limo_utils.dm.dataset.max_len * len(limo_utils.dm.dataset.symbol_to_idx)
            model = PropertyPredictor(in_dim)
            model.load_state_dict(
                torch.load(args.predictor, map_location="cpu", weights_only=True), strict=False
            )
            model.eval()
            preds_for_docked: List[float] = []
            for s in top_smiles:
                hot = limo_utils.smiles_to_one_hot(s).unsqueeze(0)
                with torch.no_grad():
                    preds_for_docked.append(float(model(hot).flatten()[0].item()))
            decile = _top_decile_recall(preds_for_docked, top_dgs)
        except Exception:
            decile = float("nan")
    elif passing and predicted and top_dgs:
        decile = _top_decile_recall(predicted[: len(top_dgs)], top_dgs)
    else:
        decile = float("nan")

    validity = _validity_rate(args.vae)

    row: Dict[str, Any] = {
        "phase_id": args.phase,
        "vae_sha": _sha256_file(args.vae),
        "predictor_sha": _sha256_file(args.predictor),
        "filter_cfg_sha": _sha256_file(args.filter_config),
        "n_decoded": n_decoded,
        "n_pass_filter": n_pass_filter,
        "n_docked": n_docked,
        "best_dg_actual": min(actuals) if actuals else float("nan"),
        "mean_top10_dg": (sum(sorted(actuals)[:10]) / max(1, min(10, len(actuals)))) if actuals else float("nan"),
        "mean_top100_dg": (sum(sorted(actuals)[:100]) / max(1, min(100, len(actuals)))) if actuals else float("nan"),
        "predictor_r_holdout": r_holdout,
        "predictor_top_decile_recall": decile,
        "drug_likeness_pass_rate": drug_pass_rate,
        "polyene_rate": polyene_top100,
        "mean_qed_top100": chem["mean_qed"],
        "mean_sa_top100": chem["mean_sa"],
        "n_distinct_scaffolds_top100": chem["n_distinct_scaffolds"],
        "diversity_top100": chem["diversity"],
        "validity_rate": validity,
        # deltas filled in by the caller.
        "delta_best_dg_vs_baseline": "",
        "delta_best_dg_vs_prev_phase": "",
        "gate_passed": "",
        "ts": _now_iso(),
        "git_sha": _git_sha(),
    }
    examples = {
        "top decile (docked)": top_smiles[: max(1, len(top_smiles) // 10 or 1)],
        "mid decile": top_smiles[len(top_smiles) // 2 : len(top_smiles) // 2 + 5],
        "filtered out (first 5)": [s for s, ok in zip(candidates, drug_flags) if not ok][:5],
    }
    return row, examples


def run_eval(args: argparse.Namespace) -> int:
    """Run the eval, append the row, return 0 on PASS / 1 on FAIL."""
    metrics_csv = Path(args.out_row)
    existing = _read_csv_rows(metrics_csv)
    baseline = existing[0] if existing else None
    prev = existing[-1] if existing else None
    # The "C ablation" row is only meaningful for the D gate.
    prev_C = next((r for r in existing if r.get("phase_id") == "C_predictor"), None)

    is_baseline = args.phase == "A_baseline" or (
        not existing and getattr(args, "force_baseline_from_outs", False)
    )

    if is_baseline:
        row, examples = _baseline_row(args)
    else:
        row, examples = _phase_row(args)

    # Compute deltas.
    if baseline is not None and not is_baseline:
        try:
            row["delta_best_dg_vs_baseline"] = float(row["best_dg_actual"]) - float(
                baseline["best_dg_actual"]
            )
        except (TypeError, ValueError):
            row["delta_best_dg_vs_baseline"] = ""
    if prev is not None and not is_baseline:
        try:
            row["delta_best_dg_vs_prev_phase"] = float(row["best_dg_actual"]) - float(
                prev["best_dg_actual"]
            )
        except (TypeError, ValueError):
            row["delta_best_dg_vs_prev_phase"] = ""

    # Evaluate gate.
    gate_fn = GATES.get(args.phase, lambda r, prev: True)
    gate_passed = bool(gate_fn(row, prev_C if args.phase == "D_decoder" else prev))
    row["gate_passed"] = gate_passed

    # Persist CSV.
    _append_csv_row(metrics_csv, row)

    # Per-phase markdown.
    phase_doc_dir = metrics_csv.parent
    phase_doc = _write_phase_summary(args.phase, row, examples, phase_doc_dir)

    # IMPROVEMENTS.md row.
    gate_action = {
        True: "proceed to next phase",
        False: "investigate; see phase doc for failure mode",
    }[gate_passed]
    improvements_path = Path(args.improvements) if args.improvements else (
        _REPO_ROOT / "docs/finetune/IMPROVEMENTS.md"
    )
    block = _format_improvements_row(
        phase_id=args.phase,
        row=row,
        baseline=baseline,
        prev=prev,
        gate_passed=gate_passed,
        gate_action=gate_action,
        notes=args.notes or "",
        yaml_link=(args.config_link or None),
        phase_doc_link=str(phase_doc.relative_to(_REPO_ROOT)) if phase_doc.is_absolute() else str(phase_doc),
    )
    _append_improvements(improvements_path, block)

    return 0 if gate_passed else 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Per-phase ablation evaluator (LIMO 2G76 finetune)")
    p.add_argument("--phase", required=True, help="phase id (e.g. A_baseline, B_filters, C_predictor, ...)")
    p.add_argument("--vae", default="limo/vae.pt", help="path to VAE checkpoint")
    p.add_argument("--predictor", default="limo/property_models/binding_affinity.pt",
                   help="path to predictor checkpoint")
    p.add_argument("--filter_config", default="configs/finetune/filter_thresholds.yaml",
                   help="YAML with filter thresholds")
    p.add_argument("--frozen_z", default="docs/finetune/eval/frozen_z.pt",
                   help="frozen latents tensor for A/B decoding")
    p.add_argument("--holdout", default="docs/finetune/eval/holdout_priors.csv",
                   help="held-out priors CSV (smiles + docked dg)")
    p.add_argument("--num_gen", type=int, default=2000)
    p.add_argument("--num_dock_top", type=int, default=100)
    p.add_argument("--out_row", default="docs/finetune/eval/per_phase_metrics.csv")
    p.add_argument("--improvements", default="docs/finetune/IMPROVEMENTS.md")
    p.add_argument("--candidates_csv", default=None,
                   help="optional pre-docked candidate CSV (smiles + actual_dg) — used by the AL orchestrator")
    p.add_argument("--protein_file", default="2g76/2g76.maps.fld")
    p.add_argument("--autodock_executable", default="../bin/autodock_gpu_128wi")
    p.add_argument("--config_link", default=None, help="YAML config path to record in the IMPROVEMENTS row")
    p.add_argument("--notes", default=None)
    p.add_argument("--force_baseline_from_outs", action="store_true",
                   help="even if the CSV is empty and phase != A_baseline, parse limo/outs/*.dlg as the baseline")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    return run_eval(args)


if __name__ == "__main__":
    sys.exit(main())
