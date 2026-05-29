#!/usr/bin/env python3
"""Final report generator (Phase 6).

Reads ``docs/finetune/eval/per_phase_metrics.csv`` plus the top-10 redock
CSV (``docs/finetune/data/final_top10_docked.csv``) and emits a single
markdown report at ``docs/finetune/06-final-report.md``.

Contents (per the plan):

  * Before/after comparison table (baseline vs final row).
  * Ablation contribution stack chart (matplotlib stacked bar) written to
    ``docs/finetune/figs/06-contribution-stack.png``.
  * Convergence plot (matplotlib) at ``docs/finetune/figs/06-convergence.png``.
  * Top-10 hits table with RDKit 2D depictions saved under
    ``docs/finetune/figs/06-top10/*.png``.
  * Decision narrative sections compiled from each per-step doc's
    ``## Decisions`` block.
  * Sanity assertion: ``sum(per_phase_delta_best_dg)`` ≈ ``total_delta_best_dg``
    within 0.2 kcal/mol — otherwise emits a ``MISMATCH`` warning section.
  * Closing recommendation: "ship" / "hold" / "iterate" based on the success
    criteria.

The script degrades gracefully when matplotlib or RDKit are unavailable —
plots and depictions are skipped and the report is annotated, rather than
hard-failing.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# CSV helpers.
# ---------------------------------------------------------------------------


def _read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _isfinite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


# ---------------------------------------------------------------------------
# Decision narrative — parse `## Decisions` blocks from per-step .md files.
# ---------------------------------------------------------------------------


def _collect_decisions(docs_dir: Path) -> List[Tuple[str, List[str]]]:
    """Return (step_doc_name, [bullet_list]) for each ``## Decisions`` block."""
    out: List[Tuple[str, List[str]]] = []
    if not docs_dir.exists():
        return out
    for md in sorted(docs_dir.glob("*.md")):
        try:
            text = md.read_text(errors="ignore")
        except OSError:
            continue
        match = re.search(r"(?im)^##\s+Decisions\s*$([\s\S]+?)(?=^##\s|\Z)", text)
        if not match:
            continue
        block = match.group(1)
        bullets = [
            line.strip().lstrip("-* ").strip()
            for line in block.splitlines()
            if line.strip().startswith(("-", "*"))
        ]
        if bullets:
            out.append((md.name, bullets))
    return out


# ---------------------------------------------------------------------------
# Plotting (matplotlib optional).
# ---------------------------------------------------------------------------


def _convergence_plot(rows: List[Dict[str, str]], out_path: Path) -> bool:
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False
    if not rows:
        return False
    phase_ids = [r["phase_id"] for r in rows]
    best_dg = [_safe_float(r.get("best_dg_actual")) for r in rows]
    drug_pass = [_safe_float(r.get("drug_likeness_pass_rate")) for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = list(range(len(phase_ids)))
    ax1.plot(x, best_dg, marker="o", color="tab:blue", label="best ΔG (kcal/mol)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(phase_ids, rotation=30, ha="right")
    ax1.set_ylabel("best ΔG (kcal/mol)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.axhline(-7.0, linestyle="--", color="tab:blue", alpha=0.4, label="target -7.0")
    ax2 = ax1.twinx()
    ax2.plot(x, drug_pass, marker="s", color="tab:orange", label="drug-likeness pass rate")
    ax2.set_ylabel("drug-likeness pass rate", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _contribution_stack(rows: List[Dict[str, str]], out_path: Path) -> bool:
    """Stacked bar of per-phase delta_best_dg_vs_prev_phase contributions."""
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False
    if len(rows) < 2:
        return False
    deltas = [_safe_float(r.get("delta_best_dg_vs_prev_phase")) for r in rows]
    labels = [r["phase_id"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = 0.0
    for i, (label, d) in enumerate(zip(labels, deltas)):
        if not _isfinite(d):
            continue
        color = "tab:green" if d < 0 else "tab:red"
        ax.bar(["per-phase contribution to best_dg"], [d], bottom=bottom, label=f"{label} ({d:+.2f})", color=color, alpha=0.7)
        bottom += d
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Δ best_dg (kcal/mol) — cumulative")
    ax.set_title("Ablation contribution stack (per-phase Δ vs prev)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _draw_top10_depictions(top10: List[Dict[str, Any]], out_dir: Path) -> int:
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import Draw  # type: ignore
    except Exception:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for i, row in enumerate(top10, start=1):
        smi = row.get("smiles") or row.get("SMILES")
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        out_png = out_dir / f"hit_{i:02d}.png"
        try:
            Draw.MolToFile(mol, str(out_png), size=(300, 300))
            count += 1
        except Exception:
            continue
    return count


# ---------------------------------------------------------------------------
# Tanimoto-to-reference helper for the top-10 table.
# ---------------------------------------------------------------------------


def _nearest_reference(
    smiles: str, ref_fps: List[Tuple[str, Any]], morgan_radius: int = 2, n_bits: int = 2048
) -> Tuple[Optional[str], float]:
    if not ref_fps:
        return None, float("nan")
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
        from rdkit import DataStructs  # type: ignore
    except Exception:
        return None, float("nan")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, float("nan")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, morgan_radius, nBits=n_bits)
    best_sim = -1.0
    best_name = None
    for name, ref_fp in ref_fps:
        sim = DataStructs.TanimotoSimilarity(fp, ref_fp)
        if sim > best_sim:
            best_sim = sim
            best_name = name
    return best_name, max(0.0, best_sim)


def _load_reference_fps(priors_csv: Path) -> List[Tuple[str, Any]]:
    rows = _read_rows(priors_csv)
    if not rows:
        return []
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
    except Exception:
        return []
    fps: List[Tuple[str, Any]] = []
    for r in rows:
        smi = r.get("smiles") or r.get("SMILES")
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        name = r.get("name") or r.get("parent") or smi[:24]
        fps.append((name, AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)))
    return fps


# ---------------------------------------------------------------------------
# Report rendering.
# ---------------------------------------------------------------------------


_BEFORE_AFTER_KEYS = (
    ("best_dg_actual", "Best ΔG (kcal/mol)"),
    ("mean_top10_dg", "Mean top-10 ΔG"),
    ("mean_top100_dg", "Mean top-100 ΔG"),
    ("drug_likeness_pass_rate", "Drug-likeness pass rate"),
    ("polyene_rate", "Polyene rate (top-100)"),
    ("predictor_r_holdout", "Predictor r (holdout)"),
    ("validity_rate", "Validity rate"),
    ("n_distinct_scaffolds_top100", "n distinct scaffolds (top-100)"),
    ("diversity_top100", "Top-100 diversity"),
)


def _before_after_table(first: Dict[str, str], last: Dict[str, str]) -> str:
    lines: List[str] = []
    lines.append("| Metric | Baseline (`{}`) | Final (`{}`) | Δ |".format(
        first.get("phase_id", "?"), last.get("phase_id", "?")
    ))
    lines.append("|---|---|---|---|")
    for key, label in _BEFORE_AFTER_KEYS:
        a = first.get(key, "")
        b = last.get(key, "")
        try:
            d = f"{float(b) - float(a):+.3f}"
        except (TypeError, ValueError):
            d = "n/a"
        lines.append(f"| {label} | {a} | {b} | {d} |")
    return "\n".join(lines)


def _top10_table(top10: List[Dict[str, str]], ref_fps: List[Tuple[str, Any]], fig_dir: Path) -> str:
    if not top10:
        return "_(no top-10 redock CSV found.)_"
    rows: List[str] = []
    rows.append("| # | SMILES | Predicted ΔG | Actual ΔG | Nearest ref. | Tanimoto | Depiction |")
    rows.append("|---|---|---|---|---|---|---|")
    for i, hit in enumerate(top10, start=1):
        smi = hit.get("smiles") or hit.get("SMILES") or ""
        pred = hit.get("predicted_dg", hit.get("pred_dg", ""))
        actual = hit.get("actual_dg", hit.get("dg", ""))
        ref_name, sim = _nearest_reference(smi, ref_fps) if smi else (None, float("nan"))
        depict = f"![hit{i:02d}]({fig_dir.as_posix()}/hit_{i:02d}.png)" if smi else ""
        sim_str = f"{sim:.3f}" if sim == sim else "n/a"  # NaN-aware
        ref_str = ref_name or "n/a"
        rows.append(
            f"| {i} | `{smi}` | {pred} | {actual} | {ref_str} | {sim_str} | {depict} |"
        )
    return "\n".join(rows)


def _decision_section(decisions: List[Tuple[str, List[str]]]) -> str:
    if not decisions:
        return "_(no per-step Decisions blocks found.)_"
    lines: List[str] = []
    for doc_name, bullets in decisions:
        lines.append(f"### {doc_name}")
        for b in bullets:
            lines.append(f"- {b}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _sanity_check(rows: List[Dict[str, str]]) -> Tuple[bool, str]:
    """Sum of per-phase Δ vs prev ≈ total Δ (within 0.2 kcal/mol)."""
    if len(rows) < 2:
        return True, "n/a (need ≥2 rows)"
    deltas = [_safe_float(r.get("delta_best_dg_vs_prev_phase")) for r in rows if _safe_float(r.get("delta_best_dg_vs_prev_phase")) == _safe_float(r.get("delta_best_dg_vs_prev_phase"))]
    sum_deltas = sum(d for d in deltas if _isfinite(d))
    try:
        total = float(rows[-1]["best_dg_actual"]) - float(rows[0]["best_dg_actual"])
    except (TypeError, ValueError, KeyError):
        return False, "could not parse best_dg_actual on first/last row"
    diff = abs(sum_deltas - total)
    msg = f"sum(per-phase Δ) = {sum_deltas:+.3f}, total Δ = {total:+.3f}, |diff| = {diff:.3f}"
    return diff < 0.2, msg


def _recommendation(last: Dict[str, str]) -> str:
    best = _safe_float(last.get("best_dg_actual"))
    polyene = _safe_float(last.get("polyene_rate"))
    gate = (last.get("gate_passed") or "").lower()
    if best < -7.0 and polyene <= 0.0 and gate in ("true", "1", "pass"):
        return "**ship** — final ΔG beats target, polyene rate clean, gate PASS."
    if best < -7.0:
        return "**hold** — ΔG target met but secondary criteria (polyene / gate) not fully clear; review before shipping."
    return "**iterate** — ΔG target not yet reached; recommend another AL iter or revisit the predictor."


def build_report(
    metrics_csv: Path,
    top10_csv: Path,
    priors_csv: Path,
    report_path: Path,
    fig_dir: Path,
    docs_dir: Path,
) -> int:
    rows = _read_rows(metrics_csv)
    if not rows:
        print(f"[report] no rows in {metrics_csv}; nothing to do", file=sys.stderr)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            "# LIMO 2G76 Finetune — Final Report\n\n"
            f"_Generated {_dt.datetime.utcnow().isoformat()}Z._\n\n"
            f"`{metrics_csv}` is empty — no rows to summarize.\n"
        )
        return 1

    first = rows[0]
    last = rows[-1]

    # Charts.
    convergence_png = fig_dir / "06-convergence.png"
    contribution_png = fig_dir / "06-contribution-stack.png"
    top10_fig_dir = fig_dir / "06-top10"

    convergence_ok = _convergence_plot(rows, convergence_png)
    contribution_ok = _contribution_stack(rows, contribution_png)

    # Top-10 hits.
    top10 = _read_rows(top10_csv)
    # Sort by docked ΔG, take 10.
    top10.sort(key=lambda r: _safe_float(r.get("actual_dg", r.get("dg", 0.0))))
    top10 = top10[:10]
    drawn = _draw_top10_depictions(top10, top10_fig_dir)

    ref_fps = _load_reference_fps(priors_csv)

    # Decisions narrative.
    decisions = _collect_decisions(docs_dir)

    # Sanity assertion.
    sanity_ok, sanity_msg = _sanity_check(rows)

    # Recommendation.
    recommendation = _recommendation(last)

    # Render markdown.
    out_lines: List[str] = []
    out_lines.append("# LIMO 2G76 Finetune — Final Report")
    out_lines.append("")
    out_lines.append(
        f"_Generated {_dt.datetime.utcnow().isoformat()}Z. "
        f"Source: `{metrics_csv.relative_to(_REPO_ROOT) if metrics_csv.is_absolute() else metrics_csv}`._"
    )
    out_lines.append("")
    out_lines.append("## Before/after — baseline vs final")
    out_lines.append("")
    out_lines.append(_before_after_table(first, last))
    out_lines.append("")
    out_lines.append("## Ablation contribution stack")
    out_lines.append("")
    if contribution_ok:
        rel = contribution_png.relative_to(_REPO_ROOT) if contribution_png.is_absolute() else contribution_png
        out_lines.append(f"![contribution stack]({rel.as_posix()})")
    else:
        out_lines.append("_(matplotlib unavailable or not enough rows — chart skipped.)_")
    out_lines.append("")
    out_lines.append("## Convergence")
    out_lines.append("")
    if convergence_ok:
        rel = convergence_png.relative_to(_REPO_ROOT) if convergence_png.is_absolute() else convergence_png
        out_lines.append(f"![convergence]({rel.as_posix()})")
    else:
        out_lines.append("_(matplotlib unavailable — chart skipped.)_")
    out_lines.append("")
    out_lines.append("## Top-10 hits (redocked)")
    out_lines.append("")
    top10_fig_rel = top10_fig_dir.relative_to(_REPO_ROOT) if top10_fig_dir.is_absolute() else top10_fig_dir
    out_lines.append(_top10_table(top10, ref_fps, top10_fig_rel))
    out_lines.append("")
    if drawn == 0 and top10:
        out_lines.append("_(RDKit unavailable — depictions skipped.)_")
        out_lines.append("")

    # Sanity assertion / MISMATCH section.
    if not sanity_ok:
        out_lines.append("## ⚠ MISMATCH — per-phase Δ does not sum to total Δ")
        out_lines.append("")
        out_lines.append(sanity_msg)
        out_lines.append("")
        out_lines.append(
            "Investigate per-phase rows for missing or stale `delta_best_dg_vs_prev_phase` "
            "values. Common causes: a phase row was hand-edited after the fact, or the "
            "first row's `best_dg_actual` was retroactively updated without re-running "
            "later phases. The sanity check uses a 0.2 kcal/mol tolerance."
        )
        out_lines.append("")

    out_lines.append("## Decisions taken (compiled from per-step docs)")
    out_lines.append("")
    out_lines.append(_decision_section(decisions))
    out_lines.append("")
    out_lines.append("## Sanity check")
    out_lines.append("")
    out_lines.append(f"- {sanity_msg}")
    out_lines.append(f"- gate (final row): `{last.get('gate_passed','?')}`")
    out_lines.append(f"- final best_dg: `{last.get('best_dg_actual','?')}`")
    out_lines.append(f"- final polyene rate (top-100): `{last.get('polyene_rate','?')}`")
    out_lines.append("")
    out_lines.append("## Recommendation")
    out_lines.append("")
    out_lines.append(recommendation)
    out_lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(out_lines))

    if not sanity_ok:
        print(f"[report] SANITY MISMATCH: {sanity_msg}", file=sys.stderr)
        return 1
    return 0


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"PyYAML is required to parse --config={path}; install it or pass the individual flags instead"
        ) from exc
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _resolve_path(p: str) -> Path:
    """Resolve a YAML-supplied path: absolute kept as-is, relative anchored at repo root."""
    if not p:
        return Path()
    path = Path(p)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate the LIMO 2G76 finetune final report (Phase 6)")
    p.add_argument("--config",
                   default=None,
                   help="Phase 6 YAML config (e.g. configs/finetune/06_final_eval.yaml). "
                        "If provided, populates defaults for --metrics_csv, --top10_csv, --out, "
                        "and --fig_dir from its `report.*` / `redock.out_csv` keys. "
                        "Explicit CLI flags still win.")
    p.add_argument("--metrics_csv",
                   default=None,
                   help="per-phase metrics CSV (the canonical source of truth)")
    p.add_argument("--top10_csv",
                   default=None,
                   help="top-10 redock CSV (smiles, predicted_dg, actual_dg)")
    p.add_argument("--priors_csv",
                   default=None,
                   help="reference inhibitor CSV (smiles, name, docked dg) for Tanimoto lookup")
    p.add_argument("--out", dest="report",
                   default=None,
                   help="output report markdown path")
    p.add_argument("--fig_dir", default=None,
                   help="directory to write generated figures into")
    p.add_argument("--docs_dir", default=None,
                   help="directory containing per-step .md files (parsed for Decisions blocks)")
    return p


# Defaults applied when neither --config nor explicit flags supply a value.
_FALLBACK_DEFAULTS: Dict[str, str] = {
    "metrics_csv": "docs/finetune/eval/per_phase_metrics.csv",
    "top10_csv": "docs/finetune/data/final_top10_docked.csv",
    "priors_csv": "docs/finetune/data/priors_docked.csv",
    "report": "docs/finetune/06-final-report.md",
    "fig_dir": "docs/finetune/figs",
    "docs_dir": "docs/finetune",
}


def _resolve_args(args: argparse.Namespace) -> Dict[str, Path]:
    """Merge --config defaults with CLI overrides; missing keys fall back to _FALLBACK_DEFAULTS."""
    cfg_values: Dict[str, str] = {}
    if args.config:
        cfg_path = _resolve_path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"--config not found: {cfg_path}")
        cfg = _load_yaml(cfg_path)
        report_section = cfg.get("report", {}) or {}
        redock_section = cfg.get("redock", {}) or {}
        if report_section.get("metrics_csv"):
            cfg_values["metrics_csv"] = report_section["metrics_csv"]
        if redock_section.get("out_csv"):
            cfg_values["top10_csv"] = redock_section["out_csv"]
        if report_section.get("out"):
            cfg_values["report"] = report_section["out"]
        if report_section.get("fig_dir"):
            cfg_values["fig_dir"] = report_section["fig_dir"]
        # priors_docked and docs_dir aren't in the canonical Phase 6 YAML but support them if present.
        if report_section.get("priors_csv"):
            cfg_values["priors_csv"] = report_section["priors_csv"]
        if report_section.get("docs_dir"):
            cfg_values["docs_dir"] = report_section["docs_dir"]

    def _pick(name: str) -> Path:
        explicit = getattr(args, name, None)
        if explicit:
            return _resolve_path(explicit)
        if name in cfg_values:
            return _resolve_path(cfg_values[name])
        return _resolve_path(_FALLBACK_DEFAULTS[name])

    return {
        "metrics_csv": _pick("metrics_csv"),
        "top10_csv": _pick("top10_csv"),
        "priors_csv": _pick("priors_csv"),
        "report": _pick("report"),
        "fig_dir": _pick("fig_dir"),
        "docs_dir": _pick("docs_dir"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = _resolve_args(args)
    return build_report(
        metrics_csv=paths["metrics_csv"],
        top10_csv=paths["top10_csv"],
        priors_csv=paths["priors_csv"],
        report_path=paths["report"],
        fig_dir=paths["fig_dir"],
        docs_dir=paths["docs_dir"],
    )


if __name__ == "__main__":
    sys.exit(main())
