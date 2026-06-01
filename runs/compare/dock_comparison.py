#!/usr/bin/env python3
"""Dock LIMO top-10 + tengen top-10 + NCT-503 control on the 2G76 HHTH grid.

Usage:
    python dock_comparison.py --mode single_seed     # ~30s; mirrors priors_docked.csv
    python dock_comparison.py --mode nrun50           # ~5-10 min; tighter ΔG ± std

Runs from outer repo root. The actual docking happens via utils.smiles_to_affinity
(which requires cwd=limo/), so this script cd's there itself.

Security note: uses subprocess shell=True with f-string SMILES interpolation,
which mirrors the existing upstream LIMO pattern at limo/utils.py:480
(smiles_to_affinity). SMILES inputs come from controlled sources only
(runs/final/top10.txt from this pipeline, ~/Downloads/2g76_nonvae_flatten.tsv
provided by the user, and a hardcoded NCT-503 PubChem SMILES) — not from any
network or untrusted input — so command-injection surface is internal.
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

OUTER = Path(__file__).resolve().parents[2]
LIMO = OUTER / "limo"
COMPARE_DIR = OUTER / "runs" / "compare"
PROTEIN_FLD = "2g76/2g76.maps.fld"          # relative to cwd=limo/
AUTODOCK = "../bin/autodock_gpu_128wi"      # relative to cwd=limo/

# Sources to dock. Order = original rank within source.
SOURCES = [
    ("limo",     COMPARE_DIR / "limo_top10.smi"),
    ("tengen",   COMPARE_DIR / "tengen_top10.smi"),
    ("control",  COMPARE_DIR / "nct503_control.smi"),
]

NAME_MAP = {
    "control": ["NCT-503"],
}

DELTA_G_TO_KD_uM = lambda dg: math.exp(dg / (0.00198720425864083 * 298.15)) * 1e6

def load_smiles(p):
    return [line.strip() for line in open(p) if line.strip()]

def archive_outs_if_needed(label):
    """Tar limo/outs and limo/ligands/0 to a labeled archive BEFORE smiles_to_affinity
    wipes them, in case anything else cares."""
    outs_tar = COMPARE_DIR / f"prev_outs_before_{label}.tar"
    if (LIMO / "outs").exists() and not outs_tar.exists():
        try:
            with tarfile.open(outs_tar, "w") as tf:
                tf.add(LIMO / "outs", arcname="outs")
            print(f"  archived prev limo/outs -> {outs_tar.name}")
        except Exception as e:
            print(f"  warn: outs archive failed: {e}")

def dock(mode):
    sys.path.insert(0, str(LIMO))
    os.chdir(LIMO)
    from utils import smiles_to_affinity  # noqa

    # Build combined SMILES list with row metadata.
    rows = []
    seq = 0
    for src, smi_path in SOURCES:
        smis = load_smiles(smi_path)
        for rank, s in enumerate(smis, 1):
            rows.append({
                "seq": seq,
                "source": src,
                "rank": rank,
                "smiles": s,
                "name": (NAME_MAP.get(src) or [f"{src}_{rank}"])[min(rank-1, len(NAME_MAP.get(src) or [None])-1)] if src == "control" else f"{src}_{rank}",
            })
            seq += 1

    # Build the SMILES list in order; ligands will be named ligand<seq>.pdbqt etc.
    smiles_list = [r["smiles"] for r in rows]

    print(f"=== dock mode={mode} N={len(smiles_list)} ===")
    archive_outs_if_needed(label=mode)

    if mode == "single_seed":
        # Default smiles_to_affinity uses -s 0 (single seed). Returns one ΔG per mol.
        affins = smiles_to_affinity(smiles_list, AUTODOCK, PROTEIN_FLD)
    elif mode == "nrun50":
        # smiles_to_affinity doesn't accept extra flags. Replicate its logic minimally
        # but with -nrun 50 added to autodock invocation; then parse each .dlg for
        # the lowest-energy pose mean ± std from the cluster-1 entries.
        return dock_nrun50(rows, smiles_list)
    else:
        raise SystemExit(f"unknown mode {mode}")

    # Attach ΔG to rows and write CSV.
    for r, dg in zip(rows, affins):
        r["delta_g_kcal_mol"] = float(dg) if dg is not None else None
        r["kd_uM"] = DELTA_G_TO_KD_uM(dg) if (dg is not None and dg < 0) else None

    out_csv = COMPARE_DIR / "single_seed_docked.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seq","source","rank","name","smiles","delta_g_kcal_mol","kd_uM"])
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\n=== wrote {out_csv} ===")
    for r in rows:
        dg = r["delta_g_kcal_mol"]
        kd = r["kd_uM"]
        dg_s = "  None" if dg is None else f"{dg:6.2f}"
        kd_s = "    None" if kd is None else f"{kd:8.1f}"
        print(f"  {r['source']:7s} rank={r['rank']:2d}  ΔG={dg_s}  Kd={kd_s} µM  {r['smiles'][:60]}")

def dock_nrun50(rows, smiles_list):
    # Take top-3 from each source by single-seed ΔG (must have been run first).
    seed_csv = COMPARE_DIR / "single_seed_docked.csv"
    if not seed_csv.exists():
        raise SystemExit("nrun50 mode requires single_seed_docked.csv first; run --mode single_seed")
    seeded = {}
    with open(seed_csv) as f:
        for r in csv.DictReader(f):
            seeded[(r["source"], int(r["rank"]))] = float(r["delta_g_kcal_mol"]) if r["delta_g_kcal_mol"] not in ("","None") else None

    picked = []
    for src in ("limo","tengen","control"):
        candidates = sorted(
            [r for r in rows if r["source"]==src and seeded.get((src, r["rank"])) is not None],
            key=lambda r: seeded[(r["source"], r["rank"])])
        # Top-3 (lowest ΔG); for control there's only 1.
        picked.extend(candidates[:3 if src!="control" else 1])

    print(f"=== nrun50: re-docking {len(picked)} mols (top-3 each + NCT-503) ===")
    for r in picked:
        print(f"  {r['source']:7s} rank={r['rank']:2d}  single_dg={seeded[(r['source'], r['rank'])]:.2f}  {r['name']}")

    # Replicate utils.smiles_to_affinity but with -nrun 50.
    cwd_limo = str(LIMO)
    os.chdir(cwd_limo)
    for d in ("ligands_nrun50","outs_nrun50"):
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d)
        if d.startswith("ligands"):
            os.makedirs(os.path.join(d, "0"))

    # obabel prep
    procs = []
    for i, r in enumerate(picked):
        cmd = f'obabel -:"{r["smiles"]}" -O ligands_nrun50/0/ligand{i}.pdbqt -p 7.4 --partialcharge gasteiger --gen3d'
        procs.append(subprocess.Popen(cmd, shell=True, stderr=subprocess.DEVNULL))
    for p in procs: p.wait()

    # autodock with -nrun 50 (each ligand individually for simpler dlg parsing)
    for i, r in enumerate(picked):
        lig = f'ligands_nrun50/0/ligand{i}.pdbqt'
        if not os.path.exists(lig):
            print(f"  warn: obabel failed on {r['source']}_{r['rank']}; skipping")
            r["delta_g_mean"] = r["delta_g_std"] = r["delta_g_min"] = None
            continue
        out_stub = f'outs_nrun50/ligand{i}'
        cmd = f'{AUTODOCK} -M {PROTEIN_FLD} -L {lig} -N {out_stub} -nrun 50 -s 0'
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dlg = f'{out_stub}.dlg'
        if not os.path.exists(dlg):
            print(f"  warn: dock failed on {r['source']}_{r['rank']}")
            r["delta_g_mean"] = r["delta_g_std"] = r["delta_g_min"] = None
            continue
        dgs = parse_all_run_dgs(dlg)
        if not dgs:
            r["delta_g_mean"] = r["delta_g_std"] = r["delta_g_min"] = None
            continue
        import statistics as stat
        r["delta_g_mean"] = float(stat.mean(dgs))
        r["delta_g_std"]  = float(stat.pstdev(dgs)) if len(dgs)>1 else 0.0
        r["delta_g_min"]  = float(min(dgs))
        r["n_seeds"]      = len(dgs)
        r["kd_uM"]        = DELTA_G_TO_KD_uM(r["delta_g_min"]) if r["delta_g_min"]<0 else None
        r["dlg_path"]     = str(LIMO / dlg)

    out_csv = COMPARE_DIR / "nrun50_docked.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seq","source","rank","name","smiles","delta_g_mean","delta_g_std","delta_g_min","kd_uM","n_seeds","dlg_path"])
        w.writeheader()
        for r in picked: w.writerow({k: r.get(k, "") for k in w.fieldnames})

    print(f"\n=== wrote {out_csv} ===")
    for r in picked:
        m = r.get("delta_g_mean"); s = r.get("delta_g_std"); mn = r.get("delta_g_min")
        m_s = "  None" if m is None else f"{m:6.2f}"
        s_s = " None" if s is None else f"{s:4.2f}"
        mn_s = "  None" if mn is None else f"{mn:6.2f}"
        print(f"  {r['source']:7s} {r['name']:20s}  mean={m_s} ± {s_s}  min={mn_s}  n={r.get('n_seeds','-')}")

def parse_all_run_dgs(dlg):
    """Return list of best-run ΔG per autodock run in the .dlg."""
    dgs = []
    with open(dlg) as f:
        for line in f:
            # AutoDock-GPU dlg writes 'DOCKED: USER    Estimated Free Energy of Binding    = -X.XX kcal/mol'
            # once per run.
            if "Estimated Free Energy of Binding" in line and "kcal/mol" in line:
                try:
                    parts = line.split("=")
                    val = parts[1].split("kcal/mol")[0].strip()
                    dgs.append(float(val))
                except Exception:
                    pass
    return dgs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single_seed","nrun50"], required=True)
    args = ap.parse_args()
    dock(args.mode)

if __name__ == "__main__":
    main()
