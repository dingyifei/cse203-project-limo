"""
get_binding_site_center.py

Computes the geometric center of a user-defined residue range (the DNA-binding
domain, AA 103-165) directly from a PDB file.  This replaces the HETATM-based
approach used for PX16/phgdh because 2G76 has no pre-docked reference ligand.

Usage:
    python get_binding_site_center.py

Output:
    2g76_out.csv  — center_x, center_y, center_z for use with build_bindata.sh
"""

import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
PDB_FILE      = "2g76.pdb"      # must be in the same directory
PDBID         = "2g76"
CHAIN         = "A"             # chain containing the DNA-binding domain
RES_START     = 103             # first residue of DNA-binding domain
RES_END       = 165             # last  residue of DNA-binding domain
BACKBONE_ONLY = False           # True → use only CA atoms; False → all heavy atoms
EXCLUDE_H     = True            # ignore hydrogen atoms
# ─────────────────────────────────────────────────────────────────────────────


def parse_pdb_atoms(fname, chain, res_start, res_end,
                    backbone_only=False, exclude_h=True):
    """
    Parse ATOM records from a PDB file and return coordinates for residues
    in [res_start, res_end] on the specified chain.
    Returns a list of (x, y, z) tuples.
    """
    coords = []
    with open(fname) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec not in ("ATOM",):          # skip HETATM, TER, etc.
                continue

            atom_name  = line[12:16].strip()
            chain_id   = line[21].strip()
            try:
                res_seq = int(line[22:26].strip())
            except ValueError:
                continue

            element = line[76:78].strip() if len(line) > 76 else atom_name[0]

            # Chain filter
            if chain and chain_id != chain:
                continue

            # Residue range filter
            if not (res_start <= res_seq <= res_end):
                continue

            # Hydrogen filter
            if exclude_h and (element == "H" or atom_name.startswith("H")):
                continue

            # Backbone-only filter
            if backbone_only and atom_name not in ("CA", "C", "N", "O"):
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            coords.append((x, y, z))

    return coords


def main():
    print(f"Reading {PDB_FILE} ...")
    coords = parse_pdb_atoms(
        PDB_FILE, CHAIN, RES_START, RES_END,
        backbone_only=BACKBONE_ONLY, exclude_h=EXCLUDE_H
    )

    if not coords:
        raise RuntimeError(
            f"No atoms found for chain {CHAIN}, residues {RES_START}-{RES_END}. "
            "Check your PDB file, chain ID, and residue numbering."
        )

    arr = np.array(coords)
    cx, cy, cz = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()

    print(f"  Atoms used : {len(coords)}")
    print(f"  Center     : ({cx:.3f}, {cy:.3f}, {cz:.3f})")

    df = pd.DataFrame({
        "pdb_id":   [PDBID],
        "center_x": [cx],
        "center_y": [cy],
        "center_z": [cz],
    })
    out_csv = f"{PDBID}_out.csv"
    df.to_csv(out_csv, index=True)
    print(f"  Written    : {out_csv}")


if __name__ == "__main__":
    main()
