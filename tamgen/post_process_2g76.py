"""
post_process_2g76.py

Post-processes TamGen generation output for PHGDH 2G76.

Key differences from the PX16 version:
  - No fluorine-atom filter (we have no reference ligand to inherit that bias).
  - Added Lipinski / drug-likeness pre-filter to keep leads tractable.
  - Otherwise the same logic: validate SMILES, remove fragments, filter fused
    rings, fix C(=N)O amidine → amide artefacts, dump TSV + pkl.

Output:
    2g76_nonvae_flatten.tsv
    2g76_vae_flatten.tsv
    2g76_nonvae.pkl
    2g76_vae.pkl
"""

from glob import glob
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdchem import BondType
from collections import defaultdict
from tqdm import tqdm
import pickle as pkl
import numpy as np

RDLogger.DisableLog("rdApp.*")

# ── Configuration ─────────────────────────────────────────────────────────────
PDB_IDX        = "2g76"
MAX_FUSED_RINGS = 3      # same as PX16 post-processing
MAX_RING_COUNT  = 5      # ring count upper bound (PX16 used 6)

# Loose Lipinski-style filter — keeps lead-like molecules
MW_MAX   = 600
LOGP_MAX = 6
HBD_MAX  = 7
HBA_MAX  = 12
# ─────────────────────────────────────────────────────────────────────────────


# ── Helpers (carried over from post_process_phgdh_px16.py) ───────────────────

def count_fused_rings(mol):
    ri    = mol.GetRingInfo()
    rings = list(ri.AtomRings())
    ring_connections = [set() for _ in range(len(rings))]
    for i, ring1 in enumerate(rings):
        for j in range(i + 1, len(rings)):
            ring2 = rings[j]
            if len(set(ring1).intersection(set(ring2))) >= 2:
                ring_connections[i].add(j)
                ring_connections[j].add(i)
    visited = [False] * len(rings)
    def dfs(v):
        visited[v] = True
        size = 1
        for w in ring_connections[v]:
            if not visited[w]:
                size += dfs(w)
        return size
    max_fused = 0
    for i in range(len(rings)):
        if not visited[i]:
            max_fused = max(max_fused, dfs(i))
    return max_fused


_METALS = set([
    "Li","Be","Na","Mg","Al","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co",
    "Ni","Cu","Zn","Ga","Ge","As","Se","Rb","Sr","Y","Zr","Nb","Mo","Tc",
    "Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","Cs","Ba","La","Ce","Pr",
    "Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta",
    "W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Th","Pa","U",
])

def has_metal(mol):
    return any(a.GetSymbol() in _METALS for a in mol.GetAtoms())


def passes_lipinski(mol):
    mw   = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    return mw <= MW_MAX and logp <= LOGP_MAX and hbd <= HBD_MAX and hba <= HBA_MAX


def fix_CNO(smi):
    """Fix C(=N)O amidine → amide artefact introduced by TamGen."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    rw_mol   = Chem.RWMol(mol)
    substruct = Chem.MolFromSmiles("C(=N)O")
    matches  = rw_mol.GetSubstructMatches(substruct)
    if not matches:
        return smi
    for match in matches:
        idx1, idx2, idx3 = match
        rw_mol.GetBondBetweenAtoms(idx1, idx2).SetBondType(BondType.SINGLE)
        rw_mol.GetBondBetweenAtoms(idx1, idx3).SetBondType(BondType.DOUBLE)
    new_mol = rw_mol.GetMol()
    x = Chem.MolToSmiles(new_mol)
    m = Chem.MolFromSmiles(x)
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def loader_fn(fn, remove_frag=True):
    """Load and validate SMILES from a TamGen output file."""
    total, success = 0, 0
    raw_lines = []
    with open(fn) as fr:
        for line in fr:
            if not line.strip().startswith("H-"):
                continue
            if remove_frag and "*" in line:
                continue
            total += 1
            segs = line.strip().split("\t")
            if len(segs) < 3:
                continue
            if segs[2].count("(") != segs[2].count(")"):
                continue
            raw_lines.append((segs[2], float(segs[1])))

    smiles_scores = defaultdict(list)
    for smi_raw, score in tqdm(raw_lines, total=len(raw_lines), leave=False):
        smi = smi_raw.replace(" ", "").strip()
        if smi.startswith("[generation]"):
            smi = smi.replace("[generation]", "")
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        if count_fused_rings(m) > MAX_FUSED_RINGS:
            continue
        if has_metal(m):
            continue
        if not passes_lipinski(m):
            continue
        s = Chem.MolToSmiles(m)
        smiles_scores[s].append(score)
        success += 1

    return smiles_scores, success, total


# ── Main ──────────────────────────────────────────────────────────────────────

for prefix in ["nonvae", "vae"]:
    print(f"\n{'='*60}")
    print(f"Processing: {prefix}")
    print(f"{'='*60}")

    all_results = {}
    success_total, total_total = 0, 0

    ff = glob(f"{PDB_IDX}-results/{prefix}*")
    if not ff:
        print(f"  WARNING: no files found matching {PDB_IDX}-results/{prefix}*")
        continue

    for fn in ff:
        A, s, t = loader_fn(fn)
        success_total += s
        total_total   += t
        for k, v in A.items():
            if k in all_results:
                all_results[k].extend(v)
            else:
                all_results[k] = v

    print(f"  Valid / total SMILES: {success_total} / {total_total}")

    # Fix amidine artefacts
    all_results_fix = {}
    for k, v in all_results.items():
        k2 = fix_CNO(k)
        if k2 is None:
            continue
        if k2 in all_results_fix:
            all_results_fix[k2].extend(v)
        else:
            all_results_fix[k2] = v

    # Save full pkl
    pkl_path = f"{PDB_IDX}_{prefix}.pkl"
    with open(pkl_path, "wb") as fw:
        pkl.dump(all_results_fix, fw)
    print(f"  Saved pkl → {pkl_path}  ({len(all_results_fix)} unique SMILES)")

    # Sort by mean score (higher is better)
    DB2 = sorted(all_results_fix.items(), key=lambda x: np.mean(x[1]), reverse=True)

    # Final filters for the flat TSV
    remaining = []
    for smi, scores in DB2:
        if "p" in smi or "P" in smi:   # phosphorus — often unstable
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        ssr = Chem.GetSymmSSSR(m)
        if len(ssr) == 0:              # must have at least one ring
            continue
        if len(ssr) >= MAX_RING_COUNT: # not too many rings
            continue
        remaining.append((smi, np.mean(scores)))

    tsv_path = f"{PDB_IDX}_{prefix}_flatten.tsv"
    with open(tsv_path, "w") as fw:
        for smi, score in remaining:
            print(f"{smi}\t{score:.6f}", file=fw)
    print(f"  Saved TSV → {tsv_path}  ({len(remaining)} molecules)")

print("\nPost-processing complete.")
