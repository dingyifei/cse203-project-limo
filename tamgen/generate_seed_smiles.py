"""
generate_seed_smiles.py

Since there is no pre-docked reference ligand for 2G76's DNA-binding domain,
this script provides two modes:

  MODE 1 (default) — use a curated set of small DNA-groove-binder / PHGDH-
                      domain scaffold fragments as seeds, then augment them
                      with random atom-renumbering (same augmentation as the
                      original augment_smiles.py).

  MODE 2           — if you later obtain a reference compound SMILES (e.g.
                      from a docking hit or literature), paste it into
                      CUSTOM_SMILES below and set USE_CUSTOM = True.

Output:
    seed_cmpd_2g76.txt  — one SMILES per line, used by build_bindata.sh
"""

from rdkit import Chem
import random

# ── Configuration ────────────────────────────────────────────────────────────
PDBID       = "2g76"
USE_CUSTOM  = False   # set True and fill CUSTOM_SMILES to override seed set
CUSTOM_SMILES = [
    # paste your own SMILES here, e.g.:
    # "CCc1ccc(NC(=O)c2cccc(Cl)c2)cc1",
]
N_AUGMENT   = 20      # random renumberings per seed molecule
# ─────────────────────────────────────────────────────────────────────────────

# Curated seed scaffolds — small, drug-like fragments with
# functional groups known to interact with protein surfaces
# (H-bond donors/acceptors, aromatic rings, amide bonds).
# Chosen to be structurally diverse and within Lipinski space.
DEFAULT_SEEDS = [
    # Aminopyrimidine — common kinase/PPI hinge-binder motif
    "Nc1ncnc2c1cccc2",
    # Benzimidazole — groove-compatible flat aromatic
    "c1ccc2[nH]cnc2c1",
    # Sulfonamide-aniline — H-bond donor/acceptor pair
    "NS(=O)(=O)c1ccc(N)cc1",
    # Piperazine-benzoyl — flexible with H-bond acceptors
    "O=C(c1ccccc1)N1CCNCC1",
    # Indole — flat aromatic, good stacking scaffold
    "c1ccc2[nH]ccc2c1",
    # Urea linker scaffold — two H-bond donors
    "O=C(Nc1ccccc1)Nc1ccccc1",
    # Morpholine-pyridine — polar, soluble fragment
    "c1cc(N2CCOCC2)ccn1",
    # Thienopyrimidine — bioisostere of aminopurine
    "Nc1ncnc2ccsc12",
    # Carboxamide-thiazole — compact H-bond pair
    "NC(=O)c1csc(N)n1",
    # Acylaminopyrazole — PPI inhibitor motif
    "O=C(Nc1ccc(F)cc1)c1cn[nH]c1",
]


def augment_one(smiles: str, n: int = N_AUGMENT) -> set:
    """Return a set of valid augmented SMILES for a single input SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  WARNING: could not parse '{smiles}', skipping.")
        return set()

    can = Chem.MolToSmiles(mol)
    out = {can, smiles}

    remapping = list(range(mol.GetNumAtoms()))
    for _ in range(n):
        random.shuffle(remapping)
        new_mol  = Chem.RenumberAtoms(mol, remapping)
        new_smi  = Chem.MolToSmiles(new_mol, isomericSmiles=True, canonical=False)
        check    = Chem.MolFromSmiles(new_smi)
        if check is None:
            continue
        if Chem.MolToSmiles(check) == can:   # valid augmentation
            out.add(new_smi)
    return out


def main():
    seeds = CUSTOM_SMILES if USE_CUSTOM else DEFAULT_SEEDS

    if not seeds:
        raise ValueError("No seed SMILES defined. Either keep DEFAULT_SEEDS "
                         "or set USE_CUSTOM=True and fill CUSTOM_SMILES.")

    print(f"Augmenting {len(seeds)} seed molecule(s) ...")
    all_smiles = set()
    for smi in seeds:
        aug = augment_one(smi)
        print(f"  {smi[:50]:<50s} → {len(aug):3d} variants")
        all_smiles |= aug

    out_file = f"seed_cmpd_{PDBID}.txt"
    with open(out_file, "w", encoding="utf-8") as fw:
        for s in sorted(all_smiles):
            print(s, file=fw)

    print(f"\nTotal seed SMILES written: {len(all_smiles)}")
    print(f"Output file             : {out_file}")


if __name__ == "__main__":
    main()
