# TamGen — PHGDH 2G76 DNA-Binding Domain Campaign
### One-week class project: small-molecule binders to AA 103–165

---

## Overview of Changes vs. the PX16 Pipeline

| Step | PX16 (original) | 2G76 (this project) |
|---|---|---|
| Binding site center | Centroid of HETATM ligand (F-filtered) | Centroid of Cα atoms, residues 103–165 |
| Seed SMILES | PX16's SMILES from PubChem | Curated fragment library (10 drug-like scaffolds) |
| Post-processing filters | Fluorine check, ≤3 fused rings | Lipinski pre-filter added; no F check |

---

## File Structure

```
phgdh_2g76_project/           ← copy this entire folder to TSCC
├── get_binding_site_center.py
├── generate_seed_smiles.py
├── build_bindata.sh
├── generator-2g76.sh
└── post_process_2g76.py
```

Place these files inside `TamGen/customized_example/` on TSCC alongside your `2g76.pdb`.

---

## Step-by-Step Instructions

### 0. Prerequisites — do these once
```bash
# On TSCC, activate your TamGen conda environment
conda activate TamGen
cd ~/TamGen/customized_example
```

### 1. Download the PDB structure

```bash
curl https://files.rcsb.org/download/2G76.pdb > 2g76.pdb
```

> **Check the chain ID.** Open `2g76.pdb` and confirm that the protein chain
> is `A`. If it differs, edit `get_binding_site_center.py` and change
> `CHAIN = "A"` to the correct letter.
>
> Also verify residue numbering — search for `  103 ` in the file:
> ```bash
> grep "^ATOM" 2g76.pdb | awk '$6 == 103' | head -3
> ```
> If nothing is returned, the residue numbers differ from UniProt numbering;
> adjust `RES_START` / `RES_END` accordingly.

### 2. Compute the binding site center

```bash
python get_binding_site_center.py
```

**Output:** `2g76_out.csv`  
**What it does:** Takes the geometric centroid of all heavy atoms in chain A,
residues 103–165. This defines the pocket center for TamGen.

Expected output:
```
Reading 2g76.pdb ...
  Atoms used : ~500
  Center     : (x.xxx, y.yyy, z.zzz)
  Written    : 2g76_out.csv
```

### 3. Generate seed SMILES

```bash
python generate_seed_smiles.py
```

**Output:** `seed_cmpd_2g76.txt`  
**What it does:** Takes 10 curated drug-like scaffold SMILES and creates ~200
augmented variants (random atom-renumbering) to give TamGen diverse starting
points. There is no reference ligand, so we use general surface-binder motifs.

> **Optional:** If you find a literature compound or docking hit that binds
> the DNA-binding domain, open `generate_seed_smiles.py`, set `USE_CUSTOM = True`,
> and paste its SMILES into `CUSTOM_SMILES`. Then re-run this step.

### 4. Build binary training data

```bash
bash build_bindata.sh
```

**Output:** `2g76-bin/t10/` and `2g76-bin/t12/`  
**What it does:** Calls TamGen's `prepare_pdb_ids_center_scaffold.py` with two
pocket radii (10 Å and 12 Å) to capture the domain at different granularities.

### 5. Run TamGen generation

```bash
# Submit as a SLURM job (recommended on TSCC):
sbatch generator-2g76.sh

# Or run directly if you have an interactive GPU session:
bash generator-2g76.sh
```

**Before submitting**, edit the SLURM header in `generator-2g76.sh`:
```bash
#SBATCH --account=YOUR_ACCOUNT    # ← replace with your actual allocation
#SBATCH --partition=gpu-shared    # ← or gpu-hotel, etc. — check with your PI
```

**Output:** `2g76-results/` containing 8 output files (2 radii × 2 beta values × 2 modes).

This step takes ~1–3 hours on a single GPU.

### 6. Post-process results

```bash
python post_process_2g76.py
```

**Output:**
- `2g76_nonvae.pkl` / `2g76_vae.pkl` — full dictionaries of SMILES → scores
- `2g76_nonvae_flatten.tsv` / `2g76_vae_flatten.tsv` — ranked, filtered SMILES

The TSV files are your final results, sorted by TamGen's average normalized
log probability (higher = model is more confident the compound fits the pocket).

---

## Interpreting Results

The TSV columns are:
```
SMILES    average_score
```

Higher scores indicate that TamGen's model finds the compound more compatible
with the binding site geometry. These are **not** binding affinities — they are
generation scores. For a one-week project, the top 20–50 hits are reasonable
candidates to:
- Visualize in PyMOL or UCSF Chimera
- Run a quick Glide/AutoDock-Vina docking to get an estimated ΔG
- Check for known PHGDH inhibitor pharmacophores in literature

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `No atoms found` in step 2 | Wrong chain ID or residue range | Check PDB with `grep "^ATOM" 2g76.pdb \| head` |
| `prepare_pdb_ids_center_scaffold.py` not found in step 4 | Not running from `customized_example/` | `cd ~/TamGen/customized_example` |
| Generation output files are empty | GPU not allocated / wrong partition | Check `squeue -u $USER` and SLURM logs |
| `No files found` in post-processing | Results folder path mismatch | Ensure `generator-2g76.sh` ran from `customized_example/` |
| Very few molecules survive post-processing | Generation didn't converge | Lower `MAX_FUSED_RINGS` or relax Lipinski thresholds in `post_process_2g76.py` |

---

## Quick Reference — Full Command Sequence

```bash
conda activate TamGen
cd ~/TamGen/customized_example

wget https://files.rcsb.org/download/2G76.pdb -O 2g76.pdb

python get_binding_site_center.py
python generate_seed_smiles.py
bash build_bindata.sh
sbatch generator-2g76.sh         # wait for job to finish

python post_process_2g76.py
```
