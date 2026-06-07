# LIMO on PHGDH 2G76 — Apple Silicon setup

Local setup of [rose-stl-lab/limo](https://github.com/rose-stl-lab/limo) (Eckmann et al., ICML 2022) targeting the **DNA-binding (HHTH) domain of human PHGDH** in PDB [2G76](https://www.rcsb.org/structure/2G76), for designing small-molecule inhibitors of PHGDH's transcriptional function in Alzheimer's pathology.

## Scientific context

PHGDH (D-3-phosphoglycerate dehydrogenase, UniProt O43175) is canonically a serine biosynthesis enzyme. Chen et al. ([*Cell* 2025](https://doi.org/10.1016/j.cell.2025.03.045), [PMC12204802](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12204802/)) identified an uncharacterized DNA-binding role: PHGDH binds chromatin via a **helix-helix-turn-helix (HHTH) subdomain at residues 103–165**, and drives transcription of `IKKα` and `HMGB1` in astrocytes, suppressing autophagy and accelerating amyloid pathology. They showed that the small molecule **NCT-503** inhibits this transcriptional function in a way that improves AD phenotypes in mouse and brain-organoid models.

The goal here is to use LIMO's VAE + property-predictor + AutoDock-GPU pipeline to generate novel small molecules that bind the HHTH region of 2G76 and disrupt PHGDH–DNA interaction.

## Target

- **Protein:** Human PHGDH catalytic core (chains A/B, residues 5–306 modeled, 1.70 Å).
- **Grid center:** `(14.547, 25.866, 14.134)` — sidechain centroid of 2G76 chain A **R162**, which corresponds to canonical PHGDH **R163** (the residue Chen 2025 showed via R163Q mutation is essential for DNA binding). 2G76 numbering is offset by one residue from the Chen/UniProt numbering — verified by sequence inspection.
- **Box:** 60 × 60 × 60 points at 0.375 Å spacing = 22.5 Å cube, covering the HHTH motif.

## Directory layout

```
limo/                          (this directory)
├── README.md
├── setup_env.sh               source this to activate the conda env + PATH
├── bin/                       arm64-native binaries (symlinks into tools/)
│   ├── autogrid4              4.2.8, built from ccsb-scripps/AutoGrid
│   ├── autodock_gpu_128wi     v1.6, OpenCL backend on Apple M3 GPU
│   ├── AD4_parameters.dat     AD4 parameter file (needed by autogrid)
│   └── AD4.1_bound.dat
├── limo/                      cloned rose-stl-lab/limo, with local patches
│   ├── 2g76/                  PHGDH HHTH grid (see below)
│   ├── dm.pkl                 zinc250k preprocessed VAE vocab/dataset
│   ├── vae.pt, property_models/  pre-trained LIMO models
│   ├── utils.py               patched for MPS + non-CUDA dock device count
│   ├── generate_molecules.py  patched for MPS + map_location loads
│   └── …
└── tools/
    ├── AutoGrid/              source clone + build artifacts
    └── AutoDock-GPU/          source clone + build artifacts
```

### `limo/2g76/` (PHGDH HHTH grid)

| File | Description |
|---|---|
| `2g76.pdb` | Raw PDB from RCSB |
| `2g76_chainA.pdb` | Protein-only chain A (HETATM + waters stripped) |
| `2g76.pdbqt` | Receptor, prepared with Meeko `mk_prepare_receptor.py` |
| `2g76.gpf` | Grid parameter file (`gridcenter 14.547 25.866 14.134`, 60³ pts, 0.375 Å) |
| `2g76.maps.fld` | AVS field file — pass this to LIMO's `--protein_file` |
| `2g76.*.map` | 13 atom-type affinity maps + electrostatic + desolvation |
| `ligand_ref.pdbqt` | D-malate (MLT) extracted from 2G76, smoke-test ligand |
| `dock_smoke.dlg` | Standalone-dock sanity output (best -4.93 kcal/mol on Apple M3 Max) |

## System requirements (matches the machine this was set up on)

- macOS 26 on Apple Silicon (M3 Max here; should work on M1/M2)
- Homebrew with `autoconf`, `automake`, `libtool`, `libomp`
- Xcode command line tools (provides Apple's OpenCL headers under the SDK)
- Miniconda

## Run

```bash
cd /Users/yifeiding/projects/school/BENG203/limo
source setup_env.sh

cd limo
python generate_molecules.py \
    --autodock_executable ../bin/autodock_gpu_128wi \
    --protein_file 2g76/2g76.maps.fld
```

Defaults: `--prop multi_objective_binding_affinity --num_mols 10000 --optim_steps 10 --top_k 3 --sa_cutoff 5.5 --qed_cutoff 0.4`. A 5-molecule single-step smoke run completes in seconds.

To fine-tune a single hit:
```bash
python fine-tune.py "<SMILES>"
```

## Apple Silicon GPU strategy

- **PyTorch (VAE, predictors, latent-space gradient optimization) → Metal via MPS.** `torch.backends.mps.is_available()` returns True; `utils.py` and `generate_molecules.py` were patched to select `mps` when CUDA is unavailable.
- **AutoDock-GPU → OpenCL.** AutoDock-GPU has no Metal backend, but Apple's OpenCL framework exposes the M3 GPU as an OpenCL device — `make DEVICE=GPU NUMWI=128 OVERLAP=OFF` builds an arm64 binary that runs on the M3 GPU. (`OVERLAP=OFF` disables the OpenMP pipeline because Apple clang needs libomp headers wired up; correctness is unaffected, only batch overlap is lost.)

## Patches made to the LIMO source

Necessary on Mac / non-CUDA setups:

- **`limo/utils.py`:**
  - `device` now picks `cuda → mps → cpu` in that order.
  - Added `_NUM_DOCK_DEVICES = max(1, torch.cuda.device_count())` and routed all docking subprocess loops through it. Without this, `torch.cuda.device_count() == 0` on Mac silently skips ligand creation and docking.
  - Changed batch AutoDock `-N ../../outs/` to `-N outs/`. The relative path was broken for this layout.
- **`limo/generate_molecules.py`:**
  - Same device-selection logic.
  - `torch.load(..., map_location=device, weights_only=True)` on both the VAE and property-predictor checkpoints — the upstream checkpoints were serialized on CUDA.

## How this was built

Background, in case you need to rebuild on a different machine:

### Conda environment `cse203_limo`
```bash
conda create -n cse203_limo python=3.10 -y
conda activate cse203_limo
conda install -c conda-forge -c bioconda rdkit openbabel numpy scipy tqdm -y
pip install "torch>=2.1" "pytorch-lightning>=2.1" selfies meeko gemmi
```

### autogrid4 from source (CCSB Scripps download site was down at setup time)
```bash
cd tools
git clone https://github.com/ccsb-scripps/AutoGrid.git
cd AutoGrid
brew install autoconf automake libtool   # only if not already installed
autoreconf -fvi
./configure && make
# Produces ./autogrid4 (arm64 native, ~157 KB)
```

### AutoDock-GPU with OpenCL on Apple M3
```bash
cd tools
git clone https://github.com/ccsb-scripps/AutoDock-GPU.git
cd AutoDock-GPU
brew install libomp   # only if not already installed
SDK=$(xcrun --show-sdk-path)
export GPU_INCLUDE_PATH="$SDK/System/Library/Frameworks/OpenCL.framework/Headers"
export GPU_LIBRARY_PATH="$SDK/System/Library/Frameworks/OpenCL.framework"
make DEVICE=GPU NUMWI=128 OVERLAP=OFF
# Produces ./bin/autodock_gpu_128wi (arm64 native)
```

### Receptor + grid for 2G76
```bash
cd limo && mkdir -p 2g76 && cd 2g76
curl -L -o 2g76.pdb https://files.rcsb.org/download/2G76.pdb
grep "^ATOM" 2g76.pdb | awk 'substr($0,22,1)=="A"' > 2g76_chainA.pdb
mk_prepare_receptor.py --read_pdb 2g76_chainA.pdb -p 2g76.pdbqt --allow_bad_res --default_altloc A
# Compute centroid of R162 sidechain → (14.547, 25.866, 14.134)
# Write 2g76.gpf (see file in this dir)
../../bin/autogrid4 -p 2g76.gpf -l 2g76.glg
```

### Bootstrap the VAE
```bash
cd limo
python preprocess_data.py --smiles zinc250k.smi   # builds dm.pkl
```

## Caveats

- **The 2G76 ↔ Chen 2025 numbering offset.** R162 in 2G76 = R163 in canonical PHGDH (and in Chen 2025's text). Always verify the residue identity by inspecting the sequence around the position before re-centering on a different residue.
- **Whether 2G76 actually captures the DNA-binding interface.** 2G76 is the catalytic core crystallized with NAD + D-malate. The HHTH motif (residues 103–165) is *modeled* in 2G76, but the structural state of the surface as a DNA-binder may differ from what's visible in this catalytic-state crystal. If results disappoint, an AlphaFold model of full-length PHGDH (or the AF model Chen 2025 used) is the natural next target.
- **`KMP_DUPLICATE_LIB_OK=TRUE` is required** because RDKit and PyTorch both bundle their own libomp; `setup_env.sh` sets this for you.
- **No `--top_k` enforced for smoke tests.** A 5-molecule run can produce molecules whose predicted Kd looks unrealistic — that's expected for ungenerated/random latents and one optimization step.

## References

- Chen et al., "Transcriptional regulation by PHGDH drives amyloid pathology in Alzheimer's disease," *Cell* 188(13):3513–3529 (2025). [DOI](https://doi.org/10.1016/j.cell.2025.03.045)
- Eckmann et al., "LIMO: Latent Inceptionism for Targeted Molecule Generation," ICML 2022. [GitHub](https://github.com/rose-stl-lab/limo)
- Santamaría-García et al., 2006. Crystal structure of human PHGDH, PDB [2G76](https://www.rcsb.org/structure/2G76).
