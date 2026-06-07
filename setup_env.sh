#!/bin/zsh
# Source this to activate the LIMO/PHGDH environment.
# Usage: source setup_env.sh

source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate cse203_limo

# RDKit + PyTorch both ship OpenMP — allow duplicate libomp on import
export KMP_DUPLICATE_LIB_OK=TRUE

# Tools (autogrid4, autodock_gpu_128wi) live in repo-local bin/
ROOT="$( cd "$( dirname "${(%):-%x}" )" && pwd )"
export PATH="$ROOT/bin:$PATH"

echo "cse203_limo activated."
echo "  Python:       $(python -V 2>&1)"
echo "  PyTorch MPS:  $(python -c 'import torch; print(torch.backends.mps.is_available())')"
echo "  autogrid4:    $(which autogrid4)"
echo "  autodock_gpu: $(which autodock_gpu_128wi)"
echo ""
echo "To run LIMO:"
echo "  cd $ROOT/limo"
echo "  python generate_molecules.py --prop binding_affinity \\"
echo "      --autodock_executable ../bin/autodock_gpu_128wi \\"
echo "      --protein_file 2g76/2g76.maps.fld"
