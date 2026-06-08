#!/usr/bin/env bash
# build_bindata.sh
#
# Builds the binary dataset for TamGen generation.
# Uses two pocket radii (10 Å and 12 Å) to capture the DNA-binding domain.
#
# Prerequisites:
#   - 2g76_out.csv         (produced by get_binding_site_center.py)
#   - seed_cmpd_2g76.txt   (produced by generate_seed_smiles.py)
#   - 2g76.pdb             (downloaded from RCSB)
#
# Run from inside the TamGen/customized_example/ directory (same as before).

set -euo pipefail

PDBID="2g76"
SCRIPT="../scripts/build_data/prepare_pdb_ids_center_scaffold.py"

for thr in "10" "12"; do
    echo "Building binary data for pocket radius ${thr} Å ..."
    python "$SCRIPT" \
        "${PDBID}_out.csv" test \
        -o "${PDBID}-bin/t${thr}" \
        -t "$thr" \
        --scaffold-file "seed_cmpd_${PDBID}.txt" \
        --pdb-path "./"
    echo "  Done → ${PDBID}-bin/t${thr}"
done

echo ""
echo "All binary data built successfully."
