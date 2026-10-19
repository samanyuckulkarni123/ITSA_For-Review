#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
BIN=${1:-foldseek}
PDB="$ROOT/Data/pdb"
OUT="$ROOT/Results/Foldseek/Full_Heldout"
mkdir -p "$OUT"
"$BIN" easy-search "$PDB" "$PDB" "$OUT/foldseek_exhaustive_raw.tsv" "$OUT/tmp" --exhaustive-search 1 -e 1e30 --max-seqs 100000 --format-output query,target,prob,alntmscore,qtmscore,ttmscore,bits,evalue
