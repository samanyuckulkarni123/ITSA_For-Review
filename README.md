# ITSA refined benchmark submission package

This directory contains the source code, benchmark domains, final evaluation outputs, figures, and tables used in the revised ITSA study.

## Contents

- `Code/itsa.cpp`: standalone ITSA local-alignment scorer and final three-term model.
- `Code/tokenizer.py`: converts one extracted protein domain into interaction tokens.
- `Code/batch_tokenize.py`: reads the benchmark manifest and tokenizes all 222 domains into the C++ scoring input.
- `Code/itsa.py`: reproduces nested leave-one-superfamily-out model selection and evaluation.
- `Code/TMalign.cpp`: TM-align source used for the structural baseline.
- `Code/Foldseek/`: Foldseek version record and exhaustive-search command. Install the listed Foldseek release separately.
- `Code/generate_paper_figures.py`: regenerates manuscript figures and summary tables.
- `Data/benchmark_manifest.csv`: one row per benchmark domain with its SCOPe assignments and extraction metadata.
- `Data/pdb/`: the 222 extracted SCOPe domain structures supplied to every comparison method.
- `Data/source_pdb/`: the 211 source structures used as DSSP context before assignments were mapped to the extracted domains.
- `Data/reference/`: official SCOPe 2.08 and ASTRAL 40% reference files.
- `Data/scoring_input.tsv`: encoded interaction-token input for the C++ scorer.
- `Data/itsa_feature_matrix.csv`: pairwise feature matrix used by nested model evaluation.
- `Results/`: final ITSA, exhaustive Foldseek, and TM-align pair scores and evaluation outputs.
- `Figures/` and `Tables/`: manuscript-ready derived outputs.

## Reproduce the ITSA scores

From the package root:

```bash
python Code/batch_tokenize.py --output Data/scoring_input.tsv --force
g++ -O3 -std=c++17 Code/itsa.cpp -o itsa
./itsa Data/scoring_input.tsv pair_scores.csv
```

The batch tokenizer uses `Data/benchmark_manifest.csv` as the authoritative domain list. It reads each extracted domain from `Data/pdb/<sid>.pdb` and selects its DSSP context from `Data/source_pdb/<pdb_id>.pdb`. Contacts are calculated within the extracted domain, while DSSP is calculated in the source PDB context and mapped back to the retained domain residues. The script fails if a listed structure is missing or DSSP omits a retained residue. Add `--evidence-dir token_evidence` to save residue-level token and interaction JSONL files; these are optional and are not created during the normal compact workflow.

The standalone executable writes raw alignment scores and the final three-term score. The embedded all-data coefficients are provided for application to external structures. Performance on this benchmark must be taken from the held-out predictions in `Results/ITSA/Optimized_Nested/`.

To tokenize a structure:

```bash
python Code/tokenizer.py Data/pdb/d1asha_.pdb --dssp-context Data/pdb/d1asha_.pdb --tokens-jsonl tokens.jsonl --encoded-output encoded_tokens.txt
```

DSSP 4.5.8 must be installed and available as `mkdssp`, or its path can be supplied with `--dssp`. Python dependencies, including Biopython 1.88 or later, are listed in `requirements.txt`.

## Reproduce nested evaluation

Install the Python dependencies and run:

```bash
python -m pip install -r requirements.txt
python Code/itsa.py
```

The evaluation holds out one SCOPe superfamily at a time. Feature selection and regularization are performed inside each training split. The resulting out-of-fold predictions are stored in `Results/ITSA/Optimized_Nested/directional_oof_scores.csv`.

## Structural baselines

Compile TM-align from the included source according to its upstream instructions. The final TM-align pair scores are in `Results/TM-align/Full_Heldout/tmalign_pair_scores_clean.csv`.

Install the Foldseek release recorded in `Code/Foldseek/VERSION.txt`, then run:

```bash
bash Code/Foldseek/run_foldseek.sh /path/to/foldseek
```

The search uses exhaustive mode. The submitted parsed scores contain all 24,520 eligible pairs.

## Figures and tables

```bash
python Code/generate_paper_figures.py
```

This regenerates the manuscript figures and CSV tables from the packaged results.
