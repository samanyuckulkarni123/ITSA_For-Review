#!/usr/bin/env python3
"""Tokenize every domain listed in the benchmark manifest.

The output is the five-column TSV consumed directly by ``itsa.cpp``.  Domain
order and metadata are taken from the CSV manifest; PDB filenames are the SCOPe
domain identifiers in the ``sid`` column.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

from tokenizer import tokenize


REQUIRED_COLUMNS = ("sid", "family_id", "pdb_id", "sccs")
OUTPUT_COLUMNS = (*REQUIRED_COLUMNS, "encoded_tokens")


def encode(tokens) -> str:
    """Return the compact token representation expected by itsa.cpp."""
    return ";".join(
        token.primary
        + token.secondary_structure
        + "".join(sorted(set(token.interactions[1:])))
        for token in tokens
    )


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("manifest contains no domains")
    sids = [row["sid"].strip() for row in rows]
    duplicates = sorted({sid for sid in sids if sids.count(sid) > 1})
    if duplicates:
        raise ValueError(f"manifest contains duplicate sid values: {duplicates[:5]}")
    return rows


def write_evidence(directory: Path, sid: str, tokens, interactions) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{sid}_tokens.jsonl").open("w") as handle:
        for token in tokens:
            handle.write(json.dumps(token.as_dict(), sort_keys=True) + "\n")
    with (directory / f"{sid}_interactions.jsonl").open("w") as handle:
        for interaction in interactions:
            handle.write(json.dumps(interaction, sort_keys=True) + "\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Tokenize every SCOPe domain listed in benchmark_manifest.csv"
    )
    parser.add_argument("--manifest", type=Path, default=root / "Data/benchmark_manifest.csv")
    parser.add_argument("--pdb-dir", type=Path, default=root / "Data/pdb")
    parser.add_argument(
        "--dssp-context-dir",
        type=Path,
        default=root / "Data/source_pdb",
        help="source PDB directory used for DSSP before assignments are mapped to each domain",
    )
    parser.add_argument("--output", type=Path, default=root / "Data/scoring_input.tsv")
    parser.add_argument("--dssp", default="mkdssp", help="DSSP executable name or path")
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        help="optional directory for per-domain token and interaction JSONL files",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing output TSV")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        parser.error(f"output already exists: {args.output} (pass --force to replace it)")

    rows = read_manifest(args.manifest)
    missing_pdbs = [row["sid"] for row in rows if not (args.pdb_dir / f"{row['sid']}.pdb").is_file()]
    if missing_pdbs:
        raise FileNotFoundError(
            f"missing {len(missing_pdbs)} domain PDB files; first: {missing_pdbs[:5]}"
        )
    missing_contexts = sorted(
        {
            row["pdb_id"].strip().upper()
            for row in rows
            if not (args.dssp_context_dir / f"{row['pdb_id'].strip().upper()}.pdb").is_file()
        }
    )
    if missing_contexts:
        raise FileNotFoundError(
            f"missing {len(missing_contexts)} DSSP context PDB files; first: {missing_contexts[:5]}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=args.output.name + ".", suffix=".tmp", dir=args.output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for index, row in enumerate(rows, 1):
                sid = row["sid"].strip()
                domain_pdb = args.pdb_dir / f"{sid}.pdb"
                context_pdb = args.dssp_context_dir / f"{row['pdb_id'].strip().upper()}.pdb"
                # Contacts are detected within the extracted SCOPe domain. DSSP is
                # evaluated in the original PDB context and mapped onto that domain.
                tokens, interactions = tokenize(domain_pdb, args.dssp, context_pdb)
                if not tokens:
                    raise ValueError(f"tokenization returned no residues for {sid}")
                writer.writerow(
                    {
                        "sid": sid,
                        "family_id": row["family_id"].strip(),
                        "pdb_id": row["pdb_id"].strip(),
                        "sccs": row["sccs"].strip(),
                        "encoded_tokens": encode(tokens),
                    }
                )
                if args.evidence_dir:
                    write_evidence(args.evidence_dir, sid, tokens, interactions)
                print(f"[{index:03d}/{len(rows):03d}] {sid}: {len(tokens)} residues", file=sys.stderr)
        temporary.replace(args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "domains": len(rows),
                "output": str(args.output),
                "evidence_dir": str(args.evidence_dir) if args.evidence_dir else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
