"""Standalone ITSA interaction-token generator.

Requires Biopython, NumPy, and mkdssp. Produces an auditable JSONL token file
and the compact token string consumed by itsa.cpp.
"""
from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from Bio.PDB import DSSP, NeighborSearch, PDBParser
from Bio.PDB.Polypeptide import is_aa
import numpy as np

class DSSPError(RuntimeError):
    pass

LETTERS = {"hydrogen_bond": "H", "disulfide": "D", "ionic": "I", "vdw": "V", "pi_stack": "P", "hydrophobic": "Y"}
PRIORITY = {name: index for index, name in enumerate(["disulfide", "ionic", "hydrogen_bond", "pi_stack", "hydrophobic", "vdw"])}
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "PRO", "TRP", "GLY"}
AROMATIC = {"PHE", "TYR", "TRP", "HIS"}
POSITIVE = {"LYS", "ARG", "HIS"}
NEGATIVE = {"ASP", "GLU"}
HBOND_ATOMS = {
    "SER": {"OG"}, "THR": {"OG1"}, "ASN": {"OD1", "ND2"}, "GLN": {"OE1", "NE2"},
    "TYR": {"OH"}, "CYS": {"SG"}, "HIS": {"ND1", "NE2"}, "LYS": {"NZ"},
    "ARG": {"NE", "NH1", "NH2"}, "ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"},
}
IONIC_ATOMS = {name: HBOND_ATOMS[name] for name in POSITIVE | NEGATIVE}
HYDROPHOBIC_ATOMS = {
    "ALA": {"CB"}, "VAL": {"CB", "CG1", "CG2"}, "LEU": {"CB", "CG", "CD1", "CD2"},
    "ILE": {"CB", "CG1", "CG2", "CD1"}, "MET": {"CB", "CG", "SD", "CE"},
    "PHE": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": {"CB", "CG", "CD"}, "TRP": {"CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "GLY": {"CA"},
}
RING_ATOMS = {
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
}
BACKBONE = {"N", "CA", "C", "O", "OXT"}


@dataclass(frozen=True)
class Token:
    ordinal: int
    chain: str
    author_residue: str
    interactions: tuple[str, ...]
    secondary_structure: str

    @property
    def primary(self) -> str:
        return self.interactions[0] if self.interactions else "U"

    def as_dict(self):
        return {"ordinal": self.ordinal, "chain": self.chain, "author_residue": self.author_residue, "interactions": list(self.interactions), "primary": self.primary, "secondary_structure": self.secondary_structure}


def _heavy(residue, names=None, sidechain=False):
    return [a for a in residue if getattr(a, "element", "").strip().upper() != "H" and (names is None or a.name in names) and (not sidechain or a.name not in BACKBONE)]


def _minimum_distance(a, b, names_a=None, names_b=None, sidechain=False):
    aa, bb = _heavy(a, names_a, sidechain), _heavy(b, names_b, sidechain)
    if not aa or not bb:
        return math.inf, None, None
    distance, atom_a, atom_b = min((x - y, x, y) for x in aa for y in bb)
    return float(distance), atom_a.name, atom_b.name


def _ring_geometry(residue):
    atoms = [residue[name] for name in RING_ATOMS.get(residue.resname, []) if name in residue]
    if len(atoms) < 3:
        return None, None
    coordinates = np.asarray([atom.coord for atom in atoms], dtype=float)
    normal = np.cross(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0])
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None, None
    return coordinates.mean(axis=0), normal / norm


def _classify(a, b):
    ra, rb = a.resname, b.resname
    if ra == rb == "CYS" and "SG" in a and "SG" in b:
        d = float(a["SG"] - b["SG"])
        if d <= 2.3:
            return "disulfide", d, "SG", "SG"
    distance, atom_a, atom_b = _minimum_distance(a, b, IONIC_ATOMS.get(ra), IONIC_ATOMS.get(rb))
    if ((ra in POSITIVE and rb in NEGATIVE) or (rb in POSITIVE and ra in NEGATIVE)) and distance <= 4.0:
        return "ionic", distance, atom_a, atom_b
    distance, atom_a, atom_b = _minimum_distance(a, b, HBOND_ATOMS.get(ra), HBOND_ATOMS.get(rb))
    if ra in HBOND_ATOMS and rb in HBOND_ATOMS and distance <= 3.5:
        return "hydrogen_bond", distance, atom_a, atom_b
    if ra in AROMATIC and rb in AROMATIC:
        center_a, normal_a = _ring_geometry(a)
        center_b, normal_b = _ring_geometry(b)
        if center_a is not None and center_b is not None:
            distance = float(np.linalg.norm(center_a - center_b))
            angle = math.degrees(math.acos(float(np.clip(abs(np.dot(normal_a, normal_b)), -1.0, 1.0))))
            if distance <= 5.5 and angle <= 35.0:
                return "pi_stack", distance, "ring_centroid", "ring_centroid"
    distance, atom_a, atom_b = _minimum_distance(a, b, HYDROPHOBIC_ATOMS.get(ra), HYDROPHOBIC_ATOMS.get(rb), sidechain=True)
    if ra in HYDROPHOBIC and rb in HYDROPHOBIC and distance <= 4.8:
        return "hydrophobic", distance, atom_a, atom_b
    distance, atom_a, atom_b = _minimum_distance(a, b, sidechain=True)
    if distance <= 4.5:
        return "vdw", distance, atom_a, atom_b
    return None


def assign_dssp(context_pdb: Path, dssp_executable: str = "mkdssp") -> dict[tuple[str, tuple], str]:
    executable = shutil.which(dssp_executable)
    if executable is None:
        raise DSSPError(f"DSSP executable not found: {dssp_executable}")
    structure = PDBParser(QUIET=True).get_structure("dssp_context", str(context_pdb))
    model = next(structure.get_models(), None)
    if model is None:
        raise DSSPError("Extracted domain has no model")
    try:
        result = DSSP(model, str(context_pdb), dssp=executable, file_type="PDB")
    except Exception as exc:
        raise DSSPError(f"DSSP failed for {context_pdb.name}: {exc}") from exc
    mapping = {}
    for key in result.keys():
        code = result[key][2]
        mapping[key] = "H" if code in {"H", "G", "I"} else "E" if code in {"E", "B"} else "C"
    return mapping


def tokenize(domain_pdb: Path, dssp_executable: str = "mkdssp", dssp_context_pdb: Path | None = None) -> tuple[list[Token], list[dict]]:
    structure = PDBParser(QUIET=True).get_structure("domain", str(domain_pdb))
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(f"Extracted domain must contain exactly one model, found {len(models)}")
    residues = [r for c in models[0] for r in c if is_aa(r, standard=True)]
    dssp = assign_dssp(dssp_context_pdb or domain_pdb, dssp_executable)
    missing = [(r.parent.id, r.id) for r in residues if (r.parent.id, r.id) not in dssp]
    if missing:
        raise DSSPError(f"DSSP omitted {len(missing)} retained residues; first={missing[0]!r}")
    search = NeighborSearch([atom for residue in residues for atom in residue])
    evidence: list[dict] = []
    incident: dict[int, list[tuple[str, float]]] = {id(r): [] for r in residues}
    ordinal = {id(r): i + 1 for i, r in enumerate(residues)}
    seen: set[tuple[int, int]] = set()
    for residue in residues:
        for atom in search.search(residue["CA"].coord, 12.0):
            other = atom.parent
            if other is residue or id(other) not in ordinal:
                continue
            pair = tuple(sorted((ordinal[id(residue)], ordinal[id(other)])))
            if pair in seen:
                continue
            seen.add(pair)
            call = _classify(residue, other)
            if call is None:
                continue
            kind, distance, atom_a, atom_b = call
            incident[id(residue)].append((kind, distance))
            incident[id(other)].append((kind, distance))
            evidence.append({"residue_1": pair[0], "residue_2": pair[1], "type": kind, "distance": distance, "atom_1": atom_a, "atom_2": atom_b, "rule_version": "itsa-v0-auditable"})
    tokens: list[Token] = []
    for index, residue in enumerate(residues, 1):
        best: dict[str, float] = {}
        for kind, distance in incident[id(residue)]:
            best[kind] = min(distance, best.get(kind, math.inf))
        ordered = sorted(best, key=lambda kind: (best[kind], PRIORITY[kind], kind))[:3]
        residue_id = f"{residue.id[1]}{residue.id[2].strip()}"
        tokens.append(Token(index, residue.parent.id, residue_id, tuple(LETTERS[k] for k in ordered) or ("U",), dssp[(residue.parent.id, residue.id)]))
    return tokens, evidence


def main():
    import argparse, json
    parser=argparse.ArgumentParser(description="Tokenize one extracted SCOPe domain")
    parser.add_argument("domain_pdb",type=Path)
    parser.add_argument("--dssp-context",type=Path)
    parser.add_argument("--dssp",default="mkdssp")
    parser.add_argument("--tokens-jsonl",type=Path,required=True)
    parser.add_argument("--encoded-output",type=Path,required=True)
    args=parser.parse_args()
    tokens,evidence=tokenize(args.domain_pdb,args.dssp,args.dssp_context)
    args.tokens_jsonl.parent.mkdir(parents=True,exist_ok=True)
    with args.tokens_jsonl.open("w") as handle:
        for token in tokens:
            handle.write(json.dumps(token.as_dict(),sort_keys=True)+"\n")
    encoded=";".join(t.primary+t.secondary_structure+"".join(sorted(set(t.interactions[1:]))) for t in tokens)
    args.encoded_output.write_text(encoded+"\n")
    evidence_path=args.tokens_jsonl.with_name(args.tokens_jsonl.stem+"_interactions.jsonl")
    with evidence_path.open("w") as handle:
        for row in evidence:
            handle.write(json.dumps(row,sort_keys=True)+"\n")
    print(json.dumps({"domain":str(args.domain_pdb),"tokens":len(tokens),"encoded_output":str(args.encoded_output),"evidence":str(evidence_path)},indent=2))

if __name__=="__main__":
    main()
