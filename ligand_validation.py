
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Set

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors


@dataclass
class LigandValidation:
    valid: bool
    smiles: Optional[str]
    reason: str
    max_train_tanimoto: float = 0.0


def canonicalize(smiles: str | None) -> str | None:
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)


def max_train_similarity(smiles: str, refs: Iterable[str]) -> float:
    q = fingerprint(smiles)
    if q is None:
        return 0.0
    best = 0.0
    for s in refs:
        fp = fingerprint(s)
        if fp is not None:
            best = max(best, DataStructs.TanimotoSimilarity(q, fp))
    return best


def _single_component(mol) -> bool:
    return len(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)) == 1


def _looks_like_fused_known_fragments(mol, known_smiles, threshold=0.78):
    """
    Heuristic guard against a decoded structure that effectively fuses two
    library ligands. We only flag a candidate when breaking a non-ring single
    bond yields TWO substantial fragments, each highly similar to a different
    observed ligand.
    """
    if mol.GetNumBonds() < 2:
        return False

    known_fps = [(s, fingerprint(s)) for s in known_smiles]
    known_fps = [(s, fp) for s, fp in known_fps if fp is not None]

    for bond in mol.GetBonds():
        if bond.GetIsAromatic() or bond.IsInRing():
            continue
        if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
            continue

        rw = Chem.RWMol(mol)
        rw.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        broken = rw.GetMol()

        try:
            Chem.SanitizeMol(broken)
        except Exception:
            continue

        frags = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True)
        if len(frags) != 2:
            continue

        if min(f.GetNumHeavyAtoms() for f in frags) < 5:
            continue

        best_scores = []
        for frag in frags:
            fsm = Chem.MolToSmiles(frag, canonical=True)
            ffp = fingerprint(fsm)
            if ffp is None:
                best_scores.append(0.0)
                continue
            best_scores.append(
                max(DataStructs.TanimotoSimilarity(ffp, kfp)
                    for _, kfp in known_fps)
            )

        if min(best_scores) >= threshold:
            return True

    return False


def validate_generated_ligand(
    smiles: str,
    known_smiles: Iterable[str],
    allowed_atomic_numbers: Optional[Set[int]] = None,
    max_rings: int = 1,
    max_heavy_atoms: Optional[int] = None,
    min_similarity: float = 0.25,
    reject_fused_known_fragments: bool = True,
) -> LigandValidation:

    c = canonicalize(smiles)
    if not c:
        return LigandValidation(False, None, "invalid_smiles")

    mol = Chem.MolFromSmiles(c)
    if mol is None:
        return LigandValidation(False, None, "invalid_smiles")

    if not _single_component(mol):
        return LigandValidation(False, None, "multiple_fragments")

    if allowed_atomic_numbers is not None:
        elems = {a.GetAtomicNum() for a in mol.GetAtoms()}
        if not elems.issubset(allowed_atomic_numbers):
            return LigandValidation(False, None, "element_not_in_training_library")

    rings = rdMolDescriptors.CalcNumRings(mol)
    if rings > max_rings:
        return LigandValidation(False, None, f"ring_count>{max_rings}")

    heavy = mol.GetNumHeavyAtoms()
    if max_heavy_atoms is not None and heavy > max_heavy_atoms:
        return LigandValidation(False, None, "too_many_heavy_atoms")

    refs = list(known_smiles)
    sim = max_train_similarity(c, refs) if refs else 0.0

    # Novel structures must remain connected to the learned chemical space.
    if refs and sim < min_similarity:
        return LigandValidation(False, None, "outside_training_chemical_space", sim)

    if reject_fused_known_fragments and refs:
        if _looks_like_fused_known_fragments(mol, refs):
            return LigandValidation(False, None, "looks_like_fused_ligand_combination", sim)

    return LigandValidation(True, c, "valid", sim)
