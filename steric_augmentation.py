"""Adaptive steric diversification for the ML3 memory search.

Steric diversification is automatic.  The user does not select substituents.
The substitution set is chosen from the parent ligand size and the number of
available carbon-bound hydrogens, and every product is RDKit-sanitized and
canonicalized.
"""
from __future__ import annotations

from typing import Iterable
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

SUBSTITUENTS = {
    "Me": ("C", "methyl"),
    "Et": ("CC", "ethyl"),
    "nPr": ("CCC", "n-propyl"),
    "iPr": ("C(C)C", "isopropyl"),
    "tBu": ("C(C)(C)C", "tert-butyl"),
}


def canonicalize(smiles: str | None) -> str | None:
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def passes_ring_rule(smiles: str, max_rings: int = 1) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and rdMolDescriptors.CalcNumRings(mol) <= max_rings


def eligible_c_h_sites(smiles: str) -> int:
    """Number of distinct carbon sites bearing at least one H."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    return sum(
        1 for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 6
        and any(n.GetAtomicNum() == 1 for n in atom.GetNeighbors())
    )


def adaptive_labels(smiles: str) -> list[str]:
    """Choose steric transformations from the ligand itself.

    Small ligands are explored with progressively larger alkyl groups.  Larger
    ligands are kept to moderate substitutions to avoid uncontrolled growth.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    heavy = mol.GetNumHeavyAtoms()
    rings = rdMolDescriptors.CalcNumRings(mol)
    sites = eligible_c_h_sites(smiles)

    if sites == 0:
        return []

    if heavy <= 12 and rings <= 1:
        labels = ["Me", "Et", "nPr", "iPr"]
    elif heavy <= 24:
        labels = ["Me", "Et", "iPr"]
    else:
        labels = ["Me", "Et"]

    # Very crowded parents are not expanded with the largest substituents.
    if sites <= 1:
        labels = [x for x in labels if x in {"Me", "Et"}]

    return labels


def substitute_one_h(smiles: str, label: str) -> list[str]:
    """Replace one carbon-bound H with one requested alkyl group."""
    parent = canonicalize(smiles)
    if parent is None or label not in SUBSTITUENTS:
        return []

    fragment_smiles, _ = SUBSTITUENTS[label]
    mol_h = Chem.AddHs(Chem.MolFromSmiles(parent))
    products: set[str] = set()

    for atom in list(mol_h.GetAtoms()):
        if atom.GetAtomicNum() != 6:
            continue
        h_neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
        if not h_neighbors:
            continue

        try:
            marked = Chem.RWMol(mol_h)
            anchor_idx = atom.GetIdx()
            marked.GetAtomWithIdx(anchor_idx).SetAtomMapNum(999)
            marked.RemoveAtom(h_neighbors[0].GetIdx())
            anchor = next(a for a in marked.GetAtoms() if a.GetAtomMapNum() == 999)
            anchor.SetAtomMapNum(0)

            frag = Chem.MolFromSmiles(fragment_smiles)
            if frag is None:
                continue

            new_indices = [marked.AddAtom(Chem.Atom(a.GetSymbol())) for a in frag.GetAtoms()]
            marked.AddBond(anchor.GetIdx(), new_indices[0], Chem.BondType.SINGLE)
            for b in frag.GetBonds():
                i = new_indices[b.GetBeginAtomIdx()]
                j = new_indices[b.GetEndAtomIdx()]
                marked.AddBond(i, j, b.GetBondType())

            product = marked.GetMol()
            Chem.SanitizeMol(product)
            product = Chem.RemoveHs(product)
            out = Chem.MolToSmiles(product, canonical=True)
            if out and out != parent and passes_ring_rule(out):
                products.add(out)
        except Exception:
            continue

    return sorted(products)


def generate_steric_variants(
    smiles: str,
    labels: Iterable[str] | None = None,
    max_variants: int = 8,
) -> list[tuple[str, str]]:
    parent = canonicalize(smiles)
    if parent is None:
        return []

    if labels is None:
        labels = adaptive_labels(parent)

    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label in labels:
        for product in substitute_one_h(parent, label):
            if product not in seen:
                seen.add(product)
                results.append((product, label))
                if len(results) >= max_variants:
                    return results
    return results


def augment_ligand_pool(
    smiles_list: Iterable[str],
    labels: Iterable[str] | None = None,
    max_variants_per_parent: int = 8,
) -> list[dict]:
    rows = []
    for parent in smiles_list:
        p = canonicalize(parent)
        if p is None:
            continue
        use_labels = list(labels) if labels is not None else adaptive_labels(p)
        for child, label in generate_steric_variants(
            p, labels=use_labels, max_variants=max_variants_per_parent
        ):
            rows.append({
                "smiles": child,
                "source": f"adaptive_steric_{label}",
                "parent_smiles": p,
                "modification": f"H_to_{label}",
            })

    out = {}
    for row in rows:
        out.setdefault(row["smiles"], row)
    return list(out.values())
