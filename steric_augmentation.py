"""Chemically constrained ligand augmentation used by the ML3 iterative search.

The original ML3 pair-generation workflow already contained H -> methyl and
H -> ethyl substitution.  This module preserves those operations and adds
optional larger alkyl substitutions for extrapolation.

Only carbon-bound hydrogens are substituted.  Products are RDKit-sanitized,
canonicalized and limited to the <=1-ring ligand space used by the project.
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


def substitute_one_h(smiles: str, label: str) -> list[str]:
    """Replace one C-H with the requested alkyl group at every possible site."""
    parent = canonicalize(smiles)
    if parent is None or label not in SUBSTITUENTS:
        return []

    fragment, _ = SUBSTITUENTS[label]
    mol_h = Chem.AddHs(Chem.MolFromSmiles(parent))
    products: set[str] = set()

    for atom in list(mol_h.GetAtoms()):
        if atom.GetAtomicNum() != 6:
            continue
        h_neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
        if not h_neighbors:
            continue

        rw = Chem.RWMol(mol_h)
        h_idx = h_neighbors[0].GetIdx()
        anchor_idx = atom.GetIdx()
        try:
            rw.RemoveAtom(h_idx)
            # Removing a hydrogen whose index is lower than the anchor shifts
            # the anchor index by one.  Re-find the anchor by using the atom's
            # persistent isotope marker would be cumbersome, so instead mark it
            # before editing in a fresh copy.
            # The robust route is to mark the anchor before deletion.
        except Exception:
            continue

        # Repeat robustly with an atom-map marker.
        try:
            marked = Chem.RWMol(mol_h)
            anchor = marked.GetAtomWithIdx(anchor_idx)
            anchor.SetAtomMapNum(999)
            marked.RemoveAtom(h_idx)
            anchor_after = next(a for a in marked.GetAtoms() if a.GetAtomMapNum() == 999)
            anchor_after.SetAtomMapNum(0)
            frag = Chem.MolFromSmiles(fragment)
            frag_idx = marked.AddAtom(Chem.Atom("C"))
            # Add the fragment explicitly so no dummy atoms or aromaticity
            # assumptions enter the product.
            if label == "Me":
                marked.AddBond(anchor_after.GetIdx(), frag_idx, Chem.BondType.SINGLE)
            else:
                # Build the requested alkyl chain/branch from atom SMILES.
                # The first carbon is attached to the anchor.
                frag_atoms = [Chem.Atom("C") for _ in range(frag.GetNumAtoms())]
                # Remove the temporary atom; construct all fragment atoms.
                marked.RemoveAtom(frag_idx)
                new_indices = [marked.AddAtom(a) for a in frag_atoms]
                marked.AddBond(anchor_after.GetIdx(), new_indices[0], Chem.BondType.SINGLE)
                # Connect according to the fragment topology.
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



def choose_alkyl_labels(smiles: str) -> list[str]:
    """Choose steric substitutions automatically from ligand structure.

    The rule is deliberately deterministic and conservative: smaller ligands
    are allowed to explore progressively larger alkyl groups, whereas larger
    ligands are restricted to less aggressive substitutions. Only labels for
    which the ligand has at least one carbon-bound C-H site are returned.
    """
    parent = canonicalize(smiles)
    if parent is None:
        return []

    mol = Chem.MolFromSmiles(parent)
    if mol is None:
        return []

    carbon_h_sites = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() > 0:
            carbon_h_sites += 1

    if carbon_h_sites == 0:
        return []

    heavy = mol.GetNumHeavyAtoms()
    carbons = sum(a.GetAtomicNum() == 6 for a in mol.GetAtoms())
    rings = rdMolDescriptors.CalcNumRings(mol)

    # Small/compact ligands can tolerate a broader steric search; larger
    # ligands are restricted to milder substitutions.
    if heavy <= 12 and carbons <= 9:
        labels = ["Me", "Et", "nPr", "iPr", "tBu"]
    elif heavy <= 20 and rings <= 3:
        labels = ["Me", "Et", "nPr", "iPr"]
    elif heavy <= 28:
        labels = ["Me", "Et", "iPr"]
    else:
        labels = ["Me", "Et"]

    # Do not request more substitution types than there are plausible sites.
    if carbon_h_sites == 1:
        labels = labels[:2]
    elif carbon_h_sites == 2:
        labels = labels[:3]

    return labels

def generate_steric_variants(
    smiles: str,
    labels: Iterable[str] | None = None,
    max_variants: int = 12,
) -> list[tuple[str, str]]:
    parent = canonicalize(smiles)
    if parent is None:
        return []

    if labels is None:
        labels = choose_alkyl_labels(parent)

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
        # labels=None means the chemistry-aware selector chooses the allowed
        # substitution family separately for this ligand.
        ligand_labels = choose_alkyl_labels(p) if labels is None else labels
        for child, label in generate_steric_variants(
            p, labels=ligand_labels, max_variants=max_variants_per_parent
        ):
            rows.append({
                "smiles": child,
                "source": f"steric_{label}",
                "parent_smiles": p,
                "modification": f"H_to_{label}",
            })
    # Deduplicate while preserving first lineage.
    out = {}
    for row in rows:
        out.setdefault(row["smiles"], row)
    return list(out.values())
