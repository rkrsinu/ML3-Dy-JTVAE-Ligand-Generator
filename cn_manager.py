
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


@dataclass(frozen=True)
class CNAssignment:
    cn1: int
    cn2: int
    source: str
    confidence: float


def canonicalize(smiles: str | None) -> str | None:
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)


class CNManager:
    """
    CN assignment for the ML3 pair-GNN.

    Priority:
      1. Exact observed ligand pair -> exact observed CN1/CN2.
      2. Exact observed ligand in its L1/L2 role -> observed role-specific CN.
      3. Steric derivative with a known parent -> inherit parent's role CN.
      4. Novel ligand -> nearest observed ligand(s), role-aware, with a
         similarity threshold. No global arbitrary CN enumeration.

    CN is never equated to donor-atom count.
    """

    def __init__(self, df: pd.DataFrame):
        required = {"L1", "L2", "CN1", "CN2"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Dataset missing CN columns: {sorted(missing)}")

        self.df = df.copy()
        self.pair_map: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self.role_map = {
            "L1": defaultdict(Counter),
            "L2": defaultdict(Counter),
        }
        self.overall_map = defaultdict(Counter)
        self.known = set()
        self.fp_cache = {}

        for _, r in self.df.iterrows():
            a = canonicalize(r["L1"])
            b = canonicalize(r["L2"])
            if not a or not b:
                continue

            try:
                c1 = int(round(float(r["CN1"])))
                c2 = int(round(float(r["CN2"])))
            except Exception:
                continue
            if c1 < 1 or c2 < 1:
                continue

            self.known.update([a, b])
            self.pair_map[(a, b)][(c1, c2)] += 1
            self.role_map["L1"][a][c1] += 1
            self.role_map["L2"][b][c2] += 1
            self.overall_map[a][c1] += 1
            self.overall_map[b][c2] += 1

        # Observed ligands used for nearest-neighbour inference.
        self.known_list = sorted(self.known)
        for s in self.known_list:
            self.fp_cache[s] = _fp(s)

    def is_known(self, smiles: str) -> bool:
        c = canonicalize(smiles)
        return bool(c and c in self.known)

    def exact_pair(self, a: str, b: str) -> List[CNAssignment]:
        a = canonicalize(a)
        b = canonicalize(b)
        if not a or not b:
            return []

        direct = self.pair_map.get((a, b), Counter())
        if direct:
            total = sum(direct.values())
            return [
                CNAssignment(c1, c2, "observed_pair", count / total)
                for (c1, c2), count in direct.most_common()
            ]

        # If the same complex occurs in the reverse ligand order, swap CNs.
        rev = self.pair_map.get((b, a), Counter())
        if rev:
            total = sum(rev.values())
            return [
                CNAssignment(c2, c1, "observed_pair_reversed", count / total)
                for (c1, c2), count in rev.most_common()
            ]
        return []

    def role_values(self, smiles: str, role: str, max_values: int = 2):
        c = canonicalize(smiles)
        if not c:
            return []

        counter = self.role_map[role].get(c, Counter())
        if not counter:
            counter = self.overall_map.get(c, Counter())
        if not counter:
            return []

        total = sum(counter.values())
        return [
            (int(cn), float(count / total))
            for cn, count in counter.most_common(max_values)
        ]

    def nearest(self, smiles: str, role: str, top_k: int = 5,
                min_similarity: float = 0.35):
        c = canonicalize(smiles)
        if not c:
            return []

        qfp = _fp(c)
        if qfp is None:
            return []

        # Prefer role-specific ligands because CN1/CN2 are role-dependent.
        role_col = "L1" if role == "L1" else "L2"
        candidates = set(self.role_map[role].keys())
        if not candidates:
            candidates = set(self.known_list)

        scored = []
        for s in candidates:
            fp = self.fp_cache.get(s)
            if fp is None:
                continue
            sim = DataStructs.TanimotoSimilarity(qfp, fp)
            if sim >= min_similarity:
                scored.append((sim, s))

        scored.sort(reverse=True)
        return scored[:top_k]

    def novel_cn(self, smiles: str, role: str,
                 parent_smiles: Optional[str] = None,
                 max_values: int = 2,
                 min_similarity: float = 0.35):
        # Exact observed ligand.
        vals = self.role_values(smiles, role, max_values=max_values)
        if vals:
            return [
                (cn, "observed_ligand", conf)
                for cn, conf in vals
            ]

        # A steric derivative inherits the parent's observed coordination
        # environment. This is much safer than assigning an arbitrary CN.
        if parent_smiles:
            pvals = self.role_values(parent_smiles, role, max_values=max_values)
            if pvals:
                return [
                    (cn, "inherited_parent", conf)
                    for cn, conf in pvals
                ]

        # Genuine novel ligand: infer only from structurally similar observed
        # ligands in the same positional role.
        neighbors = self.nearest(
            smiles, role, top_k=5, min_similarity=min_similarity
        )
        if not neighbors:
            return []

        weighted = Counter()
        weight_sum = 0.0
        for sim, ns in neighbors:
            counter = self.role_map[role].get(ns, Counter())
            if not counter:
                continue
            for cn, count in counter.items():
                weighted[cn] += sim * count
                weight_sum += sim * count

        if not weighted or weight_sum <= 0:
            return []

        out = []
        for cn, score in weighted.most_common(max_values):
            out.append((int(cn), "nearest_neighbor", float(score / weight_sum)))
        return out

    def pair_assignments(
        self,
        a: str,
        b: str,
        parent_a: Optional[str] = None,
        parent_b: Optional[str] = None,
        max_per_ligand: int = 2,
        min_similarity: float = 0.35,
    ) -> List[CNAssignment]:

        # Strongest case: exact experimental pair.
        exact = self.exact_pair(a, b)
        if exact:
            return exact

        a_vals = self.novel_cn(
            a, "L1", parent_a,
            max_values=max_per_ligand,
            min_similarity=min_similarity,
        )
        b_vals = self.novel_cn(
            b, "L2", parent_b,
            max_values=max_per_ligand,
            min_similarity=min_similarity,
        )

        if not a_vals or not b_vals:
            return []

        combos = []
        for c1, s1, p1 in a_vals:
            for c2, s2, p2 in b_vals:
                combos.append(
                    CNAssignment(
                        int(c1), int(c2),
                        f"{s1}+{s2}",
                        float(p1 * p2),
                    )
                )

        # Only retain assignments supported by the two ligands' evidence.
        combos.sort(key=lambda x: x.confidence, reverse=True)
        return combos[: max_per_ligand * max_per_ligand]
