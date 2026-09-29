"""Memory-augmented target-directed search for the ML3 JT-VAE app.

Memory is explicit and persistent for the current app session.  Each iteration
stores generated ligands, lineage, steric modifications, pair predictions,
target errors and selection status.  The next iteration uses the elite ligand
SMILES as JT-VAE latent seeds instead of restarting entirely from random noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import random
from itertools import combinations_with_replacement, product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


TAU_REF = 100.0


class PropNet(nn.Module):
    def __init__(self, d=56, h=256, out=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, out)
        )

    def forward(self, x):
        return self.net(x)


class Graph:
    def __init__(self, x, edge_index, edge_attr):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr


class BatchGraph:
    def __init__(self, graphs):
        xs, eis, eas, bs = [], [], [], []
        off = 0
        for i, g in enumerate(graphs):
            xs.append(g.x)
            bs.append(torch.full((len(g.x),), i, dtype=torch.long))
            if g.edge_index.numel():
                eis.append(g.edge_index + off)
                eas.append(g.edge_attr)
            off += len(g.x)
        self.x = torch.cat(xs, 0)
        self.batch = torch.cat(bs, 0)
        self.edge_index = torch.cat(eis, 1) if eis else torch.empty((2, 0), dtype=torch.long)
        self.edge_attr = torch.cat(eas, 0) if eas else torch.empty((0, 6), dtype=torch.float32)


class GNNEncoder(nn.Module):
    def __init__(self, hidden=64, embed=64, layers=3):
        super().__init__()
        self.n = nn.Linear(18, hidden)
        self.e = nn.Linear(6, hidden)
        self.ms = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            for _ in range(layers)
        ])
        self.pr = nn.Sequential(nn.Linear(hidden, embed), nn.ReLU(), nn.Linear(embed, embed))

    def forward(self, g):
        x = self.n(g.x)
        s, d = g.edge_index
        for mlp in self.ms:
            agg = torch.zeros_like(x)
            if s.numel():
                msg = x[s] + self.e(g.edge_attr)
                agg.index_add_(0, d, msg)
                deg = torch.zeros((len(x), 1))
                deg.index_add_(0, d, torch.ones((len(d), 1)))
                agg /= deg.clamp_min(1.0)
            x = x + F.relu(mlp(x + agg))
        n = int(g.batch.max().item()) + 1
        pooled = torch.zeros((n, x.shape[1]))
        pooled.index_add_(0, g.batch, x)
        cnt = torch.bincount(g.batch, minlength=n).float().unsqueeze(1)
        pooled /= cnt.clamp_min(1.0)
        return self.pr(pooled)


class PairGNN(nn.Module):
    def __init__(self, hidden=64, embed=64, layers=3):
        super().__init__()
        self.enc = GNNEncoder(hidden, embed, layers)
        self.f = nn.Sequential(
            nn.Linear(embed * 2 + 2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, g1, g2, cn):
        return self.f(torch.cat([self.enc(g1), self.enc(g2), cn], 1))


def atom_features(a):
    vocab = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    f = [float(a.GetAtomicNum() == z) for z in vocab] + [float(a.GetAtomicNum() not in vocab)]
    try:
        hv = float(a.GetHybridization().real)
    except Exception:
        hv = 0.0
    f += [
        a.GetDegree() / 6.0,
        a.GetFormalCharge() / 3.0,
        a.GetTotalNumHs() / 4.0,
        float(a.GetIsAromatic()),
        float(a.IsInRing()),
        hv,
    ]
    return f


def bond_features(b):
    bt = b.GetBondType()
    return [
        float(bt == Chem.rdchem.BondType.SINGLE),
        float(bt == Chem.rdchem.BondType.DOUBLE),
        float(bt == Chem.rdchem.BondType.TRIPLE),
        float(bt == Chem.rdchem.BondType.AROMATIC),
        float(b.GetIsConjugated()),
        float(b.IsInRing()),
    ]


def smiles_graph(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    x = torch.tensor([atom_features(a) for a in m.GetAtoms()], dtype=torch.float32)
    src, dst, ea = [], [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        src += [i, j]
        dst += [j, i]
        ea += [bf, bf]
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
    e = torch.tensor(ea, dtype=torch.float32) if ea else torch.empty((0, 6), dtype=torch.float32)
    return Graph(x, ei, e)


def tor_from(ueff, tio):
    den = (math.log10(TAU_REF) - float(tio)) * math.log(10.0)
    return float(ueff / den) if den > 0 else float("nan")


def target_value_from_row(row, kind):
    if kind == "Ucal":
        return float(row["Ucal"])
    if kind == "Ueff":
        return float(row["Ueff"])
    return tor_from(row["Ueff"], row["tio"])


def target_value_from_raw(raw, kind):
    if kind == "Ucal":
        return float(raw[0])
    if kind == "Ueff":
        return float(raw[1])
    return tor_from(raw[1], raw[2])


def target_error(raw, target, kind):
    val = target_value_from_raw(raw, kind)
    scale = max(abs(float(target)), 100.0)
    return abs(val - float(target)) / scale


def fingerprint(smiles):
    m = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) if m else None


def max_similarity(smiles, refs):
    fp = fingerprint(smiles)
    if fp is None or not refs:
        return 0.0
    sims = [DataStructs.TanimotoSimilarity(fp, r) for r in refs if r is not None]
    return max(sims) if sims else 0.0


