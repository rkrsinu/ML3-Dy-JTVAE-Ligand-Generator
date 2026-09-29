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

from steric_augmentation import canonicalize, augment_ligand_pool

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


@dataclass
class SearchConfig:
    iterations: int = 4
    starts_per_iteration: int = 12
    latent_steps: int = 60
    latent_lr: float = 0.06
    decode_per_seed: int = 3
    latent_noise: float = 0.35
    max_new_ligands: int = 80
    max_memory_ligands: int = 120
    max_pairs: int = 30000
    top_pairs_per_iteration: int = 20
    steric_variants_per_parent: int = 6
    use_me: bool = True
    use_et: bool = True
    use_npr: bool = False
    use_ipr: bool = False
    use_tbu: bool = False
    diversity_weight: float = 0.15
    seed: int = 42


class MemoryArchive:
    def __init__(self):
        self.ligands = pd.DataFrame(columns=[
            "smiles", "source", "parent_smiles", "modification", "iteration",
            "latent_target_error", "memory_score", "selected"
        ])
        self.pairs = pd.DataFrame()

    def add_ligands(self, df):
        if df is None or len(df) == 0:
            return
        cols = [
            "smiles", "source", "parent_smiles", "modification", "iteration",
            "latent_target_error", "memory_score", "selected"
        ]
        x = df.copy()
        for c in cols:
            if c not in x:
                x[c] = "" if c in ["smiles", "source", "parent_smiles", "modification"] else np.nan
        x = x[cols]
        self.ligands = pd.concat([self.ligands, x], ignore_index=True)
        self.ligands = self.ligands.drop_duplicates("smiles", keep="first").reset_index(drop=True)

    def add_pairs(self, df):
        if df is None or len(df) == 0:
            return
        self.pairs = pd.concat([self.pairs, df], ignore_index=True)

    def elite_ligands(self, n=20):
        if len(self.pairs) == 0:
            return []
        rows = []
        for _, r in self.pairs.sort_values("target_error").head(n).iterrows():
            rows.extend([r["Ligand 1"], r["Ligand 2"]])
        out = []
        seen = set()
        for s in rows:
            if s not in seen:
                seen.add(s); out.append(s)
        return out

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        self.ligands.to_csv(folder / "memory_ligands.csv", index=False)
        self.pairs.to_csv(folder / "memory_pairs.csv", index=False)
        with open(folder / "memory_summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "ligands": int(len(self.ligands)),
                "pairs": int(len(self.pairs)),
                "iterations": sorted(self.ligands["iteration"].dropna().unique().tolist()) if len(self.ligands) else [],
            }, f, indent=2)


def build_target_vector(kind, target, mu, sd):
    # Unspecified properties stay at the learned training mean; the selected
    # target is driven by the requested value. This makes the search stable
    # while still allowing extrapolation in the requested direction.
    raw = mu.clone()
    if kind == "Ucal":
        raw[0] = float(target)
    elif kind == "Ueff":
        raw[1] = float(target)
    else:
        # For Tor, choose the training-mean Ueff and solve for tio.
        u = float(mu[1])
        tio = math.log10(TAU_REF) - u / max(float(target), 1e-6) / math.log(10.0)
        raw[2] = tio
    return (raw - mu) / sd


def optimize_latent(seed_z, target_kind, target, prop, mu, sd, steps, lr):
    z = seed_z.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    target_z = build_target_vector(target_kind, target, mu, sd)
    idx = {"Ucal": 0, "Ueff": 1}.get(target_kind)

    for _ in range(steps):
        opt.zero_grad()
        p = prop(z)
        if idx is not None:
            selected = (p[:, idx] - target_z[idx]) ** 2
            others = [j for j in range(3) if j != idx]
            loss = selected.mean() + 0.025 * (p[:, others] ** 2).mean()
        else:
            raw = p * sd + mu
            ueff, tio = raw[:, 1], raw[:, 2]
            den = torch.clamp((math.log10(TAU_REF) - tio) * math.log(10.0), min=0.25)
            tor = ueff / den
            loss = ((tor - float(target)) / max(abs(float(target)), 100.0)).pow(2).mean()
            loss = loss + 0.025 * p[:, 0].pow(2).mean()
        loss = loss + 1e-4 * z.pow(2).mean()
        loss.backward()
        opt.step()
    return z.detach()


def encode_ligand_latent(jt, vocab, smiles):
    """Encode one valid ligand to the 56-D JT-VAE mean latent."""
    from fast_jtnn.mol_tree import MolTree
    from fast_jtnn.datautils_prop import set_batch_nodeID
    from fast_jtnn.jtnn_enc import JTNNEncoder
    from fast_jtnn.mpn import MPN

    tree = [MolTree(smiles)]
    set_batch_nodeID(tree, vocab)
    jt_holder, _ = JTNNEncoder.tensorize(tree)
    mpn_holder = MPN.tensorize([smiles])
    with torch.no_grad():
        z_mean, _ = jt.encode_latent(jt_holder, mpn_holder)
    return z_mean


def generate_from_seeds(
    jt, vocab, prop, mu, sd, known, target_kind, target, seed_smiles,
    iteration, cfg: SearchConfig,
):
    random.seed(cfg.seed + iteration)
    torch.manual_seed(cfg.seed + iteration)
    results = {}

    # Memory seeds: encode elite ligands and locally perturb their latent vectors.
    encoded = []
    for s in seed_smiles[:cfg.starts_per_iteration]:
        try:
            encoded.append(encode_ligand_latent(jt, vocab, s))
        except Exception:
            continue

    # Always retain some fresh random starts so the search can escape memory.
    n_random = max(2, cfg.starts_per_iteration // 3)
    seeds = encoded + [torch.randn(1, 56) * 1.0 for _ in range(n_random)]
    if not seeds:
        seeds = [torch.randn(1, 56)]

    for base in seeds:
        z0 = base
        for k in range(cfg.decode_per_seed):
            noise = torch.randn_like(z0) * cfg.latent_noise if k else torch.zeros_like(z0)
            z = optimize_latent(
                z0 + noise, target_kind, target, prop, mu, sd,
                cfg.latent_steps, cfg.latent_lr
            )
            try:
                zt, zm = torch.chunk(z, 2, 1)
                smi = jt.decode(zt, zm, False)
            except Exception:
                smi = None
            smi = canonicalize(smi)
            if not smi or smi in known or smi in results:
                continue
            with torch.no_grad():
                raw = (prop(z) * sd + mu)[0].numpy()
            err = target_error(raw, target, target_kind)
            results[smi] = {
                "smiles": smi,
                "source": "jtvae_memory_search" if encoded else "jtvae_search",
                "parent_smiles": seed_smiles[0] if seed_smiles else "",
                "modification": "latent_optimization",
                "iteration": iteration,
                "latent_target_error": float(err),
                "latent_Ucal": float(raw[0]),
                "latent_Ueff": float(raw[1]),
                "latent_tio": float(raw[2]),
            }
            if len(results) >= cfg.max_new_ligands:
                return list(results.values())
    return list(results.values())


def pair_screen(candidates, cn1, cn2, gnn, gmu, gsd, target_kind, target, max_pairs):
    smiles = list(dict.fromkeys(candidates))
    if len(smiles) < 2:
        return pd.DataFrame()
    pairs = list(combinations_with_replacement(smiles, 2)) if cn1 == cn2 else list(product(smiles, smiles))
    if len(pairs) > max_pairs:
        random.Random(123).shuffle(pairs)
        pairs = pairs[:max_pairs]

    rows = []
    for start in range(0, len(pairs), 256):
        chunk = pairs[start:start + 256]
        graphs1 = [smiles_graph(a) for a, _ in chunk]
        graphs2 = [smiles_graph(b) for _, b in chunk]
        if any(g is None for g in graphs1 + graphs2):
            continue
        g1 = BatchGraph(graphs1)
        g2 = BatchGraph(graphs2)
        cn = torch.tensor([[cn1, cn2]] * len(chunk), dtype=torch.float32)
        with torch.no_grad():
            pred = gnn(g1, g2, cn) * gsd + gmu
        for (a, b), p in zip(chunk, pred.numpy()):
            ucal, ueff, tio = map(float, p)
            tor = tor_from(ueff, tio)
            raw = [ucal, ueff, tio]
            rows.append({
                "Ligand 1": a,
                "Ligand 2": b,
                "CN1": cn1,
                "CN2": cn2,
                "Predicted Ucal (K)": ucal,
                "Predicted Ueff (K)": ueff,
                "Predicted log10(tau0)": tio,
                "Predicted Tor (K)": tor,
                "target_error": target_error(raw, target, target_kind),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("target_error").reset_index(drop=True)


def run_iterative_search(
    models, target_kind, target, cn1, cn2, cfg: SearchConfig,
    initial_ligands, progress=None,
):
    jt, vocab, prop, mu, sd, gnn, gmu, gsd, known, _df = models
    archive = MemoryArchive()
    memory_seed_smiles = list(dict.fromkeys(initial_ligands))
    all_generated = []

    for iteration in range(1, cfg.iterations + 1):
        if progress:
            progress(iteration, "latent generation")

        generated = generate_from_seeds(
            jt, vocab, prop, mu, sd, known, target_kind, target,
            memory_seed_smiles, iteration, cfg,
        )

        if generated:
            gen_df = pd.DataFrame(generated)
            # Steric augmentation is deliberately applied to generated ligands,
            # not the original experimental library.
            labels = []
            if cfg.use_me: labels.append("Me")
            if cfg.use_et: labels.append("Et")
            if cfg.use_npr: labels.append("nPr")
            if cfg.use_ipr: labels.append("iPr")
            if cfg.use_tbu: labels.append("tBu")
            variants = augment_ligand_pool(
                gen_df["smiles"].tolist(),
                labels=labels,
                max_variants_per_parent=cfg.steric_variants_per_parent,
            ) if labels else []
            if variants:
                var_df = pd.DataFrame(variants)
                var_df["iteration"] = iteration
                var_df["latent_target_error"] = np.nan
                var_df["memory_score"] = np.nan
                var_df["selected"] = False
                gen_df = pd.concat([gen_df, var_df], ignore_index=True, sort=False)
            archive.add_ligands(gen_df)
            all_generated.extend(gen_df["smiles"].tolist())

        # Search memory + new molecules. Keep the pool bounded by target-relevant
        # memory first, then fresh generated molecules.
        pool = list(dict.fromkeys(
            memory_seed_smiles + all_generated[-cfg.max_new_ligands:]
        ))
        pool = [s for s in pool if canonicalize(s)]
        if len(pool) > 120:
            pool = pool[-120:]

        if progress:
            progress(iteration, f"pair screening ({len(pool)} ligands)")
        pairs = pair_screen(pool, cn1, cn2, gnn, gmu, gsd, target_kind, target, cfg.max_pairs)
        if len(pairs):
            pairs["iteration"] = iteration
            pairs["selected"] = False
            topn = min(cfg.top_pairs_per_iteration, len(pairs))
            pairs.loc[:topn - 1, "selected"] = True
            archive.add_pairs(pairs.head(max(cfg.top_pairs_per_iteration * 3, 50)))

            # Memory selection: retain ligands occurring in the best pairs, plus
            # a small diversity-preserving set of near-target ligands.
            elite = []
            for _, r in pairs.head(cfg.top_pairs_per_iteration).iterrows():
                elite.extend([r["Ligand 1"], r["Ligand 2"]])
            elite = list(dict.fromkeys(elite))
            memory_seed_smiles = elite[:cfg.max_memory_ligands]
            for s in memory_seed_smiles:
                archive.ligands.loc[archive.ligands.smiles == s, "selected"] = True

        if progress:
            progress(iteration, f"stored {len(archive.ligands)} ligands / {len(archive.pairs)} pair records")

    if len(archive.pairs):
        final = archive.pairs.sort_values("target_error").drop_duplicates(
            subset=["Ligand 1", "Ligand 2", "CN1", "CN2"], keep="first"
        ).reset_index(drop=True)
    else:
        final = pd.DataFrame()
    return archive, final
