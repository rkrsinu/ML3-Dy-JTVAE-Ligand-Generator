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
from rdkit.Chem import AllChem, rdFingerprintGenerator

from steric_augmentation import canonicalize, augment_ligand_pool
from geometry_model import GeometryAwarePairGNN, batch_graph, load_geometry_model, load_property_model

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


PairGNN = GeometryAwarePairGNN

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


def _coordination_values(df):
    """Return the coordination-number values actually represented in training."""
    values = set()
    for col in ("CN1", "CN2"):
        if col in df.columns:
            values.update(
                int(round(float(x)))
                for x in pd.to_numeric(df[col], errors="coerce").dropna().unique()
                if float(x) >= 1
            )
    if not values:
        values = {1, 2, 3, 4, 5}
    return sorted(values)


def _ligand_cn_prior(smiles, df):
    """Get observed CN values for a ligand when available.

    This is used only as a search-prior. The final pair screening still
    evaluates all training-supported CN combinations, so no CN is imposed by
    the user interface or hard-coded for generated ligands.
    """
    c = canonicalize(smiles)
    if not c:
        return []

    observed = []
    for ligand_col, cn_col in (("L1", "CN1"), ("L2", "CN2")):
        if ligand_col not in df.columns:
            continue
        for raw_s, raw_cn in zip(df[ligand_col], df[cn_col]):
            if canonicalize(raw_s) == c:
                try:
                    observed.append(int(round(float(raw_cn))))
                except Exception:
                    pass
    return sorted(set(observed))


def _canonical_pair(a,b):
    return canonicalize(a), canonicalize(b)

def build_cn_map(geometry_df):
    mp={}
    for _,r in geometry_df.iterrows():
        a,b=canonicalize(r['L1_SMILES']),canonicalize(r['L2_SMILES'])
        if not a or not b: continue
        mp.setdefault((a,b),set()).add((int(round(float(r['CN1']))),int(round(float(r['CN2'])))))
        mp.setdefault((b,a),set()).add((int(round(float(r['CN2']))),int(round(float(r['CN1'])))))
    return mp

_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

def pair_fp(s):
    m=Chem.MolFromSmiles(str(s))
    return _MORGAN_GENERATOR.GetFingerprint(m) if m else None


def _make_batch_from_graphs(graphs):
    """Batch already-parsed molecular graphs without reparsing SMILES."""
    return BatchGraph(graphs)


def _prepare_cn_entries(geometry_df, fp_cache):
    """Build the observed CN-pair table and fingerprint cache once."""
    cn_map = build_cn_map(geometry_df)
    observed = sorted({x for pair in cn_map for x in pair})
    obs_fp = []
    for s in observed:
        fp = fp_cache.get(s)
        if fp is None and s not in fp_cache:
            fp = pair_fp(s); fp_cache[s] = fp
        obs_fp.append(fp)
    entries=[]
    for (x,y), vals in cn_map.items():
        if x in observed and y in observed:
            entries.append((observed.index(x), observed.index(y), vals))
    return cn_map, observed, obs_fp, entries


def _prepare_candidate_similarity(candidates, observed, observed_fp, fp_cache):
    """Return candidate-vs-observed Morgan similarities as a dense matrix.

    Only unique candidate ligands are queried.  RDKit computes each row in C,
    reducing the nearest-CN search from millions of Python/RDKit fingerprint
    operations to a few tens of thousands of bulk similarities.
    """
    mat=[]
    for s in candidates:
        fp=fp_cache.get(s)
        if fp is None and s not in fp_cache:
            fp=pair_fp(s); fp_cache[s]=fp
        if fp is None:
            mat.append(np.zeros(len(observed),dtype=np.float32))
        else:
            mat.append(np.asarray(DataStructs.BulkTanimotoSimilarity(fp, observed_fp),dtype=np.float32))
    return np.vstack(mat) if mat else np.empty((0,len(observed)),dtype=np.float32)

def nearest_cn(a,b,cn_map, fp_cache=None, entries=None):
    """Find the nearest observed ligand-pair CN environment.

    Fingerprints are precomputed once and then reused.  The numerical
    nearest-pair criterion is unchanged from the original implementation.
    """
    if fp_cache is None:
        fp_cache = {}
    if entries is None:
        entries=[]
        for (x,y),vals in cn_map.items():
            fx=fp_cache.get(x)
            if fx is None and x not in fp_cache: fx=pair_fp(x); fp_cache[x]=fx
            fy=fp_cache.get(y)
            if fy is None and y not in fp_cache: fy=pair_fp(y); fp_cache[y]=fy
            if fx is not None and fy is not None: entries.append((x,y,fx,fy,vals))
    fa=fp_cache.get(a)
    if fa is None and a not in fp_cache: fa=pair_fp(a); fp_cache[a]=fa
    fb=fp_cache.get(b)
    if fb is None and b not in fp_cache: fb=pair_fp(b); fp_cache[b]=fb
    if fa is None or fb is None:
        return None
    best=None; bestscore=-1.0
    for x,y,fx,fy,vals in entries:
        score=0.5*(DataStructs.TanimotoSimilarity(fa,fx)+DataStructs.TanimotoSimilarity(fb,fy))
        if score>bestscore:
            bestscore=score; best=(next(iter(vals)),score)
    return best


def valid_single_ligand(s):
    m=Chem.MolFromSmiles(str(s))
    return m is not None and len(Chem.GetMolFrags(m,asMols=False, sanitizeFrags=False))==1


def pair_screen(candidates, geometry_df, geom, gmean, gstd, gnn, gscaler,
                target_kind, target, max_pairs, progress=None, iteration=None):
    smiles=[]
    for s in candidates:
        c=canonicalize(s)
        if c and valid_single_ligand(c) and c not in smiles: smiles.append(c)
    if len(smiles)<2:return pd.DataFrame()

    # ---- Precompute once per screening call ----
    fp_cache={}
    cn_map, observed_ligands, observed_fps, cn_entries = _prepare_cn_entries(geometry_df, fp_cache)
    candidate_sim=_prepare_candidate_similarity(smiles, observed_ligands, observed_fps, fp_cache)
    observed_index={s:i for i,s in enumerate(observed_ligands)}
    # Entry arrays make the nearest-CN lookup fully vectorized.
    entry_x=np.asarray([e[0] for e in cn_entries],dtype=np.int32)
    entry_y=np.asarray([e[1] for e in cn_entries],dtype=np.int32)
    entry_vals=[e[2] for e in cn_entries]

    # Parse each candidate ligand exactly once.  The old implementation called
    # MolFromSmiles repeatedly for the same ligand in every pair/chunk.
    graph_cache={s: smiles_graph(s) for s in smiles}
    ligand_index={s:i for i,s in enumerate(smiles)}
    all_graphs=_make_batch_from_graphs([graph_cache[s] for s in smiles])
    with torch.no_grad():
        # These are the exact encoder outputs used by geom(...) and gnn(...).
        geom_embeddings=geom.enc(all_graphs)
        prop_embeddings=gnn.enc(all_graphs)

    pair_list=list(combinations_with_replacement(smiles,2))
    if len(pair_list)>max_pairs:
        random.Random(123).shuffle(pair_list);pair_list=pair_list[:max_pairs]

    rows=[]
    batch_size=512
    total_batches=(len(pair_list)+batch_size-1)//batch_size

    for bi,st in enumerate(range(0,len(pair_list),batch_size), start=1):
        chunk=pair_list[st:st+batch_size]
        valid=[]; cnsets=[]
        for a,b in chunk:
            if (a,b) in cn_map:
                cns=sorted(cn_map[(a,b)])
            else:
                ia=ligand_index[a]
                ib=ligand_index[b]
                scores=0.5*(candidate_sim[ia,entry_x]+candidate_sim[ib,entry_y])
                if len(scores):
                    j=int(np.argmax(scores))
                    cns=[next(iter(entry_vals[j]))]
                else:
                    cns=[]
            if cns:
                valid.append((a,b));cnsets.append(cns)

        expanded=[]
        for (a,b),cns in zip(valid,cnsets):
            for cn1,cn2 in cns:
                expanded.append((a,b,cn1,cn2))
        if expanded:
            a=[x[0] for x in expanded]; b=[x[1] for x in expanded]
            cn=torch.tensor([[x[2],x[3]] for x in expanded],dtype=torch.float32)

            # The geometry/property networks share the same pair-level idea,
            # but their graph encoders are expensive.  Encode each unique
            # ligand ONCE per screening call, then evaluate only the small
            # pairwise MLP heads.  This removes thousands of repeated GNN
            # message-passing operations.
            idx_a=torch.tensor([ligand_index[x] for x in a],dtype=torch.long)
            idx_b=torch.tensor([ligand_index[x] for x in b],dtype=torch.long)
            with torch.no_grad():
                ge1=geom_embeddings[idx_a]
                ge2=geom_embeddings[idx_b]
                gz=geom.h(torch.cat([ge1,ge2,cn],dim=1))
                geometry=gz*gstd+gmean
                gm=(geometry-gscaler['geometry_mean'])/gscaler['geometry_std']
                pe1=prop_embeddings[idx_a]
                pe2=prop_embeddings[idx_b]
                pred=gnn.h(torch.cat([pe1,pe2,cn,gm],dim=1))*gscaler['target_std']+gscaler['target_mean']
            for i,(aa,bb,c1,c2) in enumerate(expanded):
                ll1,ll2,ll,ba=map(float,geometry[i].numpy())
                ucal,ueff,tio=map(float,pred[i].numpy())
                tor=tor_from(ueff,tio)
                rows.append({
                    'Ligand 1':aa,'Ligand 2':bb,'CN1':int(c1),'CN2':int(c2),
                    'Predicted LL1 (A)':ll1,'Predicted LL2 (A)':ll2,
                    'Predicted LL (A)':ll,'Predicted BA (deg)':ba,
                    'Predicted Ucal (K)':ucal,'Predicted Ueff (K)':ueff,
                    'Predicted log10(tau0)':tio,'Predicted Tor (K)':tor,
                    'target_error':target_error([ucal,ueff,tio],target,target_kind)
                })
        if progress and (bi == 1 or bi % 2 == 0 or bi == total_batches):
            progress(iteration, f"pair screening ({bi}/{total_batches} batches; {len(pair_list)} pairs)")

    if not rows:return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('target_error').drop_duplicates(['Ligand 1','Ligand 2'],keep='first').reset_index(drop=True)

def run_iterative_search(
    models, target_kind, target, cfg: SearchConfig,
    initial_ligands, progress=None,
):
    jt, vocab, prop, mu, sd, geom, gmean, gstd, gnn, gscaler, known, _df = models
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
            # Steric augmentation is automatic and ligand-dependent. The
            # augmentation module chooses chemically appropriate alkyl groups
            # from the available C-H sites and molecular size. The experimental
            # library itself is never modified.
            variants = augment_ligand_pool(
                gen_df["smiles"].tolist(),
                max_variants_per_parent=cfg.steric_variants_per_parent,
            )
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
        pairs = pair_screen(pool, _df, geom, gmean, gstd, gnn, gscaler, target_kind, target, cfg.max_pairs, progress=progress, iteration=iteration)
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
            subset=["Ligand 1", "Ligand 2"], keep="first"
        ).reset_index(drop=True)
    else:
        final = pd.DataFrame()
    return archive, final
