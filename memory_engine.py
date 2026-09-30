
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import random
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from steric_augmentation import canonicalize, augment_ligand_pool
from cn_manager import CNManager
from ligand_validation import validate_generated_ligand


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
        self.edge_index = (
            torch.cat(eis, 1)
            if eis else torch.empty((2, 0), dtype=torch.long)
        )
        self.edge_attr = (
            torch.cat(eas, 0)
            if eas else torch.empty((0, 6), dtype=torch.float32)
        )


class GNNEncoder(nn.Module):
    def __init__(self, hidden=64, embed=64, layers=3):
        super().__init__()
        self.n = nn.Linear(18, hidden)
        self.e = nn.Linear(6, hidden)
        self.ms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden)
            )
            for _ in range(layers)
        ])
        self.pr = nn.Sequential(
            nn.Linear(hidden, embed),
            nn.ReLU(),
            nn.Linear(embed, embed)
        )

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
        cnt = torch.bincount(
            g.batch, minlength=n
        ).float().unsqueeze(1)
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
        return self.f(
            torch.cat([self.enc(g1), self.enc(g2), cn], 1)
        )


def atom_features(a):
    vocab = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    f = [
        float(a.GetAtomicNum() == z)
        for z in vocab
    ] + [float(a.GetAtomicNum() not in vocab)]
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
    x = torch.tensor(
        [atom_features(a) for a in m.GetAtoms()],
        dtype=torch.float32
    )
    src, dst, ea = [], [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        src += [i, j]
        dst += [j, i]
        ea += [bf, bf]

    ei = (
        torch.tensor([src, dst], dtype=torch.long)
        if src else torch.empty((2, 0), dtype=torch.long)
    )
    e = (
        torch.tensor(ea, dtype=torch.float32)
        if ea else torch.empty((0, 6), dtype=torch.float32)
    )
    return Graph(x, ei, e)


def tor_from(ueff, tio):
    den = (math.log10(TAU_REF) - float(tio)) * math.log(10.0)
    return float(ueff / den) if den > 0 else float("nan")


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
    return (
        AllChem.GetMorganFingerprintAsBitVect(
            m, 2, nBits=1024
        )
        if m else None
    )


@dataclass
class SearchConfig:
    iterations: int = 5
    starts_per_iteration: int = 16
    latent_steps: int = 80
    latent_lr: float = 0.06
    decode_per_seed: int = 4
    latent_noise: float = 0.35
    max_new_ligands: int = 80
    max_memory_ligands: int = 24
    max_pairs: int = 30000
    top_pairs_per_iteration: int = 20
    steric_variants_per_parent: int = 6
    stage_fraction: float = 0.50
    cn_similarity_threshold: float = 0.35
    validation_similarity_threshold: float = 0.25
    seed: int = 42


class MemoryArchive:
    def __init__(self):
        self.ligands = pd.DataFrame(columns=[
            "smiles", "source", "parent_smiles", "modification",
            "iteration", "stage_target", "latent_target_error",
            "memory_score", "selected"
        ])
        self.pairs = pd.DataFrame()

    def add_ligands(self, df):
        if df is None or len(df) == 0:
            return
        cols = [
            "smiles", "source", "parent_smiles", "modification",
            "iteration", "stage_target", "latent_target_error",
            "memory_score", "selected"
        ]
        x = df.copy()
        for c in cols:
            if c not in x:
                x[c] = (
                    ""
                    if c in ["smiles", "source", "parent_smiles", "modification"]
                    else np.nan
                )
        x = x[cols]
        self.ligands = pd.concat(
            [self.ligands, x], ignore_index=True
        )
        self.ligands = self.ligands.drop_duplicates(
            "smiles", keep="first"
        ).reset_index(drop=True)

    def add_pairs(self, df):
        if df is None or len(df) == 0:
            return
        self.pairs = pd.concat(
            [self.pairs, df], ignore_index=True
        )

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        self.ligands.to_csv(
            folder / "memory_ligands.csv", index=False
        )
        self.pairs.to_csv(
            folder / "memory_pairs.csv", index=False
        )
        with open(
            folder / "memory_summary.json", "w",
            encoding="utf-8"
        ) as f:
            json.dump({
                "ligands": int(len(self.ligands)),
                "pairs": int(len(self.pairs)),
                "iterations": sorted(
                    self.ligands["iteration"]
                    .dropna().unique().tolist()
                ) if len(self.ligands) else [],
            }, f, indent=2)


def build_target_vector(kind, target, mu, sd):
    raw = mu.clone()
    if kind == "Ucal":
        raw[0] = float(target)
    elif kind == "Ueff":
        raw[1] = float(target)
    else:
        # Keep Ueff at the learned mean and solve the requested Tor for tio.
        u = float(mu[1])
        tio = (
            math.log10(TAU_REF)
            - u / max(float(target), 1e-6) / math.log(10.0)
        )
        raw[2] = tio
    return (raw - mu) / sd


def optimize_latent(
    seed_z, target_kind, target, prop, mu, sd, steps, lr
):
    z = seed_z.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    target_z = build_target_vector(
        target_kind, target, mu, sd
    )
    idx = {"Ucal": 0, "Ueff": 1}.get(target_kind)

    for _ in range(steps):
        opt.zero_grad()
        p = prop(z)

        if idx is not None:
            selected = (p[:, idx] - target_z[idx]) ** 2
            others = [j for j in range(3) if j != idx]
            loss = (
                selected.mean()
                + 0.025 * (p[:, others] ** 2).mean()
            )
        else:
            raw = p * sd + mu
            ueff, tio = raw[:, 1], raw[:, 2]
            den = torch.clamp(
                (math.log10(TAU_REF) - tio) * math.log(10.0),
                min=0.25
            )
            tor = ueff / den
            loss = (
                ((tor - float(target)) /
                 max(abs(float(target)), 100.0))
                .pow(2).mean()
            )
            loss = loss + 0.025 * p[:, 0].pow(2).mean()

        loss = loss + 1e-4 * z.pow(2).mean()
        loss.backward()
        opt.step()

    return z.detach()


def encode_ligand_latent(jt, vocab, smiles):
    from fast_jtnn.mol_tree import MolTree
    from fast_jtnn.datautils_prop import set_batch_nodeID
    from fast_jtnn.jtnn_enc import JTNNEncoder
    from fast_jtnn.mpn import MPN

    tree = [MolTree(smiles)]
    set_batch_nodeID(tree, vocab)
    jt_holder, _ = JTNNEncoder.tensorize(tree)
    mpn_holder = MPN.tensorize([smiles])

    with torch.no_grad():
        z_mean, _ = jt.encode_latent(
            jt_holder, mpn_holder
        )
    return z_mean


def _stage_target(
    current_best, final_target, fraction
):
    if current_best is None:
        return float(final_target)

    current_best = float(current_best)
    final_target = float(final_target)

    # Move halfway toward the final target at each generation.
    return current_best + fraction * (final_target - current_best)


def _baseline_best(df, kind, target):
    vals = []
    for _, r in df.iterrows():
        try:
            if kind == "Ucal":
                v = float(r["Ucal"])
            elif kind == "Ueff":
                v = float(r["Ueff"])
            else:
                v = tor_from(float(r["Ueff"]), float(r["tio"]))
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass

    if not vals:
        return None

    # For a high target, start from the highest observed value below target
    # where possible; for a low target, start from the lowest observed value
    # above target. Otherwise use the closest observed value.
    if target >= np.median(vals):
        below = [v for v in vals if v <= target]
        if below:
            return max(below)
    else:
        above = [v for v in vals if v >= target]
        if above:
            return min(above)

    return min(vals, key=lambda x: abs(x - target))


def generate_from_seeds(
    jt, vocab, prop, mu, sd, known,
    target_kind, stage_target,
    seed_smiles, iteration, cfg,
    validator,
):
    random.seed(cfg.seed + iteration)
    torch.manual_seed(cfg.seed + iteration)
    results = {}

    encoded = []
    for s in seed_smiles[:cfg.starts_per_iteration]:
        try:
            encoded.append(
                (s, encode_ligand_latent(jt, vocab, s))
            )
        except Exception:
            continue

    # Keep a small fraction of random starts for exploration, but the dominant
    # search is always memory-seeded.
    n_random = max(1, cfg.starts_per_iteration // 5)
    seeds = encoded + [
        ("", torch.randn(1, 56) * 1.0)
        for _ in range(n_random)
    ]

    for parent, base in seeds:
        for k in range(cfg.decode_per_seed):
            noise = (
                torch.randn_like(base) * cfg.latent_noise
                if k else torch.zeros_like(base)
            )
            z = optimize_latent(
                base + noise,
                target_kind,
                stage_target,
                prop, mu, sd,
                cfg.latent_steps,
                cfg.latent_lr,
            )

            try:
                zt, zm = torch.chunk(z, 2, 1)
                smi = jt.decode(zt, zm, False)
            except Exception:
                smi = None

            if not smi:
                continue

            check = validator(
                smi,
                parent_smiles=parent or None
            )
            if not check.valid:
                continue

            smi = check.smiles
            if smi in known or smi in results:
                continue

            with torch.no_grad():
                raw = (
                    prop(z) * sd + mu
                )[0].numpy()

            results[smi] = {
                "smiles": smi,
                "source": (
                    "jtvae_memory_search"
                    if parent else "jtvae_exploration"
                ),
                "parent_smiles": parent,
                "modification": "latent_optimization",
                "iteration": iteration,
                "stage_target": float(stage_target),
                "latent_target_error": float(
                    target_error(
                        raw, stage_target, target_kind
                    )
                ),
                "memory_score": float(check.max_train_tanimoto),
                "selected": False,
            }

            if len(results) >= cfg.max_new_ligands:
                return list(results.values())

    return list(results.values())


def _best_target_value(pairs, kind):
    if pairs is None or len(pairs) == 0:
        return None

    vals = pd.to_numeric(
        pairs[kind], errors="coerce"
    ).dropna().to_numpy()

    if len(vals) == 0:
        return None
    return float(vals.min())  # placeholder; direction handled below


def _distance_to_target_series(pairs, kind, target):
    if kind == "Ucal":
        return (pairs["Predicted Ucal (K)"] - target).abs()
    if kind == "Ueff":
        return (pairs["Predicted Ueff (K)"] - target).abs()
    return (pairs["Predicted Tor (K)"] - target).abs()


def pair_screen(
    candidates,
    candidate_meta,
    cn_manager,
    gnn,
    gmu,
    gsd,
    target_kind,
    target,
    max_pairs,
    cn_similarity_threshold,
):
    smiles = list(dict.fromkeys(candidates))
    if len(smiles) < 2:
        return pd.DataFrame()

    pair_list = list(combinations_with_replacement(smiles, 2))
    if len(pair_list) > max_pairs:
        # Deterministic truncation, but preserve the newest/memory candidates.
        rng = random.Random(123)
        rng.shuffle(pair_list)
        pair_list = pair_list[:max_pairs]

    rows = []

    for start in range(0, len(pair_list), 128):
        chunk = pair_list[start:start + 128]

        usable = []
        graphs1 = []
        graphs2 = []

        for a, b in chunk:
            meta_a = candidate_meta.get(a, {})
            meta_b = candidate_meta.get(b, {})

            assignments = cn_manager.pair_assignments(
                a, b,
                parent_a=meta_a.get("parent_smiles"),
                parent_b=meta_b.get("parent_smiles"),
                max_per_ligand=2,
                min_similarity=cn_similarity_threshold,
            )
            if not assignments:
                # No defensible CN -> do not send the pair to the GNN.
                continue

            g1 = smiles_graph(a)
            g2 = smiles_graph(b)
            if g1 is None or g2 is None:
                continue

            usable.append(
                (a, b, assignments)
            )
            graphs1.append(g1)
            graphs2.append(g2)

        if not usable:
            continue

        # Encode each graph once. CN changes only the fusion input.
        bg1 = BatchGraph(graphs1)
        bg2 = BatchGraph(graphs2)

        with torch.no_grad():
            h1 = gnn.enc(bg1)
            h2 = gnn.enc(bg2)

            expanded_h1 = []
            expanded_h2 = []
            cn_rows = []
            assignment_meta = []

            for i, (_, _, assignments) in enumerate(usable):
                for ass in assignments:
                    expanded_h1.append(h1[i])
                    expanded_h2.append(h2[i])
                    cn_rows.append(
                        [float(ass.cn1), float(ass.cn2)]
                    )
                    assignment_meta.append(ass)

            h1x = torch.stack(expanded_h1, dim=0)
            h2x = torch.stack(expanded_h2, dim=0)
            cn = torch.tensor(
                cn_rows, dtype=torch.float32
            )

            pred = (
                gnn.f(
                    torch.cat([h1x, h2x, cn], dim=1)
                ) * gsd + gmu
            ).cpu().numpy()

        p = 0
        for a, b, assignments in usable:
            for ass in assignments:
                ucal, ueff, tio = map(float, pred[p])
                tor = tor_from(ueff, tio)

                rows.append({
                    "Ligand 1": a,
                    "Ligand 2": b,
                    "CN1": int(ass.cn1),
                    "CN2": int(ass.cn2),
                    "CN source": ass.source,
                    "CN confidence": float(ass.confidence),
                    "Predicted Ucal (K)": ucal,
                    "Predicted Ueff (K)": ueff,
                    "Predicted log10(tau0)": tio,
                    "Predicted Tor (K)": tor,
                    "target_error": target_error(
                        [ucal, ueff, tio],
                        target,
                        target_kind,
                    ),
                })
                p += 1

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    # Do NOT collapse by ligand pair before CN validity has been considered.
    # Keep the physically supported assignment with the highest CN evidence,
    # then the best target match.
    out = out.sort_values(
        ["CN confidence", "target_error"],
        ascending=[False, True],
    )
    out = out.drop_duplicates(
        subset=["Ligand 1", "Ligand 2"],
        keep="first",
    ).reset_index(drop=True)

    return out.sort_values("target_error").reset_index(drop=True)


def run_iterative_search(
    models, target_kind, target, cfg: SearchConfig,
    initial_ligands, progress=None,
):
    (
        jt, vocab, prop, mu, sd,
        gnn, gmu, gsd, known, df
    ) = models

    cn_manager = CNManager(df)

    # Allowed chemical space is learned from the actual experimental ligand
    # library. Generated ligands cannot introduce arbitrary new elements.
    allowed_atomic_numbers = set()
    max_heavy = 0

    for s in cn_manager.known:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        allowed_atomic_numbers.update(
            a.GetAtomicNum() for a in m.GetAtoms()
        )
        max_heavy = max(
            max_heavy,
            m.GetNumHeavyAtoms()
        )

    # Permit a modest size increase for H->Et/alkyl extrapolation.
    max_generated_heavy = max_heavy + 8

    def validate_candidate(smiles, parent_smiles=None):
        return validate_generated_ligand(
            smiles,
            cn_manager.known,
            allowed_atomic_numbers=allowed_atomic_numbers,
            max_rings=1,
            max_heavy_atoms=max_generated_heavy,
            min_similarity=cfg.validation_similarity_threshold,
            reject_fused_known_fragments=True,
        )

    archive = MemoryArchive()
    memory_seed_smiles = list(dict.fromkeys(initial_ligands))
    candidate_meta = {}

    for s in memory_seed_smiles:
        candidate_meta[s] = {
            "parent_smiles": None,
            "source": "experimental_seed",
        }

    all_generated = []
    current_best = _baseline_best(
        df, target_kind, float(target)
    )

    for iteration in range(1, cfg.iterations + 1):
        stage_target = _stage_target(
            current_best,
            float(target),
            cfg.stage_fraction,
        )

        if progress:
            progress(
                iteration,
                f"generation toward {stage_target:.2f} K "
                f"(final target {target:.2f} K)"
            )

        generated = generate_from_seeds(
            jt, vocab, prop, mu, sd, known,
            target_kind, stage_target,
            memory_seed_smiles,
            iteration,
            cfg,
            validate_candidate,
        )

        if generated:
            gen_df = pd.DataFrame(generated)

            # Automatic H -> Me / Et / larger alkyl modifications.
            variants = augment_ligand_pool(
                gen_df["smiles"].tolist(),
                max_variants_per_parent=cfg.steric_variants_per_parent,
            )

            valid_variants = []
            for v in variants:
                check = validate_candidate(
                    v["smiles"],
                    parent_smiles=v.get("parent_smiles"),
                )
                if not check.valid:
                    continue
                v["smiles"] = check.smiles
                valid_variants.append(v)

            if valid_variants:
                var_df = pd.DataFrame(valid_variants)
                var_df["iteration"] = iteration
                var_df["stage_target"] = stage_target
                var_df["latent_target_error"] = np.nan
                var_df["memory_score"] = np.nan
                var_df["selected"] = False
                gen_df = pd.concat(
                    [gen_df, var_df],
                    ignore_index=True,
                    sort=False,
                )

            archive.add_ligands(gen_df)

            for _, r in gen_df.iterrows():
                s = r["smiles"]
                all_generated.append(s)
                candidate_meta[s] = {
                    "parent_smiles": (
                        r.get("parent_smiles")
                        if pd.notna(r.get("parent_smiles"))
                        else None
                    ),
                    "source": r.get("source", ""),
                }

        # The next screen is dominated by memory seeds and newly generated
        # ligands from THIS iteration.
        pool = list(dict.fromkeys(
            memory_seed_smiles
            + all_generated[-cfg.max_new_ligands:]
        ))
        pool = [
            s for s in pool
            if canonicalize(s)
        ]

        if len(pool) > 120:
            pool = pool[-120:]

        if progress:
            progress(
                iteration,
                f"pair screening ({len(pool)} validated ligands)"
            )

        pairs = pair_screen(
            pool,
            candidate_meta,
            cn_manager,
            gnn,
            gmu,
            gsd,
            target_kind,
            stage_target,
            cfg.max_pairs,
            cfg.cn_similarity_threshold,
        )

        if len(pairs):
            pairs["iteration"] = iteration
            pairs["stage_target"] = stage_target
            pairs["selected"] = False

            # Rank by the CURRENT stage target, not the final target.
            pairs = pairs.sort_values(
                "target_error"
            ).reset_index(drop=True)

            topn = min(
                cfg.top_pairs_per_iteration,
                len(pairs)
            )
            pairs.loc[:topn - 1, "selected"] = True
            archive.add_pairs(
                pairs.head(
                    max(cfg.top_pairs_per_iteration * 3, 50)
                )
            )

            # Determine the best actual result from this generation.
            best_row = pairs.iloc[0]
            if target_kind == "Ucal":
                observed_best = float(
                    best_row["Predicted Ucal (K)"]
                )
            elif target_kind == "Ueff":
                observed_best = float(
                    best_row["Predicted Ueff (K)"]
                )
            else:
                observed_best = float(
                    best_row["Predicted Tor (K)"]
                )

            if current_best is None:
                current_best = observed_best
            else:
                # Move only in the target direction.
                if target >= current_best:
                    current_best = max(
                        current_best, observed_best
                    )
                else:
                    current_best = min(
                        current_best, observed_best
                    )

            # IMPORTANT: winners become the actual seeds for the next
            # generation. This is the memory-augmentation mechanism.
            elite = []
            for _, r in pairs.head(
                cfg.top_pairs_per_iteration
            ).iterrows():
                elite.extend(
                    [r["Ligand 1"], r["Ligand 2"]]
                )

            elite = list(dict.fromkeys(elite))
            memory_seed_smiles = elite[
                :cfg.max_memory_ligands
            ]

            for s in memory_seed_smiles:
                mask = archive.ligands.smiles == s
                archive.ligands.loc[
                    mask, "selected"
                ] = True

        if progress:
            progress(
                iteration,
                f"best={current_best:.2f} K; "
                f"memory={len(memory_seed_smiles)} ligands"
            )

    if len(archive.pairs):
        final = archive.pairs.copy()
        # Final ranking is against the USER'S FINAL target, not the staged
        # target used to generate each iteration.
        final["final_target_error"] = _distance_to_target_series(
            final, target_kind, float(target)
        ) / max(abs(float(target)), 100.0)

        final = final.sort_values(
            "final_target_error"
        ).drop_duplicates(
            subset=[
                "Ligand 1", "Ligand 2",
                "CN1", "CN2"
            ],
            keep="first",
        ).reset_index(drop=True)

        final["target_error"] = final["final_target_error"]
        final = final.drop(
            columns=["final_target_error"]
        )
    else:
        final = pd.DataFrame()

    return archive, final
