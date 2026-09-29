from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from rdkit import Chem

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from fast_jtnn import JTNNVAE, Vocab
from memory_engine import PropNet, PairGNN, SearchConfig, run_iterative_search, tor_from

st.set_page_config(
    page_title="ML3 Memory-Augmented JT-VAE Ligand Generator",
    page_icon="🧪",
    layout="wide",
)

DATA = BASE / "ML3_Ucal_Ueff_tio_2.csv"
JT_MODEL = BASE / "true_jtvae_model"
JT_VOCAB = BASE / "true_jtvae_vocab.txt"
LATENT = BASE / "latent_oracle"
GNN_DIR = BASE / "gnn_oracle"
PREGEN = BASE / "generated_candidates" / "generated_candidates.csv"


def canonical(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m, canonical=True) if m else None


def target_from_row(row, kind):
    if kind == "Ucal":
        return float(row["Ucal"])
    if kind == "Ueff":
        return float(row["Ueff"])
    return tor_from(float(row["Ueff"]), float(row["tio"]))


@st.cache_resource

def load_models():
    cfg = json.loads((JT_MODEL / "config.json").read_text())
    vocab = Vocab([x.strip() for x in JT_VOCAB.read_text().splitlines() if x.strip()])
    jt = JTNNVAE(
        vocab,
        cfg["hidden_size"],
        cfg["latent_size"],
        cfg["depthT"],
        cfg["depthG"],
    )
    jt.load_state_dict(torch.load(JT_MODEL / "best_model.pt", map_location="cpu", weights_only=False))
    jt.eval()

    prop = PropNet()
    prop.load_state_dict(torch.load(LATENT / "property_oracle.pt", map_location="cpu", weights_only=False))
    prop.eval()
    sc = pd.read_csv(LATENT / "property_scaler.csv")
    mu = torch.tensor(sc["mean"].values, dtype=torch.float32)
    sd = torch.tensor(sc["std"].values, dtype=torch.float32)

    ck = torch.load(GNN_DIR / "model.pt", map_location="cpu", weights_only=False)
    gc = ck.get("config", {})
    gnn = PairGNN(gc.get("hidden", 64), gc.get("embed", 64), gc.get("gnn_layers", 3))
    gnn.load_state_dict(ck["model_state_dict"])
    gnn.eval()
    gmu = torch.tensor(ck["target_mean"], dtype=torch.float32)
    gsd = torch.tensor(ck["target_std"], dtype=torch.float32)

    df = pd.read_csv(DATA)
    known = set()
    for col in ["L1", "L2"]:
        for s in df[col].dropna().astype(str):
            c = canonical(s)
            if c:
                known.add(c)

    return jt, vocab, prop, mu, sd, gnn, gmu, gsd, known, df


def initial_seed_pool(df, kind, target, generated_path, n=24):
    work = df.copy()
    work["target_for_search"] = work.apply(lambda r: target_from_row(r, kind), axis=1)
    work["distance"] = (work["target_for_search"] - float(target)).abs()
    work = work.sort_values("distance")

    seeds = []
    for _, r in work.head(max(n, 8)).iterrows():
        for c in [canonical(r["L1"]), canonical(r["L2"])]:
            if c and c not in seeds:
                seeds.append(c)
            if len(seeds) >= n:
                break
        if len(seeds) >= n:
            break

    if generated_path.exists():
        try:
            g = pd.read_csv(generated_path)
            for col in ["smiles", "SMILES", "L1", "L2"]:
                if col in g.columns:
                    for s in g[col].dropna().astype(str):
                        c = canonical(s)
                        if c and c not in seeds:
                            seeds.append(c)
                        if len(seeds) >= n * 2:
                            break
                    break
        except Exception:
            pass
    return seeds[:n]


def in_dataset_range(df, kind, target):
    vals = df.apply(lambda r: target_from_row(r, kind), axis=1)
    return float(vals.min()), float(vals.max()), float(target) < float(vals.min()) or float(target) > float(vals.max())


st.title("🧪 ML3 Memory-Augmented JT-VAE Target-Directed Ligand Generator")
st.caption(
    "Target → JT-VAE latent extrapolation → ligand modification → pair GNN screening → memory → next iteration"
)

try:
    models = load_models()
except Exception as e:
    st.error(f"Model loading failed: {type(e).__name__}: {e}")
    st.stop()

_, _, _, _, _, _, _, _, known, df = models

with st.sidebar:
    st.header("Target")
    target_kind = st.selectbox("Optimize target", ["Ueff", "Ucal", "Tor"])
    default = {"Ueff": 3000.0, "Ucal": 3000.0, "Tor": 100.0}[target_kind]
    target = st.number_input(
        f"Target {target_kind} (K)",
        min_value=1.0,
        value=default,
        step=50.0 if target_kind != "Tor" else 1.0,
    )

    st.header("Dy coordination")
    cn1 = st.number_input("CN of ligand 1", min_value=1, max_value=8, value=2, step=1)
    cn2 = st.number_input("CN of ligand 2", min_value=1, max_value=8, value=2, step=1)

    st.header("Memory search")
    iterations = st.slider("Iterations", 1, 8, 4)
    starts = st.slider("Latent starts / iteration", 4, 32, 12)
    steps = st.slider("Latent optimization steps", 20, 150, 60)
    decode_per_seed = st.slider("Decodes / seed", 1, 6, 3)
    new_ligands = st.slider("New ligands / iteration", 10, 120, 60)
    top_pairs = st.slider("Elite pairs retained / iteration", 5, 50, 20)

    st.header("Steric expansion")
    me = st.checkbox("H → Me", True)
    et = st.checkbox("H → Et", True)
    npr = st.checkbox("H → nPr", False)
    ipr = st.checkbox("H → iPr", False)
    tbu = st.checkbox("H → tBu", False)
    variants = st.slider("Max steric variants / generated ligand", 1, 12, 6)

    st.header("Reproducibility")
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

lo, hi, extrapolating = in_dataset_range(df, target_kind, target)
if extrapolating:
    st.warning(
        f"Extrapolation mode: requested {target_kind} = {target:g} K is outside the dataset-derived range "
        f"({lo:.1f}–{hi:.1f} K). The app will still search the learned latent space and iteratively refine candidates."
    )
else:
    st.info(f"Target is inside the dataset-derived {target_kind} range: {lo:.1f}–{hi:.1f} K.")

c1, c2, c3 = st.columns(3)
c1.metric("Dataset complexes", f"{len(df):,}")
c2.metric("Known unique ligands", f"{len(known):,}")
c3.metric("Search mode", "EXTRAPOLATION" if extrapolating else "INTERPOLATION")

st.markdown("### How this version extrapolates")
st.markdown(
    """
1. **JT-VAE** searches its continuous 56-dimensional latent space toward the requested target.
2. Decoded ligands are **chemically augmented** by controlled C–H → alkyl substitutions (Me/Et and optional larger groups).
3. The **two-ligand GNN** evaluates L1 + L2 rather than evaluating a ligand in isolation.
4. The best pairs become **memory**.
5. Their ligands are encoded back into JT-VAE latent space and become seeds for the **next iteration**, together with fresh random starts.
6. This repeats for the requested number of iterations; the final table is the accumulated target-ranked ligand combinations.
"""
)

if st.button("🚀 Run memory-augmented extrapolative search", type="primary", use_container_width=True):
    cfg = SearchConfig(
        iterations=iterations,
        starts_per_iteration=starts,
        latent_steps=steps,
        decode_per_seed=decode_per_seed,
        max_new_ligands=new_ligands,
        top_pairs_per_iteration=top_pairs,
        steric_variants_per_parent=variants,
        use_me=me,
        use_et=et,
        use_npr=npr,
        use_ipr=ipr,
        use_tbu=tbu,
        seed=int(seed),
    )

    seeds = initial_seed_pool(df, target_kind, target, PREGEN, n=min(24, starts * 2))
    st.write(f"Initial memory seeds: **{len(seeds)}**")

    status = st.empty()
    bar = st.progress(0)

    def progress(iteration, message):
        status.info(f"Iteration {iteration}/{iterations}: {message}")
        bar.progress(min(iteration / iterations, 1.0))

    with st.spinner("Running JT-VAE → steric modification → pair GNN → memory loop..."):
        archive, final = run_iterative_search(
            models,
            target_kind,
            float(target),
            float(cn1),
            float(cn2),
            cfg,
            seeds,
            progress=progress,
        )

    st.session_state["archive"] = archive
    st.session_state["final"] = final
    st.session_state["target_kind"] = target_kind
    st.session_state["target"] = float(target)
    status.success("Iterative memory search completed.")

if "final" in st.session_state:
    final = st.session_state["final"]
    archive = st.session_state["archive"]
    target_kind = st.session_state["target_kind"]
    target = st.session_state["target"]

    st.markdown("## Final target-ranked ligand combinations")
    if len(final):
        show = final.copy()
        show.insert(0, "Rank", np.arange(1, len(show) + 1))
        st.dataframe(show.head(100), use_container_width=True, hide_index=True)

        st.download_button(
            "Download final ligand combinations",
            show.to_csv(index=False).encode("utf-8"),
            file_name=f"ML3_{target_kind}_{target:g}K_final_ligand_combinations.csv",
            mime="text/csv",
        )
    else:
        st.warning("No valid pair predictions were produced. Increase iterations/starts or reduce the target distance.")

    a, b, c = st.columns(3)
    a.metric("Memory ligands", f"{len(archive.ligands):,}")
    b.metric("Memory pair records", f"{len(archive.pairs):,}")
    c.metric("Final unique pairs", f"{len(final):,}")

    st.markdown("## Memory / lineage")
    if len(archive.ligands):
        st.dataframe(archive.ligands.sort_values(["iteration", "selected"], ascending=[True, False]).head(300), use_container_width=True, hide_index=True)
        st.download_button(
            "Download ligand memory",
            archive.ligands.to_csv(index=False).encode("utf-8"),
            file_name="ML3_memory_ligands.csv",
            mime="text/csv",
        )
    if len(archive.pairs):
        st.download_button(
            "Download pair memory",
            archive.pairs.to_csv(index=False).encode("utf-8"),
            file_name="ML3_memory_pairs.csv",
            mime="text/csv",
        )

    st.markdown("## What was actually modified?")
    st.write(
        "Only generated JT-VAE ligands are sterically augmented. The original experimental ligand library is not chemically modified. "
        "Every modified ligand retains its parent SMILES and modification label in the memory table."
    )
