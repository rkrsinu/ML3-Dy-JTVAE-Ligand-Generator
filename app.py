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
from memory_engine import PropNet, SearchConfig, run_iterative_search, tor_from
from geometry_model import load_geometry_model, load_property_model

st.set_page_config(
    page_title="ML3 Memory-Augmented JT-VAE Ligand Generator",
    page_icon="🧪",
    layout="wide",
)

DATA = BASE / "ML3_Ucal_Ueff_tio_2.csv"
JT_MODEL = BASE / "true_jtvae_model"
JT_VOCAB = BASE / "true_jtvae_vocab.txt"
LATENT = BASE / "latent_oracle"
GNN_DIR = BASE / "geometry_models"
GEOM_DATA = BASE / "all_BL_BA_SMILES.xlsx"
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

    # ------------------------------------------------------------
    # Latent property oracle
    # ------------------------------------------------------------
    # The ML3 checkpoint has existed in two formats:
    #   A) raw state_dict: {"net.0.weight": ..., ...}
    #   B) wrapped checkpoint: {"model_state_dict": {...}, ...}
    # The deployment loader must support BOTH formats.
    prop_ckpt = torch.load(
        LATENT / "property_oracle.pt",
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(prop_ckpt, dict) and "model_state_dict" in prop_ckpt:
        prop_state = prop_ckpt["model_state_dict"]
    elif isinstance(prop_ckpt, dict):
        # Raw PyTorch state_dict.
        prop_state = prop_ckpt
    else:
        raise RuntimeError(
            "property_oracle.pt is neither a PyTorch state_dict nor a "
            "wrapped checkpoint containing 'model_state_dict'."
        )

    # The trained ML3 latent representation is 56-dimensional. Prefer the
    # checkpoint/config value when available, otherwise infer it directly
    # from net.0.weight.
    latent_dim = int(
        prop_ckpt.get("latent_dim", prop_state["net.0.weight"].shape[1])
        if isinstance(prop_ckpt, dict)
        else prop_state["net.0.weight"].shape[1]
    )

    prop = PropNet(d=latent_dim)
    try:
        prop.load_state_dict(prop_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Latent property-oracle architecture does not match "
            "property_oracle.pt.\n\n"
            f"Checkpoint keys: {list(prop_state.keys())[:12]}\n"
            f"Latent dimension inferred: {latent_dim}\n\n"
            "Do not use partial/random loading."
        ) from exc
    prop.eval()

    # The scaler is part of the trained oracle and is required to convert
    # standardized oracle outputs back to physical units.
    scaler_path = LATENT / "property_scaler.csv"
    if scaler_path.exists():
        sc = pd.read_csv(scaler_path)
        required_scaler_cols = {"mean", "std"}
        if not required_scaler_cols.issubset(sc.columns):
            raise RuntimeError(
                "property_scaler.csv must contain 'mean' and 'std' columns."
            )
        mu = torch.tensor(sc["mean"].values, dtype=torch.float32)
        sd = torch.tensor(sc["std"].values, dtype=torch.float32)
    elif isinstance(prop_ckpt, dict) and {"target_mean", "target_std"}.issubset(prop_ckpt):
        mu = torch.tensor(prop_ckpt["target_mean"], dtype=torch.float32).reshape(-1)
        sd = torch.tensor(prop_ckpt["target_std"], dtype=torch.float32).reshape(-1)
    else:
        raise FileNotFoundError(
            "Neither latent_oracle/property_scaler.csv nor target_mean/target_std "
            "metadata was found in property_oracle.pt."
        )

    if len(mu) != 3 or len(sd) != 3:
        raise RuntimeError(
            f"Property oracle scaler must contain 3 targets (Ucal, Ueff, tio); "
            f"found mean={len(mu)}, std={len(sd)}."
        )

    sd = torch.clamp(sd, min=1e-8)

    geom, gmean, gstd = load_geometry_model(GNN_DIR / "geometry_model.pt")
    gnn, gscaler = load_property_model(GNN_DIR / "geometry_aware_property_gnn.pt")

    df = pd.read_csv(DATA)
    geometry_df = pd.read_excel(GEOM_DATA)
    # Geometry-aware models use the Excel ligand-pair/geometry dataset as the
    # coordination environment reference during screening.
    df = geometry_df
    known = set()
    for col in ["L1_SMILES", "L2_SMILES"]:
        for s in df[col].dropna().astype(str):
            c = canonical(s)
            if c:
                known.add(c)

    return jt, vocab, prop, mu, sd, geom, gmean, gstd, gnn, gscaler, known, df


def initial_seed_pool(df, kind, target, generated_path, n=24):
    work = df.copy()
    work["target_for_search"] = work.apply(lambda r: target_from_row(r, kind), axis=1)
    work["distance"] = (work["target_for_search"] - float(target)).abs()
    work = work.sort_values("distance")

    seeds = []
    for _, r in work.head(max(n, 8)).iterrows():
        for c in [canonical(r["L1_SMILES"]), canonical(r["L2_SMILES"])]:
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



st.title("🧪 ML3 Memory-Augmented JT-VAE Target-Directed Ligand Generator")
st.caption(
    "Property-guided inverse design of Dy(III) ligand pairs using JT-VAE latent search and pair-level oracle screening."
)

try:
    models = load_models()
except Exception as e:
    st.error(f"Model loading failed: {type(e).__name__}: {e}")
    st.stop()

_, _, _, _, _, _, _, _, _, _, known, df = models

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

    st.header("Search")
    iterations = st.slider("Iterations", 1, 8, 5)

if st.button("Run target-directed search", type="primary", use_container_width=True):
    # Search settings are deliberately fixed internally so the public interface
    # exposes only the scientifically meaningful controls.
    cfg = SearchConfig(
        iterations=int(iterations),
        starts_per_iteration=16,
        latent_steps=80,
        latent_lr=0.06,
        decode_per_seed=4,
        latent_noise=0.35,
        max_new_ligands=80,
        max_memory_ligands=120,
        max_pairs=30000,
        top_pairs_per_iteration=20,
        steric_variants_per_parent=6,
        seed=42,
    )

    seeds = initial_seed_pool(df, target_kind, target, PREGEN, n=24)

    status = st.empty()
    bar = st.progress(0)

    def progress(iteration, message):
        status.info(f"Iteration {iteration}/{iterations}: {message}")
        bar.progress(min(iteration / iterations, 1.0))

    with st.spinner("Running target-directed JT-VAE search and pair-oracle screening..."):
        archive, final = run_iterative_search(
            models,
            target_kind,
            float(target),
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
        st.warning("No valid ligand pairs were found for the requested target.")


