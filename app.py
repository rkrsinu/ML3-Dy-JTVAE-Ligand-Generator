
from __future__ import annotations

import json
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
from memory_engine import (
    PropNet,
    PairGNN,
    SearchConfig,
    run_iterative_search,
    tor_from,
)


st.set_page_config(
    page_title="ML3 Memory-Augmented JT-VAE",
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
    return tor_from(
        float(row["Ueff"]),
        float(row["tio"])
    )


@st.cache_resource
def load_models():
    # ---------------- JT-VAE ----------------
    cfg = json.loads(
        (JT_MODEL / "config.json").read_text()
    )
    vocab = Vocab([
        x.strip()
        for x in JT_VOCAB.read_text().splitlines()
        if x.strip()
    ])

    jt = JTNNVAE(
        vocab,
        cfg["hidden_size"],
        cfg["latent_size"],
        cfg["depthT"],
        cfg["depthG"],
    )
    jt.load_state_dict(
        torch.load(
            JT_MODEL / "best_model.pt",
            map_location="cpu",
            weights_only=False,
        )
    )
    jt.eval()

    # ---------------- latent property oracle ----------------
    prop_ckpt = torch.load(
        LATENT / "property_oracle.pt",
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(prop_ckpt, dict)
        and "model_state_dict" in prop_ckpt
    ):
        prop_state = prop_ckpt["model_state_dict"]
    elif isinstance(prop_ckpt, dict):
        prop_state = prop_ckpt
    else:
        raise RuntimeError(
            "Invalid property_oracle.pt format."
        )

    latent_dim = int(
        prop_ckpt.get(
            "latent_dim",
            prop_state["net.0.weight"].shape[1],
        )
        if isinstance(prop_ckpt, dict)
        else prop_state["net.0.weight"].shape[1]
    )

    prop = PropNet(d=latent_dim)
    prop.load_state_dict(
        prop_state,
        strict=True,
    )
    prop.eval()

    scaler_path = LATENT / "property_scaler.csv"

    if scaler_path.exists():
        sc = pd.read_csv(scaler_path)
        mu = torch.tensor(
            sc["mean"].values,
            dtype=torch.float32,
        )
        sd = torch.tensor(
            sc["std"].values,
            dtype=torch.float32,
        )
    elif (
        isinstance(prop_ckpt, dict)
        and "target_mean" in prop_ckpt
        and "target_std" in prop_ckpt
    ):
        mu = torch.tensor(
            prop_ckpt["target_mean"],
            dtype=torch.float32,
        ).reshape(-1)
        sd = torch.tensor(
            prop_ckpt["target_std"],
            dtype=torch.float32,
        ).reshape(-1)
    else:
        raise FileNotFoundError(
            "Latent property scaler not found."
        )

    if len(mu) != 3 or len(sd) != 3:
        raise RuntimeError(
            "Property scaler must contain Ucal, Ueff and tio."
        )

    sd = torch.clamp(sd, min=1e-8)

    # ---------------- pair GNN ----------------
    ck = torch.load(
        GNN_DIR / "model.pt",
        map_location="cpu",
        weights_only=False,
    )
    gc = ck.get("config", {})

    gnn = PairGNN(
        gc.get("hidden", 64),
        gc.get("embed", 64),
        gc.get("gnn_layers", 3),
    )

    gnn.load_state_dict(
        ck["model_state_dict"],
        strict=True,
    )
    gnn.eval()

    gmu = torch.tensor(
        ck["target_mean"],
        dtype=torch.float32,
    )
    gsd = torch.tensor(
        ck["target_std"],
        dtype=torch.float32,
    )

    df = pd.read_csv(DATA)

    known = set()
    for col in ["L1", "L2"]:
        for s in df[col].dropna().astype(str):
            c = canonical(s)
            if c:
                known.add(c)

    return (
        jt, vocab, prop, mu, sd,
        gnn, gmu, gsd, known, df
    )


def initial_seed_pool(
    df, kind, target, generated_path, n=24
):
    work = df.copy()
    work["target_for_search"] = work.apply(
        lambda r: target_from_row(r, kind),
        axis=1,
    )
    work["distance"] = (
        work["target_for_search"] - float(target)
    ).abs()
    work = work.sort_values("distance")

    seeds = []

    for _, r in work.head(max(n, 8)).iterrows():
        for c in [
            canonical(r["L1"]),
            canonical(r["L2"]),
        ]:
            if c and c not in seeds:
                seeds.append(c)
            if len(seeds) >= n:
                break
        if len(seeds) >= n:
            break

    # Existing generated candidates can provide additional starting diversity.
    if generated_path.exists():
        try:
            g = pd.read_csv(generated_path)
            for col in [
                "smiles", "SMILES", "L1", "L2"
            ]:
                if col not in g.columns:
                    continue
                for s in g[col].dropna().astype(str):
                    c = canonical(s)
                    if c and c not in seeds:
                        seeds.append(c)
                    if len(seeds) >= n:
                        break
                break
        except Exception:
            pass

    return seeds[:n]


st.title("ML3 Memory-Augmented JT-VAE")
st.caption(
    "Target-directed ligand-pair generation with iterative memory refinement"
)

try:
    models = load_models()
except Exception as e:
    st.error(
        f"Model loading failed: {type(e).__name__}: {e}"
    )
    st.stop()

_, _, _, _, _, _, _, _, _, df = models

col1, col2, col3 = st.columns(3)

with col1:
    target_kind = st.selectbox(
        "Target property",
        ["Ueff", "Ucal", "Tor"],
    )

with col2:
    default = {
        "Ueff": 3000.0,
        "Ucal": 3000.0,
        "Tor": 100.0,
    }[target_kind]

    target = st.number_input(
        f"Target {target_kind} (K)",
        min_value=1.0,
        value=default,
        step=50.0 if target_kind != "Tor" else 1.0,
    )

with col3:
    iterations = st.number_input(
        "Iterations",
        min_value=1,
        max_value=12,
        value=5,
        step=1,
    )

if st.button(
    "Run target-directed search",
    type="primary",
    use_container_width=True,
):
    cfg = SearchConfig(
        iterations=int(iterations),

        # Internal search settings. The UI intentionally stays minimal.
        starts_per_iteration=16,
        latent_steps=80,
        latent_lr=0.06,
        decode_per_seed=4,
        latent_noise=0.35,

        max_new_ligands=80,
        max_memory_ligands=24,
        max_pairs=30000,
        top_pairs_per_iteration=20,

        # Automatic H -> Me / Et / larger alkyl exploration.
        steric_variants_per_parent=6,

        # Each generation moves halfway from the current best toward
        # the final target rather than jumping directly to it.
        stage_fraction=0.50,

        # CN is inferred only from observed ligand/role evidence or
        # structurally similar observed ligands.
        cn_similarity_threshold=0.35,

        # Generated ligands must remain connected to the learned chemical space.
        validation_similarity_threshold=0.25,

        seed=42,
    )

    seeds = initial_seed_pool(
        df,
        target_kind,
        float(target),
        PREGEN,
        n=24,
    )

    status = st.empty()
    bar = st.progress(0.0)

    def progress(iteration, message):
        status.info(
            f"Iteration {iteration}/{iterations}: {message}"
        )
        bar.progress(
            min(iteration / float(iterations), 1.0)
        )

    with st.spinner(
        "Running iterative JT-VAE + memory + PairGNN search..."
    ):
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

    status.success(
        "Iterative memory search completed."
    )

if "final" in st.session_state:
    final = st.session_state["final"]
    target_kind = st.session_state["target_kind"]
    target = st.session_state["target"]

    st.markdown(
        "## Final target-ranked ligand combinations"
    )

    if len(final):
        show = final.copy()
        show.insert(
            0,
            "Rank",
            np.arange(1, len(show) + 1),
        )

        # Keep the table scientific and readable.
        preferred_cols = [
            "Rank",
            "Ligand 1",
            "Ligand 2",
            "CN1",
            "CN2",
            "CN source",
            "Predicted Ucal (K)",
            "Predicted Ueff (K)",
            "Predicted log10(tau0)",
            "Predicted Tor (K)",
            "target_error",
            "iteration",
            "stage_target",
            "selected",
        ]
        preferred_cols = [
            c for c in preferred_cols
            if c in show.columns
        ]

        st.dataframe(
            show[preferred_cols].head(100),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download final ligand combinations",
            show.to_csv(
                index=False
            ).encode("utf-8"),
            file_name=(
                f"ML3_{target_kind}_"
                f"{target:g}K_final_ligand_combinations.csv"
            ),
            mime="text/csv",
        )
    else:
        st.warning(
            "No valid ligand pairs were found for the requested target."
        )
