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
    layout="wide",
)

DATA = BASE / "ML3_Ucal_Ueff_tio_2.csv"
JT_MODEL = BASE / "true_jtvae_model"
JT_VOCAB = BASE / "true_jtvae_vocab.txt"
LATENT = BASE / "latent_oracle"
GNN_DIR = BASE / "gnn_oracle"
PREGEN = BASE / "generated_candidates" / "generated_candidates.csv"

# Fixed production settings. Only the number of iterative memory rounds is
# exposed to the user. These values are deliberately not presented as UI
# hyperparameters so that the deployed workflow remains consistent.
DEFAULT_ITERATIONS = 4
FIXED_SEED = 42
FIXED_STARTS = 16
FIXED_LATENT_STEPS = 80
FIXED_DECODE_PER_SEED = 4
FIXED_NEW_LIGANDS = 80
FIXED_TOP_PAIRS = 20
FIXED_STERIC_VARIANTS = 6
FIXED_MAX_PAIRS = 30000


def canonical(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m, canonical=True) if m else None


def target_from_row(row, kind):
    if kind == "Ucal":
        return float(row["Ucal"])
    if kind == "Ueff":
        return float(row["Ueff"])
    return tor_from(float(row["Ueff"]), float(row["tio"]))


def available_cn_pairs(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Return all CN1/CN2 combinations represented in the training data."""
    required = {"CN1", "CN2"}
    if not required.issubset(df.columns):
        raise RuntimeError("Training dataset must contain CN1 and CN2 columns.")

    pairs = set()
    for _, row in df[["CN1", "CN2"]].dropna().iterrows():
        try:
            pairs.add((int(row["CN1"]), int(row["CN2"])))
        except (TypeError, ValueError):
            continue

    if not pairs:
        raise RuntimeError("No valid coordination-number pairs were found in the training dataset.")

    return sorted(pairs)


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
    jt.load_state_dict(
        torch.load(
            JT_MODEL / "best_model.pt",
            map_location="cpu",
            weights_only=False,
        )
    )
    jt.eval()

    # The latent property-oracle checkpoint is accepted in either raw
    # state_dict or {'model_state_dict': ...} format.
    prop_ckpt = torch.load(
        LATENT / "property_oracle.pt",
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(prop_ckpt, dict) and "model_state_dict" in prop_ckpt:
        prop_state = prop_ckpt["model_state_dict"]
    elif isinstance(prop_ckpt, dict):
        prop_state = prop_ckpt
    else:
        raise RuntimeError(
            "property_oracle.pt is neither a PyTorch state_dict nor a wrapped checkpoint."
        )

    latent_dim = int(
        prop_ckpt.get("latent_dim", prop_state["net.0.weight"].shape[1])
        if isinstance(prop_ckpt, dict)
        else prop_state["net.0.weight"].shape[1]
    )

    prop = PropNet(d=latent_dim)
    prop.load_state_dict(prop_state, strict=True)
    prop.eval()

    scaler_path = LATENT / "property_scaler.csv"
    if scaler_path.exists():
        sc = pd.read_csv(scaler_path)
        if not {"mean", "std"}.issubset(sc.columns):
            raise RuntimeError("property_scaler.csv must contain mean and std columns.")
        mu = torch.tensor(sc["mean"].values, dtype=torch.float32)
        sd = torch.tensor(sc["std"].values, dtype=torch.float32)
    elif isinstance(prop_ckpt, dict) and {"target_mean", "target_std"}.issubset(prop_ckpt):
        mu = torch.tensor(prop_ckpt["target_mean"], dtype=torch.float32).reshape(-1)
        sd = torch.tensor(prop_ckpt["target_std"], dtype=torch.float32).reshape(-1)
    else:
        raise FileNotFoundError(
            "No latent-oracle property scaler was found."
        )

    if len(mu) != 3 or len(sd) != 3:
        raise RuntimeError("The latent property oracle must have three targets: Ucal, Ueff and tio.")
    sd = torch.clamp(sd, min=1e-8)

    ck = torch.load(GNN_DIR / "model.pt", map_location="cpu", weights_only=False)
    gc = ck.get("config", {})
    gnn = PairGNN(
        gc.get("hidden", 64),
        gc.get("embed", 64),
        gc.get("gnn_layers", 3),
    )
    gnn.load_state_dict(ck["model_state_dict"], strict=True)
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

    cn_pairs = available_cn_pairs(df)
    return jt, vocab, prop, mu, sd, gnn, gmu, gsd, known, df, cn_pairs


def initial_seed_pool(df, kind, target, generated_path, n=24):
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
        for c in [canonical(r["L1"]), canonical(r["L2"])]:
            if c and c not in seeds:
                seeds.append(c)
            if len(seeds) >= n:
                break
        if len(seeds) >= n:
            break

    # A previous generated library may provide additional starting chemistry,
    # but it is only a seed source; it is not treated as experimental data.
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


st.title("ML3 Memory-Augmented JT-VAE")
st.caption("Target-directed ligand-pair generation with iterative memory refinement")

try:
    models = load_models()
except Exception as exc:
    st.error(f"Model loading failed: {type(exc).__name__}: {exc}")
    st.stop()

_, _, _, _, _, _, _, _, known, df, cn_pairs = models

with st.sidebar:
    st.header("Target")
    target_kind = st.selectbox(
        "Optimize target",
        ["Ueff", "Ucal", "Tor"],
    )
    default = {"Ueff": 3000.0, "Ucal": 3000.0, "Tor": 100.0}[target_kind]
    target = st.number_input(
        f"Target {target_kind} (K)",
        min_value=1.0,
        value=default,
        step=50.0 if target_kind != "Tor" else 1.0,
    )

    st.header("Iterations")
    iterations = st.slider(
        "Memory iterations",
        min_value=1,
        max_value=8,
        value=DEFAULT_ITERATIONS,
    )

if st.button(
    "Run target-directed search",
    type="primary",
    use_container_width=True,
):
    cfg = SearchConfig(
        iterations=int(iterations),
        starts_per_iteration=FIXED_STARTS,
        latent_steps=FIXED_LATENT_STEPS,
        decode_per_seed=FIXED_DECODE_PER_SEED,
        max_new_ligands=FIXED_NEW_LIGANDS,
        top_pairs_per_iteration=FIXED_TOP_PAIRS,
        steric_variants_per_parent=FIXED_STERIC_VARIANTS,
        max_pairs=FIXED_MAX_PAIRS,
        seed=FIXED_SEED,
    )

    seeds = initial_seed_pool(
        df,
        target_kind,
        target,
        PREGEN,
        n=24,
    )

    status = st.empty()
    bar = st.progress(0)

    def progress(iteration, message):
        status.info(f"Iteration {iteration}/{iterations}: {message}")
        bar.progress(min(iteration / iterations, 1.0))

    with st.spinner("Running target-directed JT-VAE memory search..."):
        archive, final = run_iterative_search(
            models,
            target_kind,
            float(target),
            cn_pairs,
            cfg,
            seeds,
            progress=progress,
        )

    st.session_state["archive"] = archive
    st.session_state["final"] = final
    st.session_state["target_kind"] = target_kind
    st.session_state["target"] = float(target)
    status.success("Search completed.")


if "final" in st.session_state:
    final = st.session_state["final"]
    archive = st.session_state["archive"]
    target_kind = st.session_state["target_kind"]
    target = st.session_state["target"]

    st.markdown("## Target-ranked ligand combinations")

    if len(final):
        show = final.copy()
        show.insert(0, "Rank", np.arange(1, len(show) + 1))

        preferred = [
            "Rank",
            "Ligand 1",
            "Ligand 2",
            "CN1",
            "CN2",
            "Predicted Ucal (K)",
            "Predicted Ueff (K)",
            "Predicted log10(tau0)",
            "Predicted Tor (K)",
            "target_error",
            "iteration",
        ]
        cols = [c for c in preferred if c in show.columns]
        remaining = [c for c in show.columns if c not in cols]
        show = show[cols + remaining]

        st.dataframe(
            show.head(100),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download results",
            show.to_csv(index=False).encode("utf-8"),
            file_name=f"ML3_{target_kind}_{target:g}K_ligand_combinations.csv",
            mime="text/csv",
        )
    else:
        st.warning("No valid ligand-pair predictions were produced.")

    with st.expander("Search provenance"):
        st.caption(
            "Coordination numbers were evaluated automatically across the CN1/CN2 combinations represented in the training dataset. "
            "Steric diversification was applied automatically according to each generated ligand's available C–H sites and molecular size."
        )

        if len(archive.ligands):
            st.dataframe(
                archive.ligands.sort_values(
                    ["iteration", "selected"],
                    ascending=[True, False],
                ).head(300),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download ligand lineage",
                archive.ligands.to_csv(index=False).encode("utf-8"),
                file_name="ML3_ligand_lineage.csv",
                mime="text/csv",
            )

        if len(archive.pairs):
            st.download_button(
                "Download pair memory",
                archive.pairs.to_csv(index=False).encode("utf-8"),
                file_name="ML3_pair_memory.csv",
                mime="text/csv",
            )
