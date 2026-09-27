import json
import math
import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors

# -----------------------------------------------------------------------------
# App paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT))

TREF = 100.0  # s, fixed as requested
LATENT_DIM = 56

st.set_page_config(
    page_title="Dy-SMM JT-VAE Generator",
    page_icon="🧲",
    layout="wide",
)


class PropNet(nn.Module):
    def __init__(self, d=56, h=256, out=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, out),
        )

    def forward(self, x):
        return self.net(x)


def tor_from_ueff_tio(ueff, tio, tref=TREF):
    """T_or = -Ueff / ln(tau0/tref), with tio = log10(tau0)."""
    tau0 = 10.0 ** np.asarray(tio, dtype=float)
    denom = np.log(tau0 / tref)
    with np.errstate(divide="ignore", invalid="ignore"):
        tor = -np.asarray(ueff, dtype=float) / denom
    return tor


def tio_from_ueff_tor(ueff, tor, tref=TREF):
    """Given Ueff and T_or, calculate log10(tau0)."""
    if tor <= 0:
        raise ValueError("T_or must be greater than zero.")
    if ueff <= 0:
        raise ValueError("Ueff must be greater than zero.")
    # tau0 = tref * exp(-Ueff/Tor)
    return math.log10(tref) - ueff / (tor * math.log(10.0))


@st.cache_resource(show_spinner=False)
def load_models():
    # Import the supplied JT-VAE implementation bundled in this repository.
    from fast_jtnn import JTNNVAE, Vocab

    with open(MODEL_DIR / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    vocab = Vocab(
        [x.strip() for x in open(DATA_DIR / "vocab.txt", encoding="utf-8") if x.strip()]
    )

    model = JTNNVAE(
        vocab,
        cfg["hidden_size"],
        cfg["latent_size"],
        cfg["depthT"],
        cfg["depthG"],
    )
    model.load_state_dict(
        torch.load(MODEL_DIR / "best_model.pt", map_location="cpu")
    )
    model.eval()

    oracle = PropNet(d=56, h=256, out=3)
    oracle.load_state_dict(
        torch.load(MODEL_DIR / "property_oracle.pt", map_location="cpu")
    )
    oracle.eval()

    scaler = pd.read_csv(MODEL_DIR / "property_scaler.csv")
    mu = torch.tensor(scaler["mean"].values, dtype=torch.float32)
    sd = torch.tensor(scaler["std"].values, dtype=torch.float32)

    known_df = pd.read_csv(DATA_DIR / "valid_ligands.csv")
    known = set()
    for s in known_df["smiles"].dropna().astype(str):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            known.add(Chem.MolToSmiles(m, canonical=True))

    return model, oracle, mu, sd, known


def mol_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


@st.cache_data(show_spinner=False)
def known_fingerprints(known_tuple):
    return [mol_fp(s) for s in known_tuple]


def generate_candidates(
    model,
    oracle,
    mu,
    sd,
    known,
    target_ueff,
    target_tor,
    n_candidates,
    n_starts,
    steps,
    lr,
    seed,
    max_rings=1,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    target_tio = tio_from_ueff_tor(target_ueff, target_tor)
    target = torch.tensor(
        [0.0, target_ueff, target_tio], dtype=torch.float32
    )
    target_z = (target - mu) / sd

    # We optimize only Ueff and tio. Ucal is intentionally NOT a target.
    # It remains a model-predicted output for each generated ligand.
    starts = torch.randn(n_starts, LATENT_DIM)
    results = {}

    progress = st.progress(0, text="Optimizing latent vectors...")
    status = st.empty()

    for j in range(n_starts):
        z = starts[j : j + 1].clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)

        for _ in range(steps):
            opt.zero_grad()
            pred = oracle(z)
            # Property indices: Ucal=0, Ueff=1, tio=2
            loss_ueff = (pred[:, 1] - target_z[1]) ** 2
            loss_tio = (pred[:, 2] - target_z[2]) ** 2
            latent_reg = 1e-4 * (z * z).mean()
            loss = 0.5 * loss_ueff.mean() + 0.5 * loss_tio.mean() + latent_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_([z], 5.0)
            opt.step()

        with torch.no_grad():
            pz = oracle(z).cpu().numpy()[0]
            pred_props = pz * sd.numpy() + mu.numpy()
            zt, zm = torch.chunk(z.detach(), 2, dim=1)

        try:
            smiles = model.decode(zt, zm, False)
        except Exception:
            smiles = None

        if not smiles:
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        smiles = Chem.MolToSmiles(mol, canonical=True)
        if smiles in known:
            continue

        ring_count = rdMolDescriptors.CalcNumRings(mol)
        if ring_count > max_rings:
            continue

        ucal_pred, ueff_pred, tio_pred = map(float, pred_props)
        tor_pred = float(tor_from_ueff_tio(ueff_pred, tio_pred))
        if not np.isfinite(tor_pred) or tor_pred <= 0:
            continue

        # Distance is only in the requested Ueff + tio target space.
        dist = math.sqrt(
            0.5 * ((ueff_pred - target_ueff) / float(sd[1])) ** 2
            + 0.5 * ((tio_pred - target_tio) / float(sd[2])) ** 2
        )

        if smiles not in results or dist < results[smiles]["target_distance"]:
            results[smiles] = {
                "smiles": smiles,
                "Ucal_pred": ucal_pred,
                "Ueff_pred": ueff_pred,
                "tio_pred": tio_pred,
                "tau0_s_pred": 10.0 ** tio_pred,
                "Tor_pred_K": tor_pred,
                "target_distance": dist,
                "ring_count": int(ring_count),
                "MolWt": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "NumHeavyAtoms": mol.GetNumHeavyAtoms(),
            }

        if (j + 1) % max(1, n_starts // 20) == 0 or j == n_starts - 1:
            progress.progress(
                (j + 1) / n_starts,
                text=f"Optimizing latent vectors: {j + 1}/{n_starts}",
            )
            status.write(f"Unique valid novel candidates: {len(results)}")

    progress.empty()
    status.empty()

    df = pd.DataFrame(results.values())
    if len(df):
        df = (
            df.sort_values("target_distance", ascending=True)
            .head(n_candidates)
            .reset_index(drop=True)
        )
        df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df, target_tio


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("🧲 Dy-based SMM Ligand Generator")
st.markdown(
    "**Property-guided JT-VAE generation for pseudo-linear Dy-based molecular magnets**"
)

st.info(
    "The model generates individual ligand candidates. The target is specified as "
    "Ueff + Tₒᵣ; log₁₀(τ₀) is calculated from the supplied equation with τref = 100 s. "
    "Ucal is predicted but is not used as a generation target."
)

st.latex(
    r"T_{Or}=-\frac{U_{eff}}{\ln(\tau_0/\tau_{ref})},\qquad "
    r"\tau_{ref}=100\;s"
)

with st.sidebar:
    st.header("Generation target")
    target_ueff = st.number_input(
        "Target Ueff (K)", min_value=100.0, max_value=3000.0, value=2000.0, step=50.0
    )
    target_tor = st.number_input(
        "Target Tₒᵣ (K)", min_value=5.0, max_value=200.0, value=100.0, step=5.0
    )

    st.divider()
    st.subheader("Generation settings")
    n_candidates = st.slider("Number of candidates", 5, 100, 20, 5)
    n_starts = st.slider("Latent starts", 25, 500, 150, 25)
    steps = st.slider("Optimization steps", 10, 100, 50, 10)
    lr = st.slider("Latent learning rate", 0.01, 0.20, 0.08, 0.01)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42)

try:
    target_tio = tio_from_ueff_tor(target_ueff, target_tor)
    target_tau0 = 10.0 ** target_tio
except ValueError as e:
    st.error(str(e))
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Target Ueff", f"{target_ueff:.1f} K")
c2.metric("Target Tₒᵣ", f"{target_tor:.1f} K")
c3.metric("Calculated log₁₀(τ₀)", f"{target_tio:.4f}")

st.caption(f"Corresponding τ₀ = {target_tau0:.4e} s (τref = {TREF:.0f} s)")

# Explain the inverse relationship directly.
with st.expander("How is log(τ₀) calculated?"):
    st.write(
        "For a selected Ueff and Tₒᵣ, the equation can be rearranged as "
        "τ₀ = τref × exp(−Ueff/Tₒᵣ). Therefore:"
    )
    st.latex(
        r"\log_{10}(\tau_0)=\log_{10}(100)-\frac{U_{eff}}{T_{Or}\ln(10)}"
    )
    st.warning(
        "Tₒᵣ alone cannot uniquely determine both Ueff and τ₀. "
        "The app therefore uses Ueff + Tₒᵣ as the two user-defined targets and "
        "calculates the required log₁₀(τ₀)."
    )

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

load_clicked = st.button("Load trained model", type="secondary", use_container_width=True)
if load_clicked:
    with st.spinner("Loading JT-VAE and property oracle..."):
        try:
            load_models.clear()
            load_models()
            st.session_state.model_loaded = True
            st.success("Trained JT-VAE and property oracle loaded successfully.")
        except Exception as e:
            st.exception(e)
            st.stop()

if not st.session_state.model_loaded:
    st.write("Click **Load trained model** before generating candidates.")
    st.stop()

model, oracle, mu, sd, known = load_models()

st.success(f"Model loaded. Known JT-VAE-compatible ligands: {len(known)}")

if st.button("🚀 Generate ligands", type="primary", use_container_width=True):
    with st.spinner("Generating novel ligands with the trained JT-VAE..."):
        try:
            df, calc_tio = generate_candidates(
                model=model,
                oracle=oracle,
                mu=mu,
                sd=sd,
                known=known,
                target_ueff=float(target_ueff),
                target_tor=float(target_tor),
                n_candidates=int(n_candidates),
                n_starts=int(n_starts),
                steps=int(steps),
                lr=float(lr),
                seed=int(seed),
                max_rings=1,
            )
            st.session_state.generated_df = df
            st.session_state.generated_target_tio = calc_tio
        except Exception as e:
            st.exception(e)
            st.stop()

if "generated_df" in st.session_state:
    df = st.session_state.generated_df

    st.divider()
    st.subheader("Generated candidates")

    if len(df) == 0:
        st.warning(
            "No valid novel candidates were obtained. Increase latent starts/steps "
            "or use a target closer to the training-property range."
        )
    else:
        display_cols = [
            "rank",
            "smiles",
            "Ucal_pred",
            "Ueff_pred",
            "tio_pred",
            "tau0_s_pred",
            "Tor_pred_K",
            "target_distance",
            "ring_count",
        ]
        st.dataframe(
            df[display_cols],
            use_container_width=True,
            hide_index=True,
        )

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download generated candidates (CSV)",
            data=csv,
            file_name="generated_Dy_ligands.csv",
            mime="text/csv",
            use_container_width=True,
        )

        best = df.iloc[0]
        st.subheader("Best-ranked candidate")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Ueff", f"{best.Ueff_pred:.1f} K")
        b2.metric("Tₒᵣ", f"{best.Tor_pred_K:.1f} K")
        b3.metric("log₁₀(τ₀)", f"{best.tio_pred:.4f}")
        b4.metric("Ucal", f"{best.Ucal_pred:.1f} K")

        st.code(str(best.smiles), language="text")

st.divider()
st.caption(
    "The displayed Ueff, Ucal, log₁₀(τ₀), and Tₒᵣ values are latent-oracle predictions. "
    "They should be validated with the downstream computational workflow (e.g., CASSCF/RASSI-SO/SINGLE_ANISO) before experimental interpretation."
)
