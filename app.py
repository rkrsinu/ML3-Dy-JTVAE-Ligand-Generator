
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "data" / "pair_predictions.csv"
META_FILE = APP_DIR / "data" / "app_metadata.json"

TREF_SECONDS = 100.0


st.set_page_config(
    page_title="Dy(III) Ligand-Pair Generator",
    page_icon="🧲",
    layout="wide",
)

st.title("🧲 Dy(III) Ligand-Pair Generator")
st.caption(
    "Target-directed screening of original and JT-VAE-generated ligand combinations "
    "using the trained pair-property oracle."
)


@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing {DATA_FILE}. Run prepare_app_data.py first."
        )

    df = pd.read_csv(DATA_FILE)

    required = [
        "L1", "L2", "Ucal_pred", "Ueff_pred", "tio_pred",
        "tau0_pred_s", "Tor_pred_K",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            "pair_predictions.csv is missing required columns: "
            + ", ".join(missing)
        )

    return df


@st.cache_data(show_spinner=False)
def load_metadata():
    if META_FILE.exists():
        with open(META_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def format_number(x, digits=2):
    if pd.isna(x):
        return "—"
    return f"{float(x):.{digits}f}"


try:
    df = load_data()
    meta = load_metadata()
except Exception as exc:
    st.error(str(exc))
    st.stop()


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
st.sidebar.header("Generation target")

target_type = st.sidebar.radio(
    "Target property",
    ["Ueff", "Tor"],
    index=0,
)

if target_type == "Ueff":
    target = st.sidebar.number_input(
        "Target Ueff (K)",
        min_value=0.0,
        value=2000.0,
        step=50.0,
    )
else:
    target = st.sidebar.number_input(
        "Target Tor (K)",
        min_value=0.1,
        value=100.0,
        step=5.0,
    )

st.sidebar.divider()

n_results = st.sidebar.slider(
    "Number of ligand combinations",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
)

st.sidebar.divider()

st.sidebar.header("Search options")

source_options = ["All", "Original–Original", "Original–Generated", "Generated–Generated"]
source_choice = st.sidebar.selectbox(
    "Ligand-pair source",
    source_options,
    index=0,
)

max_error = st.sidebar.number_input(
    "Maximum target error (optional)",
    min_value=0.0,
    value=0.0,
    step=1.0,
    help="Set 0 to show the closest combinations without an error cutoff.",
)

# ------------------------------------------------------------------
# FILTER SOURCE
# ------------------------------------------------------------------
work = df.copy()

if source_choice != "All":
    mapping = {
        "Original–Original": "original_x_original",
        "Original–Generated": "original_x_generated",
        "Generated–Generated": "generated_x_generated",
    }
    if "pair_source" in work.columns:
        work = work[work["pair_source"] == mapping[source_choice]].copy()

if work.empty:
    st.warning("No ligand pairs are available for the selected source.")
    st.stop()


# ------------------------------------------------------------------
# TARGET RANKING
# ------------------------------------------------------------------
if target_type == "Ueff":
    work["target_error"] = (work["Ueff_pred"] - target).abs()
    work = work.sort_values(
        ["target_error", "Ueff_uncertainty"]
        if "Ueff_uncertainty" in work.columns
        else ["target_error"],
        ascending=True,
    ).reset_index(drop=True)

    target_label = f"Target Ueff = {target:.1f} K"
    error_label = "ΔUeff (K)"

else:
    work["target_error"] = (work["Tor_pred_K"] - target).abs()
    work = work.sort_values(
        ["target_error", "Tor_uncertainty_K"]
        if "Tor_uncertainty_K" in work.columns
        else ["target_error"],
        ascending=True,
    ).reset_index(drop=True)

    target_label = f"Target Tor = {target:.1f} K"
    error_label = "ΔTor (K)"

if max_error > 0:
    work = work[work["target_error"] <= max_error].copy()

results = work.head(n_results).copy()

# ------------------------------------------------------------------
# HEADER METRICS
# ------------------------------------------------------------------
st.subheader(target_label)

c1, c2, c3 = st.columns(3)

if target_type == "Ueff":
    best = results.iloc[0]
    c1.metric("Target Ueff", f"{target:.1f} K")
    c2.metric("Best predicted Ueff", f"{best['Ueff_pred']:.1f} K")
    c3.metric("Best |ΔUeff|", f"{best['target_error']:.1f} K")
else:
    best = results.iloc[0]
    c1.metric("Target Tor", f"{target:.1f} K")
    c2.metric("Best predicted Tor", f"{best['Tor_pred_K']:.1f} K")
    c3.metric("Best |ΔTor|", f"{best['target_error']:.1f} K")

st.caption(
    f"Showing {len(results)} ligand combinations from {len(work):,} available pairs. "
    f"τref = {TREF_SECONDS:g} s."
)

if results.empty:
    st.warning("No combinations satisfy the selected error cutoff.")
    st.stop()


# ------------------------------------------------------------------
# RESULT TABLE
# ------------------------------------------------------------------
st.subheader("Ligand combinations")

display = pd.DataFrame({
    "Rank": np.arange(1, len(results) + 1),
    "Ligand 1": results["L1"].values,
    "Ligand 2": results["L2"].values,
    "Ueff (K)": results["Ueff_pred"].round(2).values,
    "log10(τ₀)": results["tio_pred"].round(4).values,
    "τ₀ (s)": results["tau0_pred_s"].map(lambda x: f"{x:.3e}").values,
    "Tor (K)": results["Tor_pred_K"].round(2).values,
    error_label: results["target_error"].round(2).values,
})

if "pair_source" in results.columns:
    display["Source"] = results["pair_source"].values

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ligand 1": st.column_config.TextColumn(
            "Ligand 1", width="large"
        ),
        "Ligand 2": st.column_config.TextColumn(
            "Ligand 2", width="large"
        ),
    },
)

# ------------------------------------------------------------------
# DETAILED CANDIDATES
# ------------------------------------------------------------------
st.subheader("Candidate details")

for idx, row in results.iterrows():
    rank = idx + 1

    with st.expander(
        f"Candidate {rank}: "
        f"{format_number(row['Ueff_pred'], 1)} K Ueff | "
        f"{format_number(row['Tor_pred_K'], 1)} K Tor"
    ):
        left, right = st.columns(2)

        with left:
            st.markdown("**Ligand 1**")
            st.code(str(row["L1"]), language="text")

        with right:
            st.markdown("**Ligand 2**")
            st.code(str(row["L2"]), language="text")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ueff", f"{row['Ueff_pred']:.2f} K")
        m2.metric("log10(τ₀)", f"{row['tio_pred']:.4f}")
        m3.metric("τ₀", f"{row['tau0_pred_s']:.3e} s")
        m4.metric("Tor", f"{row['Tor_pred_K']:.2f} K")

        if target_type == "Ueff":
            st.write(
                f"**Target Ueff:** {target:.2f} K  |  "
                f"**|ΔUeff|:** {row['target_error']:.2f} K"
            )
        else:
            st.write(
                f"**Target Tor:** {target:.2f} K  |  "
                f"**|ΔTor|:** {row['target_error']:.2f} K"
            )

        if "pair_source" in row:
            st.write(f"**Pair source:** `{row['pair_source']}`")

        if "Ueff_uncertainty" in row:
            st.write(
                f"ExtraTrees ensemble uncertainty — "
                f"Ueff: {row['Ueff_uncertainty']:.2f}, "
                f"tio: {row['tio_uncertainty']:.4f}"
            )


# ------------------------------------------------------------------
# DOWNLOAD
# ------------------------------------------------------------------
st.subheader("Download")

download_df = results.copy()
download_df.insert(0, "rank", np.arange(1, len(download_df) + 1))

csv_bytes = download_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download ligand combinations (CSV)",
    data=csv_bytes,
    file_name=f"dy_ligand_combinations_{target_type.lower()}_{target:g}.csv",
    mime="text/csv",
)

with st.expander("How Tor is calculated"):
    st.latex(
        r"T_{\mathrm{Or}}="
        r"-\frac{U_{\mathrm{eff}}}"
        r"{\ln(\tau_0/\tau_{\mathrm{ref}})}"
    )
    st.write(
        "The pair oracle predicts log10(τ₀). Therefore the app first calculates "
        "τ₀ = 10^(log10(τ₀)), then uses τref = 100 s."
    )
    st.latex(
        r"\tau_0=10^{\log_{10}(\tau_0)},\qquad "
        r"\tau_{\mathrm{ref}}=100\ {\rm s}"
    )

with st.expander("About the model"):
    st.write(
        "This app does not load the JT-VAE model in the cloud. "
        "JT-VAE generation and pair-oracle prediction are performed offline. "
        "The deployed app only ranks the precomputed ligand-pair predictions."
    )

    if meta:
        st.json(meta)
