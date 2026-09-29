import os
import re
import json
import math
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ML3 JT-VAE Ligand Pair Generator",
    page_icon="🧪",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

ORACLE_DIR = BASE_DIR / "oracle"
DATA_CANDIDATE_PATHS = [
    BASE_DIR / "data_inputs" / "ML3_Ucal_Ueff_tio_2.csv",
    BASE_DIR / "data" / "ML3_Ucal_Ueff_tio_2.csv",
    BASE_DIR / "ML3_Ucal_Ueff_tio_2.csv",
]

GENERATED_CANDIDATE_PATHS = [
    BASE_DIR / "data_inputs" / "generated_candidates.csv",
    BASE_DIR / "data" / "generated_candidates.csv",
    BASE_DIR / "generated_candidates.csv",
]

FP_BITS = 1024
FP_RADIUS = 2
DESCRIPTOR_DIM = 9
FEATURE_DIM = 4137

TARGETS = ("Ucal", "Ueff", "tio")
TAU_REF_SECONDS = 100.0
DEFAULT_TOP = 20


# ============================================================
# BASIC UTILITIES
# ============================================================

def canonicalize_smiles(smiles):
    if pd.isna(smiles):
        return None

    s = str(smiles).strip()
    if not s:
        return None

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def detect_smiles_column(df):
    candidates = [
        "smiles",
        "SMILES",
        "canonical_smiles",
        "Canonical_SMILES",
        "ligand",
        "L1",
        "L2",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "No SMILES column was found. "
        f"Available columns: {list(df.columns)}"
    )


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


# ============================================================
# T_OR / tau0 CONVERSION
#
# The app uses the same reference time shown in the previous UI:
# tau_ref = 100 s.
#
# log10(tau0) = log10(tau_ref) - Ueff/(T_or*ln(10))
#
# Therefore:
# T_or = Ueff / ((log10(tau_ref)-log10(tau0))*ln(10))
# ============================================================

def tor_from_ueff_tio(ueff, tio, tau_ref=TAU_REF_SECONDS):
    ueff = np.asarray(ueff, dtype=float)
    tio = np.asarray(tio, dtype=float)

    log_tau_ref = math.log10(tau_ref)
    denominator = (log_tau_ref - tio) * math.log(10.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        tor = ueff / denominator

    tor[~np.isfinite(tor)] = np.nan
    tor[tor <= 0] = np.nan
    return tor


def tio_from_ueff_tor(ueff, tor, tau_ref=TAU_REF_SECONDS):
    return (
        math.log10(tau_ref)
        - np.asarray(ueff, dtype=float)
        / (np.asarray(tor, dtype=float) * math.log(10.0))
    )


# ============================================================
# ORACLE MODEL RECONSTRUCTION
#
# GitHub contains .part01 ... .part06 files because the original
# ExtraTrees joblib files are larger than the GitHub upload limit.
# The complete joblib file is reconstructed only in the temporary
# runtime directory; the large reconstructed files are NOT stored
# in the Git repository.
# ============================================================

def _part_sort_key(path):
    m = re.search(r"\.part(\d+)$", path.name)
    return int(m.group(1)) if m else 10**9


def reconstruct_split_model(target):
    pattern = f"final_{target}_extra_trees.joblib.part*"
    parts = sorted(ORACLE_DIR.glob(pattern), key=_part_sort_key)

    if not parts:
        # Development/local fallback: allow a complete joblib if present.
        full = ORACLE_DIR / f"final_{target}_extra_trees.joblib"
        if full.exists():
            return full

        raise FileNotFoundError(
            f"No split oracle parts found for {target} in:\n{ORACLE_DIR}"
        )

    temp_dir = Path(tempfile.gettempdir()) / "ml3_jtvae_oracle"
    temp_dir.mkdir(parents=True, exist_ok=True)
    output = temp_dir / f"final_{target}_extra_trees.joblib"

    # Reconstruct every time only if necessary.
    newest_part_time = max(p.stat().st_mtime for p in parts)
    if output.exists() and output.stat().st_mtime >= newest_part_time:
        return output

    with open(output, "wb") as fout:
        for part in parts:
            with open(part, "rb") as fin:
                while True:
                    chunk = fin.read(1024 * 1024)
                    if not chunk:
                        break
                    fout.write(chunk)

    return output


@st.cache_resource(show_spinner=False)
def load_oracle_models():
    models = {}

    for target in TARGETS:
        model_path = reconstruct_split_model(target)
        model = joblib.load(model_path)

        if not hasattr(model, "n_features_in_"):
            raise RuntimeError(
                f"{target} oracle does not expose n_features_in_."
            )

        expected = int(model.n_features_in_)
        if expected != FEATURE_DIM:
            raise RuntimeError(
                f"{target} oracle expects {expected} features, "
                f"but the application feature builder creates {FEATURE_DIM}."
            )

        models[target] = model

    # Optional feature configuration check.
    config_path = ORACLE_DIR / "feature_config.joblib"
    if config_path.exists():
        config = joblib.load(config_path)
        if isinstance(config, dict):
            cfg_dim = config.get("feature_dim")
            if cfg_dim is not None and int(cfg_dim) != FEATURE_DIM:
                raise RuntimeError(
                    f"feature_config.joblib reports {cfg_dim} features, "
                    f"not {FEATURE_DIM}."
                )

    return models


# ============================================================
# EXACT 4,137-FEATURE ORACLE REPRESENTATION
#
# 1024  L1 Morgan FP
# 1024  L2 Morgan FP
# 1024  |FP1-FP2|
# 1024  FP1*FP2
#    9  L1 descriptors
#    9  L2 descriptors
#    9  |D1-D2|
#    9  D1*D2
#    5  CN features
# ---------------------
# 4137 total
# ============================================================

def mol_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        FP_RADIUS,
        nBits=FP_BITS,
    )

    arr = np.zeros(FP_BITS, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)

    descriptors = np.array(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            mol.GetNumHeavyAtoms(),
        ],
        dtype=np.float32,
    )

    if descriptors.shape[0] != DESCRIPTOR_DIM:
        raise RuntimeError(
            f"Descriptor dimension mismatch: "
            f"{descriptors.shape[0]} != {DESCRIPTOR_DIM}"
        )

    return arr, descriptors


def pair_vector(a_fp, a_d, b_fp, b_d, cn1, cn2):
    # Exact canonical ordering used by the supplied oracle-generation code.
    if tuple(a_fp) > tuple(b_fp):
        a_fp, b_fp = b_fp, a_fp
        a_d, b_d = b_d, a_d
        cn1, cn2 = cn2, cn1

    x = np.concatenate(
        [
            a_fp,
            b_fp,
            np.abs(a_fp - b_fp),
            a_fp * b_fp,
            a_d,
            b_d,
            np.abs(a_d - b_d),
            a_d * b_d,
            np.array(
                [
                    cn1,
                    cn2,
                    cn1 + cn2,
                    abs(cn1 - cn2),
                    cn1 * cn2,
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)

    if x.shape[0] != FEATURE_DIM:
        raise RuntimeError(
            f"Oracle feature construction failed: {x.shape[0]} != {FEATURE_DIM}"
        )

    return x


def build_feature_cache(smiles_list):
    cache = {}
    for smiles in smiles_list:
        cache[smiles] = mol_features(smiles)
    return cache


# ============================================================
# DATA LOADING
# ============================================================

def load_generated_candidates():
    path = first_existing(GENERATED_CANDIDATE_PATHS)

    if path is None:
        raise FileNotFoundError(
            "generated_candidates.csv was not found.\n\n"
            "Expected one of:\n"
            + "\n".join(str(p) for p in GENERATED_CANDIDATE_PATHS)
        )

    df = pd.read_csv(path)
    smiles_col = detect_smiles_column(df)

    df["canonical_smiles"] = df[smiles_col].apply(canonicalize_smiles)
    df = df.dropna(subset=["canonical_smiles"]).copy()
    df = df.drop_duplicates("canonical_smiles").reset_index(drop=True)

    return pd.DataFrame(
        {
            "smiles": df["canonical_smiles"],
            "source": "JT-VAE generated",
        }
    )


def load_original_dataset():
    path = first_existing(DATA_CANDIDATE_PATHS)

    if path is None:
        return None, None

    df = pd.read_csv(path)

    required = ["L1", "L2", "CN1", "CN2", "Ucal", "Ueff", "tio"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Original dataset is missing required columns: "
            + ", ".join(missing)
        )

    df = df.copy()
    df["L1_canonical"] = df["L1"].apply(canonicalize_smiles)
    df["L2_canonical"] = df["L2"].apply(canonicalize_smiles)

    df = df.dropna(
        subset=[
            "L1_canonical",
            "L2_canonical",
            "CN1",
            "CN2",
            "Ucal",
            "Ueff",
            "tio",
        ]
    ).copy()

    return df, path


def select_original_ligands_for_target(
    df,
    target_mode,
    target_value,
    cn1,
    cn2,
    original_fraction,
    original_tor_window,
):
    if df is None or df.empty:
        return pd.DataFrame(columns=["smiles", "source"])

    cn_df = df[
        (df["CN1"] == cn1)
        & (df["CN2"] == cn2)
    ].copy()

    if cn_df.empty:
        return pd.DataFrame(columns=["smiles", "source"])

    if target_mode == "Ucal":
        cutoff = target_value * original_fraction
        relevant = cn_df[cn_df["Ucal"] >= cutoff].copy()

    elif target_mode == "Ueff":
        cutoff = target_value * original_fraction
        relevant = cn_df[cn_df["Ueff"] >= cutoff].copy()

    else:  # T_or
        observed_tor = tor_from_ueff_tio(
            cn_df["Ueff"].to_numpy(),
            cn_df["tio"].to_numpy(),
        )
        cn_df = cn_df.copy()
        cn_df["T_or_observed"] = observed_tor

        low = target_value - original_tor_window
        high = target_value + original_tor_window

        relevant = cn_df[
            cn_df["T_or_observed"].between(low, high, inclusive="both")
        ].copy()

    records = []

    for _, row in relevant.iterrows():
        records.append(
            {
                "smiles": row["L1_canonical"],
                "source": "original",
            }
        )
        records.append(
            {
                "smiles": row["L2_canonical"],
                "source": "original",
            }
        )

    if not records:
        return pd.DataFrame(columns=["smiles", "source"])

    return (
        pd.DataFrame(records)
        .drop_duplicates("smiles")
        .reset_index(drop=True)
    )


# ============================================================
# PAIR GENERATION
# ============================================================

def combine_candidate_ligands(original, generated):
    records = {}

    for _, row in original.iterrows():
        records[row["smiles"]] = row["source"]

    for _, row in generated.iterrows():
        smiles = row["smiles"]
        if smiles in records:
            records[smiles] = "original + JT-VAE generated"
        else:
            records[smiles] = "JT-VAE generated"

    return pd.DataFrame(
        [
            {"smiles": smiles, "source": source}
            for smiles, source in records.items()
        ]
    )


def build_pairs(candidates, cn1, cn2, allow_self_pairs=False):
    rows = candidates.to_dict("records")
    pairs = []

    for i in range(len(rows)):
        start_j = i if allow_self_pairs else i + 1

        for j in range(start_j, len(rows)):
            r1 = rows[i]
            r2 = rows[j]

            if not allow_self_pairs and r1["smiles"] == r2["smiles"]:
                continue

            s1 = r1["source"]
            s2 = r2["source"]

            if "original" in s1.lower() and "original" in s2.lower():
                pair_source = "original × original"
            elif "generated" in s1.lower() and "generated" in s2.lower():
                pair_source = "generated × generated"
            else:
                pair_source = "original × generated"

            pairs.append(
                {
                    "L1": r1["smiles"],
                    "L2": r2["smiles"],
                    "L1_source": s1,
                    "L2_source": s2,
                    "pair_source": pair_source,
                    "CN1": cn1,
                    "CN2": cn2,
                }
            )

    return pd.DataFrame(pairs)


# ============================================================
# PREDICTION
# ============================================================

def screen_pairs(pairs, cache, models, batch_size=256):
    if pairs.empty:
        return pairs.copy()

    results = []

    for start in range(0, len(pairs), batch_size):
        chunk = pairs.iloc[start:start + batch_size]

        X = np.vstack(
            [
                pair_vector(
                    *cache[row.L1],
                    *cache[row.L2],
                    row.CN1,
                    row.CN2,
                )
                for row in chunk.itertuples(index=False)
            ]
        ).astype(np.float32)

        out = chunk[["L1", "L2", "L1_source", "L2_source", "pair_source", "CN1", "CN2"]].copy()

        for target in TARGETS:
            model = models[target]

            if int(model.n_features_in_) != X.shape[1]:
                raise RuntimeError(
                    f"{target} oracle expects {model.n_features_in_} "
                    f"features, but the app generated {X.shape[1]}."
                )

            out[f"{target}_pred"] = model.predict(X)

        results.append(out)

    return pd.concat(results, ignore_index=True)


# ============================================================
# TARGET RANKING
# ============================================================

def rank_results(df, target_mode, target_value):
    df = df.copy()

    df["T_or_pred"] = tor_from_ueff_tio(
        df["Ueff_pred"].to_numpy(),
        df["tio_pred"].to_numpy(),
    )

    if target_mode == "Ueff":
        df["target_error"] = df["Ueff_pred"] - target_value
        df["target_abs_error"] = df["target_error"].abs()
        df = df.sort_values(
            ["target_abs_error", "Ucal_pred"],
            ascending=[True, False],
        )

    elif target_mode == "Ucal":
        df["target_error"] = df["Ucal_pred"] - target_value
        df["target_abs_error"] = df["target_error"].abs()
        df = df.sort_values(
            ["target_abs_error", "Ueff_pred"],
            ascending=[True, False],
        )

    else:
        valid = np.isfinite(df["T_or_pred"])
        df = df[valid].copy()

        df["target_error"] = df["T_or_pred"] - target_value
        df["target_abs_error"] = df["target_error"].abs()
        df = df.sort_values(
            ["target_abs_error", "Ueff_pred"],
            ascending=[True, False],
        )

    df = df.reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))

    return df


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🧪 ML3 JT-VAE Ligand Pair Generator")

st.markdown(
    """
This application performs **pair-aware inverse screening**.

You specify **one magnetic target**. The application evaluates complete
two-ligand combinations and returns the ligand pair itself together with
the predicted complex-level magnetic properties.

It does **not** ask you to generate one ligand and manually combine it
with another ligand.
"""
)

with st.sidebar:
    st.header("🎯 Generation target")

    target_mode = st.selectbox(
        "Target property",
        ["Ueff", "Ucal", "T_or"],
        index=0,
    )

    if target_mode == "Ueff":
        target_value = st.number_input(
            "Target Ueff (K)",
            min_value=0.0,
            max_value=5000.0,
            value=2000.0,
            step=50.0,
        )
        target_label = "Ueff"

    elif target_mode == "Ucal":
        target_value = st.number_input(
            "Target Ucal (K)",
            min_value=0.0,
            max_value=5000.0,
            value=2000.0,
            step=50.0,
        )
        target_label = "Ucal"

    else:
        target_value = st.number_input(
            "Target T_or (K)",
            min_value=1.0,
            max_value=500.0,
            value=100.0,
            step=5.0,
        )
        target_label = "T_or"

    st.divider()

    st.header("⚛️ Coordination")
    cn1 = st.number_input(
        "CN1",
        min_value=1,
        max_value=6,
        value=2,
        step=1,
    )
    cn2 = st.number_input(
        "CN2",
        min_value=1,
        max_value=6,
        value=2,
        step=1,
    )

    st.divider()

    st.header("🔬 Candidate library")

    n_results = st.slider(
        "Number of ligand combinations",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
    )

    original_fraction = st.slider(
        "Original-complex property cutoff",
        min_value=0.10,
        max_value=1.00,
        value=0.50,
        step=0.05,
        help=(
            "For Ueff/Ucal, original complexes are retained when the "
            "corresponding observed property is at least this fraction "
            "of the requested target."
        ),
    )

    original_tor_window = st.number_input(
        "Original T_or selection window (K)",
        min_value=5.0,
        max_value=200.0,
        value=50.0,
        step=5.0,
    )

    allow_self_pairs = st.checkbox(
        "Allow L1 = L2",
        value=False,
    )

    run = st.button(
        "🚀 Generate ligand combinations",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# STATIC INFORMATION
# ============================================================

with st.expander("How the target is evaluated"):
    if target_mode == "T_or":
        st.latex(
            r"""
            T_{\mathrm{or}}
            =
            \frac{U_{\mathrm{eff}}}
            {\left[\log_{10}(\tau_{\mathrm{ref}})
            -\log_{10}(\tau_0)\right]\ln 10}
            """
        )
        st.write(
            f"The application uses τref = {TAU_REF_SECONDS:g} s and "
            "the oracle prediction of log10(τ0/s)."
        )
    else:
        st.write(
            f"The selected property ({target_label}) is used directly "
            "as the single ranking target."
        )

with st.expander("Oracle feature definition"):
    st.write(
        "Each complete ligand pair is represented by exactly 4,137 features:"
    )
    st.code(
        "1024  L1 Morgan fingerprint\n"
        "1024  L2 Morgan fingerprint\n"
        "1024  |FP1 - FP2|\n"
        "1024  FP1 × FP2\n"
        "   9  L1 RDKit descriptors\n"
        "   9  L2 RDKit descriptors\n"
        "   9  |D1 - D2|\n"
        "   9  D1 × D2\n"
        "   5  CN features\n"
        "-------------------------\n"
        "4137 total",
        language="text",
    )


# ============================================================
# RUN
# ============================================================

if run:
    try:
        with st.status("Loading ML3 models and candidate library...", expanded=True) as status:
            st.write("Loading split ExtraTrees oracle models...")
            models = load_oracle_models()

            st.write("Loading JT-VAE-generated ligand library...")
            generated = load_generated_candidates()

            st.write(f"Generated ligand library: {len(generated)} unique ligands")

            original_df, original_path = load_original_dataset()

            if original_df is not None:
                st.write(f"Original complex dataset: {len(original_df)} complexes")
                original = select_original_ligands_for_target(
                    original_df,
                    target_mode=target_mode,
                    target_value=float(target_value),
                    cn1=float(cn1),
                    cn2=float(cn2),
                    original_fraction=float(original_fraction),
                    original_tor_window=float(original_tor_window),
                )
                st.write(
                    f"Target-relevant original ligands: {len(original)}"
                )
            else:
                original = pd.DataFrame(columns=["smiles", "source"])
                st.warning(
                    "Original dataset was not found. "
                    "The app will screen the JT-VAE-generated ligand library only."
                )

            candidates = combine_candidate_ligands(original, generated)

            if len(candidates) < 2:
                raise RuntimeError(
                    "Fewer than two unique ligand candidates are available."
                )

            st.write(
                f"Final ligand pool: {len(candidates)} unique ligands"
            )

            pairs = build_pairs(
                candidates,
                cn1=float(cn1),
                cn2=float(cn2),
                allow_self_pairs=allow_self_pairs,
            )

            if pairs.empty:
                raise RuntimeError("No ligand pairs could be constructed.")

            # Safety guard for Streamlit deployment.
            MAX_PAIRS = 30000

            if len(pairs) > MAX_PAIRS:
                # Retain generated-containing pairs first, then truncate.
                generated_mask = pairs["pair_source"].str.contains(
                    "generated", case=False, na=False
                )

                preferred = pairs[generated_mask].copy()
                remaining = pairs[~generated_mask].copy()

                keep_preferred = min(len(preferred), MAX_PAIRS)
                preferred = preferred.head(keep_preferred)

                remaining_slots = MAX_PAIRS - len(preferred)
                if remaining_slots > 0:
                    remaining = remaining.head(remaining_slots)

                pairs = pd.concat(
                    [preferred, remaining],
                    ignore_index=True,
                )

                st.warning(
                    f"The candidate pool produced more than {MAX_PAIRS:,} "
                    "pairs. The application screened a deployment-safe "
                    "subset, prioritizing pairs containing JT-VAE-generated ligands."
                )

            st.write(f"Pairs to screen: {len(pairs):,}")

            st.write("Building exact 4,137-feature pair representations...")
            unique_smiles = pd.unique(
                pd.concat([pairs["L1"], pairs["L2"]], ignore_index=True)
            )

            cache = build_feature_cache(unique_smiles)
            st.write(f"Feature cache built for {len(cache)} ligands")

            st.write("Running the three independent pair oracles...")
            predictions = screen_pairs(
                pairs,
                cache,
                models,
                batch_size=128,
            )

            status.update(
                label="Screening complete",
                state="complete",
                expanded=False,
            )

        ranked = rank_results(
            predictions,
            target_mode=target_mode,
            target_value=float(target_value),
        )

        top = ranked.head(n_results).copy()

        st.success(
            f"Generated {len(top)} ranked ligand combinations for "
            f"target {target_label} = {target_value:g}"
        )

        # --------------------------------------------------------
        # Summary metrics
        # --------------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Target", f"{target_value:g} K")
        c2.metric("Ligand pairs screened", f"{len(predictions):,}")
        c3.metric("Ligands in pool", f"{len(candidates):,}")

        if target_mode == "T_or":
            best_value = top.iloc[0]["T_or_pred"]
            c4.metric("Closest T_or", f"{best_value:.2f} K")
        elif target_mode == "Ueff":
            best_value = top.iloc[0]["Ueff_pred"]
            c4.metric("Closest Ueff", f"{best_value:.1f} K")
        else:
            best_value = top.iloc[0]["Ucal_pred"]
            c4.metric("Closest Ucal", f"{best_value:.1f} K")

        st.subheader("🧬 Generated ligand combinations")

        display = top[
            [
                "Rank",
                "L1",
                "L2",
                "CN1",
                "CN2",
                "pair_source",
                "Ucal_pred",
                "Ueff_pred",
                "T_or_pred",
                "tio_pred",
                "target_abs_error",
            ]
        ].copy()

        display = display.rename(
            columns={
                "L1": "Ligand 1",
                "L2": "Ligand 2",
                "pair_source": "Pair source",
                "Ucal_pred": "Pred. Ucal (K)",
                "Ueff_pred": "Pred. Ueff (K)",
                "T_or_pred": "Pred. T_or (K)",
                "tio_pred": "Pred. log10(tau0/s)",
                "target_abs_error": f"|Target error| ({target_label})",
            }
        )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ligand 1": st.column_config.TextColumn(
                    "Ligand 1",
                    width="large",
                ),
                "Ligand 2": st.column_config.TextColumn(
                    "Ligand 2",
                    width="large",
                ),
                "Pred. Ucal (K)": st.column_config.NumberColumn(
                    format="%.1f",
                ),
                "Pred. Ueff (K)": st.column_config.NumberColumn(
                    format="%.1f",
                ),
                "Pred. T_or (K)": st.column_config.NumberColumn(
                    format="%.2f",
                ),
                "Pred. log10(tau0/s)": st.column_config.NumberColumn(
                    format="%.4f",
                ),
            },
        )

        # --------------------------------------------------------
        # Individual candidate cards
        # --------------------------------------------------------

        st.subheader("🔎 Candidate details")

        for _, row in top.head(min(10, len(top))).iterrows():
            title = (
                f"#{int(row['Rank'])}  "
                f"{row['pair_source']}  |  "
                f"{row['L1'][:24]}… + {row['L2'][:24]}…"
            )

            with st.expander(title):
                left, right = st.columns(2)

                with left:
                    st.markdown("**Ligand 1**")
                    st.code(row["L1"], language="text")
                    st.caption(f"Source: {row['L1_source']}")

                with right:
                    st.markdown("**Ligand 2**")
                    st.code(row["L2"], language="text")
                    st.caption(f"Source: {row['L2_source']}")

                prop_df = pd.DataFrame(
                    {
                        "Property": [
                            "Ucal",
                            "Ueff",
                            "T_or",
                            "log10(tau0/s)",
                            "Target absolute error",
                        ],
                        "Prediction": [
                            row["Ucal_pred"],
                            row["Ueff_pred"],
                            row["T_or_pred"],
                            row["tio_pred"],
                            row["target_abs_error"],
                        ],
                    }
                )

                st.dataframe(
                    prop_df,
                    hide_index=True,
                    use_container_width=True,
                )

        # --------------------------------------------------------
        # Download
        # --------------------------------------------------------

        download_df = ranked.head(n_results).copy()

        csv_bytes = download_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download ligand combinations",
            data=csv_bytes,
            file_name="ML3_ligand_combinations.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.info(
            "The listed rows are complete L1 + L2 ligand combinations. "
            "The property values are ExtraTrees oracle predictions and "
            "should be treated as a screening layer before CASSCF/ab initio validation."
        )

    except Exception as exc:
        st.error("The application could not complete the screening.")
        st.exception(exc)

else:
    st.info(
        "Set one target property in the sidebar and click "
        "**Generate ligand combinations**."
    )

    st.markdown(
        """
### Expected output

The result is a **complete ligand pair**, for example:

`Ligand A + Ligand B`

together with:

- predicted Ucal
- predicted Ueff
- predicted T_or
- predicted log10(τ0/s)
- coordination numbers CN1/CN2
- pair source
- target error

The user never has to manually combine two separately generated ligand lists.
"""
    )
