
"""
prepare_app_data.py

OFFLINE PREPARATION SCRIPT
--------------------------
Creates the file used by the Streamlit app:

    data/pair_predictions.csv

The pair representation is copied from the validated ML3 pair-oracle
generation code:

    L1 Morgan FP        1024
    L2 Morgan FP        1024
    |L1-L2|             1024
    L1*L2               1024
    L1 descriptors      9
    L2 descriptors      9
    |descriptor diff|   9
    descriptor product  9
    CN features         5

Total = 4137 features.

NO padding or truncation is performed.
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

warnings.filterwarnings("ignore")

FP_BITS = 1024
FP_RADIUS = 2
EXPECTED_DESCRIPTOR_DIM = 9
EXPECTED_FEATURE_DIM = 4137
TARGETS = ["Ucal", "Ueff", "tio"]
TREF_SECONDS = 100.0


# ------------------------------------------------------------------
# SMILES
# ------------------------------------------------------------------
def canonical(smiles):
    if pd.isna(smiles):
        return None

    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def detect_smiles_column(df):
    candidates = [
        "smiles",
        "canonical_smiles",
        "SMILES",
        "ligand",
        "L1",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "No SMILES column found. Available columns: "
        + str(df.columns.tolist())
    )


# ------------------------------------------------------------------
# EXACT ORACLE FEATURE BUILDER
# ------------------------------------------------------------------
def mol_features(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        FP_RADIUS,
        nBits=FP_BITS,
    )

    arr = np.zeros((FP_BITS,), dtype=np.float32)

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

    if descriptors.shape[0] != EXPECTED_DESCRIPTOR_DIM:
        raise RuntimeError(
            f"Descriptor dimension mismatch: "
            f"expected {EXPECTED_DESCRIPTOR_DIM}, "
            f"got {descriptors.shape[0]}"
        )

    return arr, descriptors


def pair_vector(a_fp, a_d, b_fp, b_d, cn1, cn2):
    # EXACT ordering used by the validated oracle pipeline.
    if tuple(a_fp) > tuple(b_fp):
        a_fp, b_fp = b_fp, a_fp
        a_d, b_d = b_d, a_d
        cn1, cn2 = cn2, cn1

    parts = [
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

    x = np.concatenate(parts).astype(np.float32)

    if x.shape[0] != EXPECTED_FEATURE_DIM:
        raise RuntimeError(
            f"EXACT FEATURE BUILDER FAILED: "
            f"expected {EXPECTED_FEATURE_DIM}, got {x.shape[0]}"
        )

    return x


def self_test():
    a_fp, a_d = mol_features("CC")
    b_fp, b_d = mol_features("CCC")

    x = pair_vector(a_fp, a_d, b_fp, b_d, 2, 2)

    print("=" * 72)
    print("FEATURE BUILDER SELF-TEST")
    print("=" * 72)
    print(f"Expected feature dimension: {EXPECTED_FEATURE_DIM}")
    print(f"Actual feature dimension  : {x.shape[0]}")

    if x.shape[0] != EXPECTED_FEATURE_DIM:
        raise RuntimeError("Feature self-test failed.")

    print("SELF-TEST: PASSED")


# ------------------------------------------------------------------
# ORACLE
# ------------------------------------------------------------------
def load_oracle(oracle_dir):
    oracle_dir = Path(oracle_dir)

    models = {}

    for target in TARGETS:
        path = oracle_dir / f"final_{target}_extra_trees.joblib"

        if not path.exists():
            raise FileNotFoundError(f"Missing oracle model: {path}")

        model = joblib.load(path)

        if not hasattr(model, "n_features_in_"):
            raise RuntimeError(
                f"{target} model does not expose n_features_in_."
            )

        if int(model.n_features_in_) != EXPECTED_FEATURE_DIM:
            raise RuntimeError(
                f"{target} oracle expects {model.n_features_in_} features, "
                f"but exact builder creates {EXPECTED_FEATURE_DIM}."
            )

        models[target] = model

        print(
            f"{target:5s}: {path} "
            f"({model.n_features_in_} features)"
        )

    config_path = oracle_dir / "feature_config.joblib"

    if config_path.exists():
        config = joblib.load(config_path)

        if isinstance(config, dict):
            saved_dim = config.get("feature_dim", EXPECTED_FEATURE_DIM)
            saved_bits = config.get("fp_bits", FP_BITS)
            saved_radius = config.get("radius", FP_RADIUS)

            if saved_dim != EXPECTED_FEATURE_DIM:
                raise RuntimeError(
                    f"feature_config feature_dim={saved_dim}, "
                    f"expected {EXPECTED_FEATURE_DIM}"
                )

            if saved_bits != FP_BITS:
                raise RuntimeError(
                    f"feature_config fp_bits={saved_bits}, expected {FP_BITS}"
                )

            if saved_radius != FP_RADIUS:
                raise RuntimeError(
                    f"feature_config radius={saved_radius}, expected {FP_RADIUS}"
                )

    return models


# ------------------------------------------------------------------
# CANDIDATE LIBRARY
# ------------------------------------------------------------------
def add_candidate(store, smiles, source):
    cs = canonical(smiles)

    if cs is None:
        return

    if cs not in store:
        store[cs] = {
            "smiles": cs,
            "source": source,
        }
    else:
        old = store[cs]["source"]

        if source not in old.split("+"):
            store[cs]["source"] = old + "+" + source


def load_original_ligands(data_file):
    df = pd.read_csv(data_file)

    required = ["L1", "L2"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Original dataset is missing columns: {missing}"
        )

    store = {}

    for col in ["L1", "L2"]:
        for s in df[col].dropna().astype(str):
            add_candidate(store, s, "original")

    return store, df


def load_extra_ligands(store, filename, default_source):
    if filename is None:
        return 0

    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    smiles_col = detect_smiles_column(df)

    before = len(store)

    for s in df[smiles_col].dropna().astype(str):
        add_candidate(store, s, default_source)

    return len(store) - before


def build_candidate_library(data_file, generated_file, augmented_file):
    store, original_df = load_original_ligands(data_file)

    original_count = len(store)

    generated_added = load_extra_ligands(
        store,
        generated_file,
        "generated",
    )

    augmented_added = load_extra_ligands(
        store,
        augmented_file,
        "augmented",
    )

    candidates = pd.DataFrame(list(store.values()))

    candidates = candidates.sort_values(
        ["source", "smiles"]
    ).reset_index(drop=True)

    print()
    print("=" * 72)
    print("CANDIDATE LIGAND LIBRARY")
    print("=" * 72)
    print(f"Original unique ligands : {original_count}")
    print(f"Generated additions     : {generated_added}")
    print(f"Augmented additions     : {augmented_added}")
    print(f"Final unique ligands    : {len(candidates)}")

    return candidates, original_df


# ------------------------------------------------------------------
# PAIR SCREEN
# ------------------------------------------------------------------
def pair_source(source1, source2):
    a = "original" if "original" in source1.split("+") else "generated"
    b = "original" if "original" in source2.split("+") else "generated"

    if a == "original" and b == "original":
        return "original_x_original"

    if a == "generated" and b == "generated":
        return "generated_x_generated"

    return "original_x_generated"


def calculate_tor(ueff, tio):
    """
    tio = log10(tau0)

    tau0 = 10**tio
    Tor = -Ueff / ln(tau0 / 100)
    """
    tau0 = np.power(10.0, tio)

    denominator = np.log(tau0 / TREF_SECONDS)

    if np.isclose(denominator, 0.0):
        return np.nan, tau0

    tor = -ueff / denominator

    return tor, tau0


def build_predictions(candidates, models, cn1, cn2, batch_size):
    records = candidates.to_dict("records")

    cache = {}

    print()
    print("=" * 72)
    print("BUILDING LIGAND FEATURE CACHE")
    print("=" * 72)

    for i, row in enumerate(records, start=1):
        cache[row["smiles"]] = mol_features(row["smiles"])

        if i % 25 == 0 or i == len(records):
            print(f"Ligands: {i}/{len(records)}")

    n = len(records)
    total_pairs = n * (n - 1) // 2

    print()
    print("=" * 72)
    print("PAIR SCREEN")
    print("=" * 72)
    print(f"Unique ligands : {n}")
    print(f"Unordered pairs: {total_pairs:,}")

    rows = []

    pair_counter = 0

    for start in range(0, total_pairs, batch_size):
        X_rows = []
        batch_records = []

        # Generate exactly the pair range belonging to this batch.
        # We avoid materializing the entire X matrix.
        needed_start = start
        needed_end = min(start + batch_size, total_pairs)

        count = 0

        for i in range(n):
            row_start = i * (2 * n - i - 1) // 2
            row_end = (i + 1) * (2 * n - i - 2) // 2

            if row_end < needed_start:
                continue

            if row_start >= needed_end:
                break

            j0 = i + 1

            skip = max(0, needed_start - row_start)
            j0 += skip

            j1 = i + 1 + (needed_end - max(needed_start, row_start))
            j1 = min(j1, n)

            for j in range(j0, j1):
                l1 = records[i]
                l2 = records[j]

                a_fp, a_d = cache[l1["smiles"]]
                b_fp, b_d = cache[l2["smiles"]]

                X_rows.append(
                    pair_vector(
                        a_fp,
                        a_d,
                        b_fp,
                        b_d,
                        cn1,
                        cn2,
                    )
                )

                batch_records.append((l1, l2))
                count += 1

        if not X_rows:
            continue

        X = np.vstack(X_rows).astype(np.float32)

        if X.shape[1] != EXPECTED_FEATURE_DIM:
            raise RuntimeError(
                f"Generated X has {X.shape[1]} features; "
                f"expected {EXPECTED_FEATURE_DIM}."
            )

        predictions = {
            target: models[target].predict(X)
            for target in TARGETS
        }

        uncertainties = {}

        for target in TARGETS:
            model = models[target]

            # ExtraTrees standard deviation across trees.
            tree_preds = np.vstack(
                [tree.predict(X) for tree in model.estimators_]
            )

            uncertainties[target] = tree_preds.std(
                axis=0,
                ddof=1,
            )

        for k, (l1, l2) in enumerate(batch_records):
            ucal = float(predictions["Ucal"][k])
            ueff = float(predictions["Ueff"][k])
            tio = float(predictions["tio"][k])

            tor, tau0 = calculate_tor(ueff, tio)

            rows.append(
                {
                    "L1": l1["smiles"],
                    "L2": l2["smiles"],
                    "L1_source": l1["source"],
                    "L2_source": l2["source"],
                    "pair_source": pair_source(
                        l1["source"],
                        l2["source"],
                    ),
                    "CN1": cn1,
                    "CN2": cn2,
                    "Ucal_pred": ucal,
                    "Ueff_pred": ueff,
                    "tio_pred": tio,
                    "tau0_pred_s": tau0,
                    "Tor_pred_K": tor,
                    "Ucal_uncertainty": float(
                        uncertainties["Ucal"][k]
                    ),
                    "Ueff_uncertainty": float(
                        uncertainties["Ueff"][k]
                    ),
                    "tio_uncertainty": float(
                        uncertainties["tio"][k]
                    ),
                }
            )

        pair_counter += len(batch_records)

        print(
            f"Pairs predicted: {pair_counter:,}/{total_pairs:,}"
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data",
        required=True,
        help="ML3_Ucal_Ueff_tio_2.csv",
    )

    p.add_argument(
        "--generated",
        required=True,
        help="generated_candidates.csv",
    )

    p.add_argument(
        "--augmented",
        default=None,
        help="Optional augmented_generated_candidates.csv",
    )

    p.add_argument(
        "--oracle_dir",
        required=True,
        help="pair_oracle_cv",
    )

    p.add_argument(
        "--out",
        default="data",
        help="Output data directory.",
    )

    p.add_argument(
        "--cn1",
        type=float,
        default=2,
    )

    p.add_argument(
        "--cn2",
        type=float,
        default=2,
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=2000,
    )

    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    self_test()

    print()
    print("=" * 72)
    print("LOADING ORACLE")
    print("=" * 72)

    models = load_oracle(args.oracle_dir)

    candidates, original_df = build_candidate_library(
        args.data,
        args.generated,
        args.augmented,
    )

    candidate_file = out_dir / "candidate_ligand_library.csv"
    candidates.to_csv(candidate_file, index=False)

    print(f"Saved candidate library: {candidate_file}")

    results = build_predictions(
        candidates,
        models,
        args.cn1,
        args.cn2,
        args.batch_size,
    )

    results_file = out_dir / "pair_predictions.csv"
    results.to_csv(results_file, index=False)

    metadata = {
        "feature_dimension": EXPECTED_FEATURE_DIM,
        "fingerprint_bits": FP_BITS,
        "fingerprint_radius": FP_RADIUS,
        "descriptor_dimension": EXPECTED_DESCRIPTOR_DIM,
        "cn1": args.cn1,
        "cn2": args.cn2,
        "tau_reference_seconds": TREF_SECONDS,
        "number_of_ligands": int(len(candidates)),
        "number_of_pairs": int(len(results)),
        "oracle_models": {
            target: int(models[target].n_features_in_)
            for target in TARGETS
        },
        "feature_definition": (
            "a_fp,b_fp,abs(a_fp-b_fp),a_fp*b_fp,"
            "a_d,b_d,abs(a_d-b_d),a_d*b_d,"
            "cn1,cn2,cn1+cn2,abs(cn1-cn2),cn1*cn2"
        ),
        "source_note": (
            "Original ligands are collected from L1/L2 in the original "
            "dataset. Generated and augmented ligands are canonicalized "
            "and merged. Duplicate canonical SMILES are retained once."
        ),
    }

    with open(
        out_dir / "app_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 72)
    print("APP DATA PREPARATION COMPLETE")
    print("=" * 72)
    print(f"Candidate ligands : {len(candidates):,}")
    print(f"Ligand pairs      : {len(results):,}")
    print(f"Pair predictions   : {results_file}")
    print(f"Metadata           : {out_dir / 'app_metadata.json'}")


if __name__ == "__main__":
    main()
