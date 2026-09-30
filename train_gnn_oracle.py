# ============================================================
# 05_train_gnn_oracle.py
# ============================================================
# ML3 Gen-AI project
#
# Purpose:
#   Train a molecular GNN oracle for:
#       Ucal
#       Ueff
#       tio
#
# Input:
#   ML3_Ucal_Ueff_tio_2.csv
#
# Three evaluation modes:
#   1. random
#   2. pair-disjoint
#   3. ligand-disjoint
#
# Architecture:
#
#        L1 SMILES -> GINE encoder -> z1 ----\
#                                              \
#                                               Fusion -> Ucal
#                                              /         Ueff
#        L2 SMILES -> GINE encoder -> z2 ----/          tio
#
#                    + CN1 + CN2
#
# Requirements:
#   pip install torch pandas numpy scikit-learn matplotlib joblib
#   pip install torch-geometric
#
# Run:
#   python src/05_train_gnn_oracle.py
#
# Optional:
#   python src/05_train_gnn_oracle.py --data ML3_Ucal_Ueff_tio_2.csv
#   python src/05_train_gnn_oracle.py --split random
#   python src/05_train_gnn_oracle.py --split pair
#   python src/05_train_gnn_oracle.py --split ligand
#
# Default:
#   runs ALL THREE splits.
#
# ============================================================

import argparse
import json
import os
import random
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from rdkit import Chem

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool


# ============================================================
# Configuration
# ============================================================

TARGETS = ["Ucal", "Ueff", "tio"]

SEED = 42

HIDDEN = 128
EMBED = 128
GNN_LAYERS = 4
DROPOUT = 0.15

BATCH_SIZE = 64
EPOCHS = 400
PATIENCE = 60
LR = 1e-3
WEIGHT_DECAY = 2e-4

TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10

ATOM_FEATURE_DIM = 12
BOND_FEATURE_DIM = 6

DEFAULT_DATA = "ML3_Ucal_Ueff_tio_2.csv"
DEFAULT_OUTDIR = "gnn_oracle"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# SMILES canonicalization
# ============================================================

def canonicalize_smiles(smiles):

    if pd.isna(smiles):
        return None

    smiles = str(smiles).strip()

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return Chem.MolToSmiles(
        mol,
        canonical=True
    )


# ============================================================
# Atom features
# ============================================================

def atom_features(atom):

    atomic_number = atom.GetAtomicNum()

    # Common elements in organic ligand datasets.
    # Unknown elements are represented by the final bucket.
    atomic_vocab = [
        1, 5, 6, 7, 8, 9,
        15, 16, 17, 35, 53
    ]

    element_onehot = [
        float(atomic_number == z)
        for z in atomic_vocab
    ]

    element_onehot.append(
        float(atomic_number not in atomic_vocab)
    )

    features = (
        element_onehot
        + [
            float(atom.GetDegree()) / 6.0,
            float(atom.GetFormalCharge()) / 3.0,
            float(atom.GetTotalNumHs()) / 4.0,
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            float(atom.GetHybridization().real)
            if hasattr(atom.GetHybridization(), "real")
            else 0.0,
        ]
    )

    return features


# ============================================================
# Bond features
# ============================================================

def bond_features(bond):

    bt = bond.GetBondType()

    return [
        float(bt == Chem.rdchem.BondType.SINGLE),
        float(bt == Chem.rdchem.BondType.DOUBLE),
        float(bt == Chem.rdchem.BondType.TRIPLE),
        float(bt == Chem.rdchem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


# ============================================================
# SMILES -> PyG graph
# ============================================================

def smiles_to_graph(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            f"Invalid SMILES: {smiles}"
        )

    # -----------------------------
    # Nodes
    # -----------------------------

    x = torch.tensor(
        [
            atom_features(atom)
            for atom in mol.GetAtoms()
        ],
        dtype=torch.float32
    )

    # -----------------------------
    # Edges
    # -----------------------------

    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bf = bond_features(bond)

        edge_index.append([i, j])
        edge_index.append([j, i])

        edge_attr.append(bf)
        edge_attr.append(bf)

    if len(edge_index) == 0:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )

        edge_attr = torch.empty(
            (0, BOND_FEATURE_DIM),
            dtype=torch.float32
        )

    else:

        edge_index = torch.tensor(
            edge_index,
            dtype=torch.long
        ).t().contiguous()

        edge_attr = torch.tensor(
            edge_attr,
            dtype=torch.float32
        )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )


# ============================================================
# Pair dataset
# ============================================================

class PairDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataframe,
        graph_cache,
        target_mean,
        target_std,
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.graph_cache = graph_cache

        self.target_mean = torch.tensor(
            target_mean,
            dtype=torch.float32
        )

        self.target_std = torch.tensor(
            target_std,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        g1 = self.graph_cache[row["L1"]]
        g2 = self.graph_cache[row["L2"]]

        y = torch.tensor(
            [
                row["Ucal"],
                row["Ueff"],
                row["tio"],
            ],
            dtype=torch.float32
        )

        y = (
            y - self.target_mean
        ) / self.target_std

        return (
            g1.clone(),
            g2.clone(),
            torch.tensor(
                [
                    float(row["CN1"]),
                    float(row["CN2"]),
                ],
                dtype=torch.float32
            ),
            y,
            idx,
        )


# ============================================================
# Custom collate
# ============================================================

def collate_pairs(batch):

    g1 = Batch.from_data_list(
        [x[0] for x in batch]
    )

    g2 = Batch.from_data_list(
        [x[1] for x in batch]
    )

    cn = torch.stack(
        [x[2] for x in batch]
    )

    y = torch.stack(
        [x[3] for x in batch]
    )

    indices = torch.tensor(
        [x[4] for x in batch],
        dtype=torch.long
    )

    return g1, g2, cn, y, indices


# ============================================================
# GINE encoder
# ============================================================

class GINEEncoder(nn.Module):

    def __init__(
        self,
        hidden=HIDDEN,
        embed=EMBED,
        layers=GNN_LAYERS,
        dropout=DROPOUT,
    ):

        super().__init__()

        self.dropout = dropout

        self.node_encoder = nn.Linear(
            ATOM_FEATURE_DIM,
            hidden
        )

        self.convs = nn.ModuleList()

        self.norms = nn.ModuleList()

        for _ in range(layers):

            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )

            self.convs.append(
                GINEConv(
                    mlp,
                    edge_dim=BOND_FEATURE_DIM
                )
            )

            self.norms.append(
                nn.BatchNorm1d(hidden)
            )

        self.projection = nn.Sequential(
            nn.Linear(hidden, embed),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed, embed),
        )

    def forward(self, data):

        x = self.node_encoder(
            data.x
        )

        for conv, norm in zip(
            self.convs,
            self.norms
        ):

            residual = x

            x = conv(
                x,
                data.edge_index,
                data.edge_attr
            )

            x = norm(x)

            x = F.relu(x)

            x = F.dropout(
                x,
                p=self.dropout,
                training=self.training
            )

            x = x + residual

        x = global_mean_pool(
            x,
            data.batch
        )

        x = self.projection(x)

        return x


# ============================================================
# Full oracle
# ============================================================

class PairGNNOracle(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = GINEEncoder()

        self.fusion = nn.Sequential(
            nn.Linear(
                EMBED * 2 + 2,
                256
            ),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(
                256,
                128
            ),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(
                128,
                3
            )
        )

    def forward(
        self,
        g1,
        g2,
        cn
    ):

        z1 = self.encoder(g1)

        z2 = self.encoder(g2)

        z = torch.cat(
            [
                z1,
                z2,
                cn
            ],
            dim=1
        )

        return self.fusion(z)


# ============================================================
# Data splitting
# ============================================================

def random_split(df):

    indices = np.arange(
        len(df)
    )

    rng = np.random.default_rng(
        SEED
    )

    rng.shuffle(indices)

    n = len(indices)

    n_test = int(
        n * TEST_FRACTION
    )

    n_val = int(
        n * VAL_FRACTION
    )

    test_idx = indices[:n_test]

    val_idx = indices[
        n_test:n_test + n_val
    ]

    train_idx = indices[
        n_test + n_val:
    ]

    return (
        df.iloc[train_idx].copy(),
        df.iloc[val_idx].copy(),
        df.iloc[test_idx].copy()
    )


def pair_disjoint_split(df):

    # Unordered pair:
    # A+B and B+A belong to the same group.
    groups = df.apply(
        lambda r: "||".join(
            sorted(
                [r["L1"], r["L2"]]
            )
        ),
        axis=1
    )

    unique_groups = np.array(
        groups.unique()
    )

    rng = np.random.default_rng(
        SEED
    )

    rng.shuffle(unique_groups)

    n = len(unique_groups)

    n_test = max(
        1,
        int(n * TEST_FRACTION)
    )

    n_val = max(
        1,
        int(n * VAL_FRACTION)
    )

    test_groups = set(
        unique_groups[:n_test]
    )

    val_groups = set(
        unique_groups[
            n_test:n_test + n_val
        ]
    )

    train_groups = set(
        unique_groups[
            n_test + n_val:
        ]
    )

    train = df[
        groups.isin(train_groups)
    ].copy()

    val = df[
        groups.isin(val_groups)
    ].copy()

    test = df[
        groups.isin(test_groups)
    ].copy()

    return train, val, test


def ligand_disjoint_split(df):

    # Strict ligand-disjoint split.
    #
    # Test ligands are completely absent from training.
    # To preserve this condition, cross-partition pairs are
    # discarded from this particular evaluation.
    #
    # This is intentionally a harder generalization test.

    ligands = np.array(
        sorted(
            set(df["L1"]) |
            set(df["L2"])
        )
    )

    rng = np.random.default_rng(
        SEED
    )

    rng.shuffle(ligands)

    n_test = max(
        1,
        int(len(ligands) * TEST_FRACTION)
    )

    n_val = max(
        1,
        int(len(ligands) * VAL_FRACTION)
    )

    test_ligands = set(
        ligands[:n_test]
    )

    val_ligands = set(
        ligands[
            n_test:n_test + n_val
        ]
    )

    train_ligands = set(
        ligands[
            n_test + n_val:
        ]
    )

    def both_in(row, ligand_set):

        return (
            row["L1"] in ligand_set
            and
            row["L2"] in ligand_set
        )

    train = df[
        df.apply(
            lambda r: both_in(
                r,
                train_ligands
            ),
            axis=1
        )
    ].copy()

    val = df[
        df.apply(
            lambda r: both_in(
                r,
                val_ligands
            ),
            axis=1
        )
    ].copy()

    test = df[
        df.apply(
            lambda r: both_in(
                r,
                test_ligands
            ),
            axis=1
        )
    ].copy()

    return train, val, test


# ============================================================
# Target statistics
# ============================================================

def target_statistics(train_df):

    mean = (
        train_df[TARGETS]
        .mean()
        .values
        .astype(np.float32)
    )

    std = (
        train_df[TARGETS]
        .std()
        .values
        .astype(np.float32)
    )

    std[std < 1e-8] = 1.0

    return mean, std


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    output = {}

    for i, target in enumerate(
        TARGETS
    ):

        yt = y_true[:, i]

        yp = y_pred[:, i]

        output[target] = {
            "R2": float(
                r2_score(yt, yp)
            ),
            "MAE": float(
                mean_absolute_error(
                    yt,
                    yp
                )
            ),
            "RMSE": float(
                np.sqrt(
                    mean_squared_error(
                        yt,
                        yp
                    )
                )
            )
        }

    output["mean_R2"] = float(
        np.mean(
            [
                output[t]["R2"]
                for t in TARGETS
            ]
        )
    )

    output["mean_MAE"] = float(
        np.mean(
            [
                output[t]["MAE"]
                for t in TARGETS
            ]
        )
    )

    output["mean_RMSE"] = float(
        np.mean(
            [
                output[t]["RMSE"]
                for t in TARGETS
            ]
        )
    )

    return output


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    target_mean,
    target_std,
    device
):

    model.eval()

    all_true = []
    all_pred = []
    all_idx = []

    for (
        g1,
        g2,
        cn,
        y,
        indices
    ) in loader:

        g1 = g1.to(device)
        g2 = g2.to(device)
        cn = cn.to(device)
        y = y.to(device)

        pred = model(
            g1,
            g2,
            cn
        )

        all_true.append(
            y.cpu().numpy()
        )

        all_pred.append(
            pred.cpu().numpy()
        )

        all_idx.append(
            indices.cpu().numpy()
        )

    y_true = np.vstack(
        all_true
    )

    y_pred = np.vstack(
        all_pred
    )

    indices = np.concatenate(
        all_idx
    )

    # Undo standardization.
    y_true = (
        y_true
        * target_std
        + target_mean
    )

    y_pred = (
        y_pred
        * target_std
        + target_mean
    )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    return (
        metrics,
        y_true,
        y_pred,
        indices
    )


# ============================================================
# Training
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    target_mean,
    target_std,
    outdir,
    device
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=15,
        min_lr=1e-6
    )

    criterion = nn.SmoothL1Loss()

    best_state = None
    best_val = np.inf
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        train_losses = []

        for (
            g1,
            g2,
            cn,
            y,
            _
        ) in train_loader:

            g1 = g1.to(device)
            g2 = g2.to(device)
            cn = cn.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            pred = model(
                g1,
                g2,
                cn
            )

            loss = criterion(
                pred,
                y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            train_losses.append(
                loss.item()
            )

        train_loss = float(
            np.mean(train_losses)
        )

        # --------------------------------------------
        # Validation loss
        # --------------------------------------------

        model.eval()

        val_losses = []

        with torch.no_grad():

            for (
                g1,
                g2,
                cn,
                y,
                _
            ) in val_loader:

                g1 = g1.to(device)
                g2 = g2.to(device)
                cn = cn.to(device)
                y = y.to(device)

                pred = model(
                    g1,
                    g2,
                    cn
                )

                loss = criterion(
                    pred,
                    y
                )

                val_losses.append(
                    loss.item()
                )

        val_loss = float(
            np.mean(val_losses)
        )

        scheduler.step(
            val_loss
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["lr"].append(
            current_lr
        )

        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            print(
                f"Epoch {epoch:4d} | "
                f"Train {train_loss:.5f} | "
                f"Val {val_loss:.5f} | "
                f"LR {current_lr:.2e}"
            )

        # --------------------------------------------
        # Early stopping
        # --------------------------------------------

        if val_loss < best_val:

            best_val = val_loss

            best_state = deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                f"\nEarly stopping at epoch {epoch}"
            )

            break

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return model, history


# ============================================================
# Plot training
# ============================================================

def plot_training(
    history,
    path
):

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        history["train_loss"],
        label="Train"
    )

    plt.plot(
        history["val_loss"],
        label="Validation"
    )

    plt.xlabel(
        "Epoch",
        fontsize=13
    )

    plt.ylabel(
        "SmoothL1 loss",
        fontsize=13
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()


# ============================================================
# Plot predicted vs true
# ============================================================

def plot_predictions(
    y_true,
    y_pred,
    metrics,
    outdir
):

    for i, target in enumerate(
        TARGETS
    ):

        plt.figure(
            figsize=(6, 6)
        )

        plt.scatter(
            y_true[:, i],
            y_pred[:, i],
            s=22,
            alpha=0.65
        )

        lo = min(
            y_true[:, i].min(),
            y_pred[:, i].min()
        )

        hi = max(
            y_true[:, i].max(),
            y_pred[:, i].max()
        )

        plt.plot(
            [lo, hi],
            [lo, hi],
            linestyle="--"
        )

        plt.xlabel(
            f"True {target}",
            fontsize=13
        )

        plt.ylabel(
            f"Predicted {target}",
            fontsize=13
        )

        plt.title(
            f"{target}: "
            f"R²={metrics[target]['R2']:.3f}, "
            f"MAE={metrics[target]['MAE']:.3f}",
            fontsize=12
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                outdir,
                f"{target}_predicted_vs_true.png"
            ),
            dpi=300
        )

        plt.close()


# ============================================================
# Save predictions
# ============================================================

def save_predictions(
    dataframe,
    y_true,
    y_pred,
    path
):

    output = dataframe.reset_index(
        drop=True
    ).copy()

    for i, target in enumerate(
        TARGETS
    ):

        output[
            f"{target}_true"
        ] = y_true[:, i]

        output[
            f"{target}_pred"
        ] = y_pred[:, i]

        output[
            f"{target}_error"
        ] = (
            y_pred[:, i]
            - y_true[:, i]
        )

    output.to_csv(
        path,
        index=False
    )


# ============================================================
# Train one split
# ============================================================

def run_split(
    df,
    split_name,
    graph_cache,
    root_outdir
):

    print()
    print("=" * 72)
    print(
        f"RUNNING SPLIT: {split_name.upper()}"
    )
    print("=" * 72)

    if split_name == "random":

        train_df, val_df, test_df = (
            random_split(df)
        )

    elif split_name == "pair":

        train_df, val_df, test_df = (
            pair_disjoint_split(df)
        )

    elif split_name == "ligand":

        train_df, val_df, test_df = (
            ligand_disjoint_split(df)
        )

    else:

        raise ValueError(
            f"Unknown split: {split_name}"
        )

    print(
        f"Train: {len(train_df):,}"
    )

    print(
        f"Val  : {len(val_df):,}"
    )

    print(
        f"Test : {len(test_df):,}"
    )

    if (
        len(train_df) < 10
        or len(val_df) < 2
        or len(test_df) < 2
    ):

        raise RuntimeError(
            f"{split_name} split produced too few rows."
        )

    # --------------------------------------------
    # Target scaling based ONLY on training data
    # --------------------------------------------

    target_mean, target_std = (
        target_statistics(
            train_df
        )
    )

    # --------------------------------------------
    # Datasets
    # --------------------------------------------

    train_ds = PairDataset(
        train_df,
        graph_cache,
        target_mean,
        target_std
    )

    val_ds = PairDataset(
        val_df,
        graph_cache,
        target_mean,
        target_std
    )

    test_ds = PairDataset(
        test_df,
        graph_cache,
        target_mean,
        target_std
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_pairs
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_pairs
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_pairs
    )

    # --------------------------------------------
    # Model
    # --------------------------------------------

    model = PairGNNOracle().to(
        DEVICE
    )

    n_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{n_parameters:,}"
    )

    # --------------------------------------------
    # Train
    # --------------------------------------------

    split_outdir = os.path.join(
        root_outdir,
        split_name
    )

    os.makedirs(
        split_outdir,
        exist_ok=True
    )

    model, history = train_model(
        model,
        train_loader,
        val_loader,
        target_mean,
        target_std,
        split_outdir,
        DEVICE
    )

    # --------------------------------------------
    # Evaluate
    # --------------------------------------------

    metrics, y_true, y_pred, _ = evaluate(
        model,
        test_loader,
        target_mean,
        target_std,
        DEVICE
    )

    print()
    print(
        f"RESULTS — {split_name.upper()}"
    )

    for target in TARGETS:

        print(
            f"{target:5s} | "
            f"R2={metrics[target]['R2']:.4f} | "
            f"MAE={metrics[target]['MAE']:.4f} | "
            f"RMSE={metrics[target]['RMSE']:.4f}"
        )

    print(
        f"Mean R2  = {metrics['mean_R2']:.4f}"
    )

    print(
        f"Mean MAE = {metrics['mean_MAE']:.4f}"
    )

    # --------------------------------------------
    # Save
    # --------------------------------------------

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "target_mean": target_mean,
            "target_std": target_std,
            "config": {
                "hidden": HIDDEN,
                "embed": EMBED,
                "gnn_layers": GNN_LAYERS,
                "dropout": DROPOUT,
                "atom_feature_dim": ATOM_FEATURE_DIM,
                "bond_feature_dim": BOND_FEATURE_DIM,
            },
            "targets": TARGETS,
        },
        os.path.join(
            split_outdir,
            "model.pt"
        )
    )

    with open(
        os.path.join(
            split_outdir,
            "metrics.json"
        ),
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    with open(
        os.path.join(
            split_outdir,
            "history.json"
        ),
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )

    plot_training(
        history,
        os.path.join(
            split_outdir,
            "training_curve.png"
        )
    )

    plot_predictions(
        y_true,
        y_pred,
        metrics,
        split_outdir
    )

    save_predictions(
        test_df,
        y_true,
        y_pred,
        os.path.join(
            split_outdir,
            "test_predictions.csv"
        )
    )

    return {
        "split": split_name,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "metrics": metrics,
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=DEFAULT_DATA
    )

    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR
    )

    parser.add_argument(
        "--split",
        choices=[
            "all",
            "random",
            "pair",
            "ligand"
        ],
        default="all"
    )

    args = parser.parse_args()

    set_seed()

    print("=" * 72)
    print("ML3 GNN MAGNETIC-PROPERTY ORACLE")
    print("=" * 72)

    print(
        f"Device: {DEVICE}"
    )

    # ========================================================
    # Load
    # ========================================================

    df = pd.read_csv(
        args.data
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    required = [
        "L1",
        "L2",
        "CN1",
        "CN2",
        "Ucal",
        "Ueff",
        "tio",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    # ========================================================
    # Canonicalize
    # ========================================================

    df["L1"] = df["L1"].map(
        canonicalize_smiles
    )

    df["L2"] = df["L2"].map(
        canonicalize_smiles
    )

    df = df.dropna(
        subset=required
    ).reset_index(
        drop=True
    )

    print(
        f"Clean dataset: {len(df):,}"
    )

    # ========================================================
    # Build unique graph cache
    # ========================================================

    ligands = sorted(
        set(df["L1"]) |
        set(df["L2"])
    )

    print(
        f"Unique ligands: {len(ligands):,}"
    )

    graph_cache = {}

    for i, smiles in enumerate(
        ligands,
        start=1
    ):

        graph_cache[smiles] = (
            smiles_to_graph(smiles)
        )

        if i % 100 == 0:
            print(
                f"Graphs built: "
                f"{i}/{len(ligands)}"
            )

    print(
        f"Graph cache complete: "
        f"{len(graph_cache):,}"
    )

    # ========================================================
    # Select splits
    # ========================================================

    if args.split == "all":

        splits = [
            "random",
            "pair",
            "ligand"
        ]

    else:

        splits = [
            args.split
        ]

    all_results = []

    # ========================================================
    # Run
    # ========================================================

    for split_name in splits:

        result = run_split(
            df=df,
            split_name=split_name,
            graph_cache=graph_cache,
            root_outdir=args.outdir
        )

        all_results.append(
            result
        )

    # ========================================================
    # Compare
    # ========================================================

    comparison = []

    for result in all_results:

        row = {
            "split": result["split"],
            "n_train": result["n_train"],
            "n_val": result["n_val"],
            "n_test": result["n_test"],
        }

        for target in TARGETS:

            row[
                f"{target}_R2"
            ] = result[
                "metrics"
            ][target]["R2"]

            row[
                f"{target}_MAE"
            ] = result[
                "metrics"
            ][target]["MAE"]

            row[
                f"{target}_RMSE"
            ] = result[
                "metrics"
            ][target]["RMSE"]

        row["mean_R2"] = result[
            "metrics"
        ]["mean_R2"]

        row["mean_MAE"] = result[
            "metrics"
        ]["mean_MAE"]

        comparison.append(row)

    comparison_df = pd.DataFrame(
        comparison
    )

    comparison_df.to_csv(
        os.path.join(
            args.outdir,
            "split_comparison.csv"
        ),
        index=False
    )

    with open(
        os.path.join(
            args.outdir,
            "all_results.json"
        ),
        "w"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2
        )

    print()
    print("=" * 72)
    print("FINAL SPLIT COMPARISON")
    print("=" * 72)

    print(
        comparison_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:",
        os.path.join(
            args.outdir,
            "split_comparison.csv"
        )
    )

    print()
    print("=" * 72)
    print("GNN ORACLE TRAINING COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
