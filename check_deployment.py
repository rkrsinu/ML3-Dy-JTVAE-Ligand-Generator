
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent

required = [
    "app.py",
    "memory_engine.py",
    "cn_manager.py",
    "ligand_validation.py",
    "steric_augmentation.py",
    "requirements.txt",
    "ML3_Ucal_Ueff_tio_2.csv",
    "true_jtvae_model/best_model.pt",
    "true_jtvae_model/config.json",
    "true_jtvae_vocab.txt",
    "latent_oracle/property_oracle.pt",
    "latent_oracle/property_scaler.csv",
    "gnn_oracle/model.pt",
]

missing = [p for p in required if not (BASE / p).exists()]

if missing:
    print("Missing deployment files:")
    for p in missing:
        print("  -", p)
    sys.exit(1)

print("All required ML3 deployment files are present.")
