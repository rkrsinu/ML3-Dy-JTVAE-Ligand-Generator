
# ML3 Dy(III) Ligand-Pair Streamlit App

## Purpose

This application takes ONE target property:

- `Ueff`
- `Tor`

and returns actual ligand combinations `(L1 + L2)` ranked by the trained ML pair oracle.

The Streamlit deployment does NOT load JT-VAE, `fast_molopt`, PyTorch, or the generation code.

JT-VAE generation and pair-oracle prediction are performed OFFLINE.

The cloud app only reads:

    data/pair_predictions.csv

and ranks the precomputed ligand combinations.

---

# 1. Folder structure

Create this repository:

```text
ML3_JT_VAE_LigandPair_Streamlit/
│
├── app.py
├── prepare_app_data.py
├── requirements.txt
├── requirements_prepare.txt
├── README.md
│
└── data/
    ├── pair_predictions.csv
    ├── candidate_ligand_library.csv
    └── app_metadata.json
```

Do NOT put the JT-VAE training folder in the Streamlit repository.

Do NOT put `fast_molopt` in the Streamlit app.

The three pair-oracle `.joblib` files are only required by the OFFLINE
preparation script.

---

# 2. Offline preparation

Run this on the Windows machine where your ML3 environment and RDKit
already work.

Example:

```powershell
cd D:\2026\ML3\APP\ML3_JT_VAE_LigandPair_Streamlit
```

Then run:

```powershell
python prepare_app_data.py `
  --data "D:\2026\ML3\JTVAE_final\ML3_JTVAE_FINAL_CLEAN\ML3_Ucal_Ueff_tio_2.csv" `
  --generated "D:\2026\ML3\JTVAE_final\ML3_JTVAE_FINAL_CLEAN\generated_candidates\generated_candidates.csv" `
  --augmented "D:\2026\ML3\JTVAE_final\ML3_JTVAE_FINAL_CLEAN\generated_candidates\augmented_generated_candidates.csv" `
  --oracle_dir "D:\2026\ML3\JTVAE_final\ML3_JTVAE_FINAL_CLEAN\pair_oracle_cv" `
  --out data `
  --cn1 2 `
  --cn2 2 `
  --batch_size 2000
```

If your actual paths are different, change only the paths.

---

# 3. What the preparation script does

It combines:

1. Original ligands from L1 and L2 of the original dataset.
2. JT-VAE generated ligands.
3. Optional methyl/ethyl augmented ligands.

Duplicate canonical SMILES are merged.

It then constructs all unique unordered ligand pairs.

For every pair it uses the EXACT oracle feature representation:

```text
L1 Morgan FP       1024
L2 Morgan FP       1024
|L1-L2|            1024
L1*L2              1024
L1 descriptors       9
L2 descriptors       9
|descriptor diff|    9
descriptor product   9
CN features          5
--------------------------------
TOTAL              4137
```

No padding and no truncation are performed.

The models are:

```text
final_Ucal_extra_trees.joblib
final_Ueff_extra_trees.joblib
final_tio_extra_trees.joblib
```

The `tio` model predicts:

```text
log10(tau0)
```

---

# 4. Tor calculation

The app uses:

```text
tau0 = 10^(tio_pred)
```

with:

```text
tau_ref = 100 s
```

and:

```text
Tor = -Ueff / ln(tau0 / 100)
```

---

# 5. Run locally only if desired

The Streamlit app itself does not need RDKit.

Install:

```powershell
pip install -r requirements.txt
```

Then:

```powershell
streamlit run app.py
```

However, local testing is optional.

---

# 6. Deploy to Streamlit Community Cloud

Push these files to GitHub:

```text
app.py
requirements.txt
data/pair_predictions.csv
data/candidate_ligand_library.csv
data/app_metadata.json
```

You do NOT need:

```text
prepare_app_data.py
requirements_prepare.txt
pair_oracle_cv/
JT-VAE-tmcinvdes-main/
```

in the deployed repository.

You can keep the preparation script in GitHub if you want reproducibility,
but it is not required by the deployed app.

In Streamlit Cloud select:

```text
Repository: your GitHub repository
Branch: main
Main file: app.py
```

---

# 7. User interface

The user chooses:

```text
Target property

○ Ueff
○ Tor
```

### Ueff mode

Example:

```text
Target Ueff = 2000 K
```

The app ranks:

```text
|Predicted Ueff - 2000|
```

and returns:

```text
L1 + L2
```

along with:

- predicted Ueff
- predicted log10(tau0)
- predicted tau0
- predicted Tor
- pair source
- model uncertainty

### Tor mode

Example:

```text
Target Tor = 100 K
```

The app calculates Tor for every pair and ranks:

```text
|Predicted Tor - 100|
```

It then returns the actual:

```text
L1 + L2
```

ligand combinations.

---

# 8. Important

The app does NOT generate a new JT-VAE molecule every time the user
changes the target.

Instead:

```text
JT-VAE
   ↓
offline ligand generation
   ↓
candidate ligand library
   ↓
pair oracle
   ↓
precomputed pair predictions
   ↓
Streamlit target search
```

This makes the deployed app lightweight and avoids the previous errors:

```text
No module named 'fast_molopt'
```

and RDKit/Python-version dependency problems on Streamlit Cloud.

When you generate a new batch of JT-VAE ligands or retrain the oracle,
rerun:

```text
prepare_app_data.py
```

and replace the files inside `data/`.

Then commit/push to GitHub.
