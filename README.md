# ML3 Memory-Augmented JT-VAE Target-Directed Ligand Generator

Final deployment architecture:

`Target -> 56-D JT-VAE latent search -> novel ligand decoding -> H-to-alkyl steric expansion -> two-ligand pair GNN -> elite memory -> next latent iteration`

The deployment deliberately does **not** use ExtraTrees/joblib models.

## Runtime

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app accepts targets outside the observed dataset range (for example Ueff = 3000 K). Such a request is labelled **EXTRAPOLATION**. The target is used to optimize the continuous JT-VAE latent representation through the separately trained latent property oracle. The decoded ligands are then chemically augmented and screened as L1/L2 pairs by the pair GNN.

## Iterative memory

At every iteration the app stores generated ligands, parent/steric modification, predicted properties, pair combinations and target error. The best pairs become memory. Their ligand SMILES are encoded back into the JT-VAE latent space and reused as seeds in the next iteration, together with fresh random starts.

## Steric modification

Only newly generated JT-VAE ligands are modified. The experimental library is never edited. Supported substitutions are H -> Me, Et, nPr, iPr and tBu.

## Training

`training/` contains the all-data preparation and training scripts. The final deployment fit uses all 1,689 labelled complexes and all 474 unique ligands; no train/test split is used for the deployment fit. Training diagnostics from an independent holdout/CV should be kept separate from deployment fitting.
