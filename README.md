# ML3 JT-VAE Target-Directed Ligand Combination Generator

## Workflow
Target property -> JT-VAE latent-space search -> novel ligand SMILES -> two-ligand combinations -> pair GNN screening -> target-ranked ligand combinations.

The app supports Ucal, Ueff and T_or targets. If the requested target lies outside the observed dataset range, the app continues as a model extrapolation search and flags the result.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important
The deployment uses the supplied TRUE JT-VAE checkpoint and latent property oracle. The pair screening checkpoint included in `gnn_oracle/model.pt` was prepared from the supplied pair-GNN formulation in a pure-PyTorch deployment implementation so that `torch-geometric` is not required by Streamlit Cloud.

The GNN is a screening/ranking model; predicted extrapolated properties must be validated experimentally or with appropriate quantum-chemical calculations.
