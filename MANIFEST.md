# ML3 deployment manifest

## Runtime files
- `app.py` — complete Streamlit application.
- `requirements.txt` — Streamlit/RDKit/PyTorch runtime dependencies.
- `true_jtvae_model/best_model.pt` + `config.json` — trained TRUE JT-VAE checkpoint.
- `true_jtvae_vocab.txt` — JT-VAE vocabulary.
- `fast_jtnn/` + `fast_molopt/` — JT-VAE inference implementation required by the checkpoint.
- `latent_oracle/property_oracle.pt` + `property_scaler.csv` — target-directed latent-space search model.
- `gnn_oracle/model.pt` + `metrics.json` — two-ligand GNN screening checkpoint and validation record.
- `ML3_Ucal_Ueff_tio_2.csv` — original 1689-complex dataset used by the project.
- `generated_candidates/generated_candidates.csv` — existing valid novel JT-VAE ligand library used as an additional seed pool.
- `.streamlit/config.toml` — Streamlit settings.

## Optional reference code
`training_reference/` contains the project scripts used for JT-VAE latent-property fitting, JT-VAE property-guided generation, pair generation, and the supplied GNN training formulation. They are not required to run the deployed app.

## Deliberate exclusions
The deployment does **not** use the ExtraTrees/joblib pair oracle. No training-only CSVs, plots, caches, duplicate checkpoints, or Python cache files are included.
