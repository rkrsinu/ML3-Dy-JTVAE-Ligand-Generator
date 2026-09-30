# ML3 memory-augmented deployment manifest

## Required runtime files
- `app.py` — Streamlit interface.
- `memory_engine.py` — iterative target search, memory archive, latent-seed feedback, pair-GNN screening.
- `steric_augmentation.py` — H→Me/H→Et steric expansion plus optional nPr/iPr/tBu.
- `run_iterative_search.py` — command-line version of the same workflow.
- `smoke_test_memory.py` — checkpoint/encode/decode smoke test.
- `requirements.txt` — deployment dependencies.
- `ML3_Ucal_Ueff_tio_2.csv` — original project dataset.
- `true_jtvae_model/` — trained JT-VAE checkpoint and config.
- `true_jtvae_vocab.txt` — trained JT-VAE vocabulary.
- `fast_jtnn/` and `fast_molopt/` — JT-VAE inference implementation.
- `latent_oracle/` — latent property oracle + scaler used for target-directed latent search.
- `gnn_oracle/` — two-ligand GNN checkpoint + metrics.
- `generated_candidates/` — existing generated ligand seed library.
- `.streamlit/config.toml` — Streamlit configuration.

## Workflow added in this version

1. Target selection: Ucal, Ueff or Tor.
2. Dataset range is calculated and an out-of-range target is flagged as extrapolation.
3. JT-VAE latent optimization searches toward the target.
4. New ligands are decoded.
5. Generated ligands receive controlled steric augmentation. Original experimental ligands are never modified.
6. L1/L2 combinations are screened with the two-ligand GNN.
7. Best combinations are stored in memory.
8. Elite ligand SMILES are encoded back into the JT-VAE 56-D latent space.
9. Those elite latent vectors seed the next iteration together with fresh random exploration.
10. The final output is the ranked L1 + L2 ligand combination table.

## GitHub deployment

No ExtraTrees/joblib files are required by this version. The deployment uses the JT-VAE + latent oracle + pair GNN workflow requested for extrapolative ligand generation.
