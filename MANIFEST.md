# ML3 FINAL DEPLOYMENT MANIFEST

## Runtime
- `app.py`
- `memory_engine.py`
- `steric_augmentation.py`
- `run_iterative_search.py`
- `smoke_test_memory.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `ML3_Ucal_Ueff_tio_2.csv` — 1,689 labelled complexes
- `generated_candidates/` — prior generated seed library
- `true_jtvae_model/` — JT-VAE checkpoint/config
- `true_jtvae_vocab.txt`
- `latent_oracle/` — all-474-ligand latent property oracle and scaler
- `gnn_oracle/` — all-1,687-clean-pair pair-GNN checkpoint
- patched `fast_jtnn/` and `fast_molopt/`

## Workflow
1. User specifies Ucal, Ueff or Tor target and Dy ligand coordination numbers.
2. App computes the observed target range and explicitly labels out-of-range targets as extrapolation.
3. Initial seeds are selected from the dataset/generated library.
4. Seeds are encoded into the 56-D JT-VAE latent space.
5. Latent property oracle is optimized toward the requested target.
6. Optimized latent vectors are decoded by JT-VAE into ligand SMILES.
7. Generated ligands can receive H->Me/Et/nPr/iPr/tBu substitutions.
8. Candidate ligands are combined as L1/L2 pairs.
9. The pair GNN predicts Ucal/Ueff/tio for the combinations.
10. Target-ranked elite pairs become memory.
11. Elite ligand SMILES are encoded back to latent space and reused in the next iteration together with fresh random starts.
12. Final output is a ranked list of ligand combinations plus lineage/memory tables.

## Important
The deployment uses **no ExtraTrees/joblib models**. It is JT-VAE + latent property oracle + pair GNN + iterative memory/steric expansion.
