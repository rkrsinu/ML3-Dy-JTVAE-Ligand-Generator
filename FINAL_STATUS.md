# Final build verification

## Verified in this environment

- Input dataset: **1,689 complexes**.
- Unique ligand library: **474 ligands**.
- JT-VAE vocabulary: **33 tokens**.
- Latent property oracle: trained on **all 474 ligand latent/property rows**.
- Pair GNN: trained on **all 1,687 unique labelled pair rows** after exact duplicate removal.
- `smoke_test_memory.py`: **PASSED**.
- Extrapolation smoke search at **Ueff = 3000 K** with CN1=CN2=2: **PASSED**; novel ligand combinations were produced.
- Steric augmentation was observed in the smoke search with both **H→Me** and **H→Et** variants, with parent SMILES and modification labels stored in memory.

## JT-VAE checkpoint note

The included `true_jtvae_model/best_model.pt` is the previously trained ML3 JT-VAE checkpoint already present in the project. The attempted CPU refit of the generative JT-VAE on all 474 ligands was not completed in this environment, so this file is **not falsely labelled as a newly completed all-data refit**.

The package therefore includes `training/scripts/train_jtvae_all_unconditional.py` and `training/scripts/finetune_jtvae_all.py` for the all-data generative-model fit, together with the patched JT-VAE source that removes the Windows circular-import and empty-assembly failures.

The deployment app itself is runnable with the included checkpoint and uses the requested workflow:

**JT-VAE latent generation → steric modification → L1/L2 pair GNN screening → memory → next iteration.**
