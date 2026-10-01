# ML3 geometry-aware JT-VAE screening

This package adds a small geometry-learning layer to the existing ML3 JT-VAE iterative app. The existing JT-VAE generation and iterative memory workflow is retained.

## Added models
- `geometry_models/geometry_model.pt`: molecular-graph pair GNN predicting `LL1`, `LL2`, `LL`, `BA` from L1/L2 graphs and CN1/CN2.
- `geometry_models/geometry_aware_property_gnn.pt`: pair GNN predicting `Ucal`, `Ueff`, `tio` using L1/L2 graphs, CN1/CN2 and the predicted geometry.

## Geometry data
`all_BL_BA_SMILES.xlsx` contains the 1,579 ligand-pair structures used for geometry/property training. `LL1` and `LL2` are Dy-ligand distances, `LL` is L1-L2 distance, and `BA` is L1-Dy-L2 angle.

## GitHub replacement
Replace your current `app.py` and `memory_engine.py` with the files in this package. Add `geometry_model.py`, `geometry_models/`, and `all_BL_BA_SMILES.xlsx`. Keep your existing `true_jtvae_model/`, `true_jtvae_vocab.txt`, `latent_oracle/`, `fast_jtnn/`, `fast_molopt/`, and `steric_augmentation.py`.

The app does not expose CN or geometry as user inputs. Known ligand pairs use observed pair-specific CN configurations. Novel pairs inherit CN from the nearest observed ligand-pair environment rather than trying arbitrary CN combinations. The geometry GNN then predicts LL1/LL2/LL/BA before property screening.

## Train from scratch
`python train_geometry_and_property_models.py`

The trainer reports held-out, pair/CN-grouped MAE for geometry and properties and writes the two checkpoints.
