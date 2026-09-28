# ML3 JT-VAE Ligand-Pair App — Development Package

This package contains only the files required to develop and prepare the Streamlit ligand-combination app from the supplied clean project.

## Required offline inputs
- data_inputs/ML3_Ucal_Ueff_tio_2.csv
- data_inputs/generated_candidates.csv
- data_inputs/augmented_generated_candidates.csv (optional but recommended if already generated)
- oracle/ with the four existing pair-oracle files:
  final_Ucal_extra_trees.joblib
  final_Ueff_extra_trees.joblib
  final_tio_extra_trees.joblib
  feature_config.joblib

## App files
- app.py — Streamlit UI; it reads precomputed pair_predictions.csv and calculates/searches Tor.
- prepare_app_data.py — offline builder; creates pair_predictions.csv, candidate_ligand_library.csv and app_metadata.json using the exact 4137-feature representation.
- requirements.txt — lightweight deployment dependencies; no RDKit/PyTorch/JT-VAE.

## Important
The uploaded ZIP does NOT contain pair_oracle_cv. It contains latent_oracle, which is a different model and is not a substitute for the three ExtraTrees pair oracles required by prepare_app_data.py. Therefore copy the existing pair_oracle_cv files from your working clean project into oracle/; do not retrain if they already exist.

## Prepare app data
From this package root:

python prepare_app_data.py --data data_inputs/ML3_Ucal_Ueff_tio_2.csv --generated data_inputs/generated_candidates.csv --augmented data_inputs/augmented_generated_candidates.csv --oracle_dir oracle --out data --cn1 2 --cn2 2 --batch_size 2000

After success, data/ should contain:
- pair_predictions.csv
- candidate_ligand_library.csv
- app_metadata.json

Only these generated data files plus app.py and requirements.txt are needed in the GitHub deployment.
