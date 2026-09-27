
# DEPLOYMENT CHECKLIST

## A. On your Windows computer

1. Put these files in the app project:
   - app.py
   - prepare_app_data.py
   - requirements.txt
   - requirements_prepare.txt

2. Run the offline preparation command from README.md.

3. Confirm that these files are created:
   - data/pair_predictions.csv
   - data/candidate_ligand_library.csv
   - data/app_metadata.json

4. Check that `pair_predictions.csv` contains:
   - L1
   - L2
   - Ueff_pred
   - tio_pred
   - tau0_pred_s
   - Tor_pred_K

## B. GitHub repository

For the deployed app, commit:

    app.py
    requirements.txt
    data/pair_predictions.csv
    data/candidate_ligand_library.csv
    data/app_metadata.json

You may also commit:

    README.md

The deployed app does NOT need:

    pair_oracle_cv/
    JT-VAE-tmcinvdes-main/
    prepare_app_data.py
    requirements_prepare.txt

Keeping `prepare_app_data.py` in the repository is fine, but it is not
executed by Streamlit Cloud.

## C. Streamlit Community Cloud

Create a new app and choose:

    Repository: your GitHub repository
    Branch: main
    Main file: app.py

Open Advanced settings and select:

    Python 3.12

The current Streamlit Community Cloud documentation says Python 3.12
is the default, and the Python version can be selected in Advanced
settings during deployment.

## D. IMPORTANT

Do NOT put RDKit in the deployed `requirements.txt`.

Do NOT put PyTorch/JT-VAE dependencies in the deployed app.

The cloud application only reads the precomputed CSV and performs
sorting/filtering. Therefore the previous:

    No module named fast_molopt

error is not part of this deployment architecture.

## E. Updating the app later

Whenever you:

- generate new JT-VAE ligands,
- add methyl/ethyl variants,
- change the original ligand dataset, or
- retrain the pair oracle,

rerun `prepare_app_data.py`.

Then replace:

    data/pair_predictions.csv
    data/candidate_ligand_library.csv
    data/app_metadata.json

Commit and push to GitHub.

Streamlit Community Cloud will detect the GitHub update and restart
the application.
