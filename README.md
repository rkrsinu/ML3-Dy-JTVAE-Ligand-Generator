# Dy-based SMM JT-VAE Ligand Generator

Streamlit app for property-guided generation of novel ligands for pseudo-linear Dy-based single-molecule magnets.

## Target definition

The app uses

\[
T_{Or}=-\frac{U_{eff}}{\ln(\tau_0/\tau_{ref})}
\]

with \(\tau_{ref}=100\) s and the model property `tio = log10(tau0)`.

For a selected target `Ueff` and `TOr`, the required `log10(tau0)` is calculated as:

\[
\log_{10}(\tau_0)=2-\frac{U_{eff}}{T_{Or}\ln(10)}.
\]

The trained property oracle then guides the JT-VAE latent vector toward the requested `Ueff` and `tio`. `Ucal` is predicted as an output rather than used as a target.

A `TOr` value alone cannot uniquely determine both `Ueff` and `tau0`, so the app deliberately asks for both `Ueff` and `TOr`.

## Run locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## GitHub / Streamlit Community Cloud

Upload this complete repository to GitHub and deploy `app.py` as the main file.

The repository contains the trained JT-VAE checkpoint, latent property oracle, vocabulary, known-ligand library, and the required `fast_jtnn` implementation.

No external path such as `D:\2026\...` is used by the app.

## Model provenance

The model files and JT-VAE implementation are taken from the user's cleaned ML3 JT-VAE workflow. The app performs generation with the actual JT-VAE decoder; it is not a direct SMILES-regression model.

## Validation

Generated molecules are RDKit-checked, canonicalized, required to be novel relative to the supplied JT-VAE-compatible ligand library, and restricted to at most one ring in this app. Final candidates should still be validated computationally with the user's Dy CASSCF/RASSI-SO/SINGLE_ANISO workflow.
