# ML3 Streamlit deployment fixes

Apply these files to the existing repository.

## 1. app.py
Replace the current app.py.

Fixes:
- correct 12-value model tuple unpacking
- keeps geometry-aware model loading unchanged

## 2. memory_engine.py
Replace the current memory_engine.py.

Fixes:
- `pair_screen()` now accepts the property-model scaler as one object
- call and function signature are consistent
- geometry normalization uses `geometry_mean/geometry_std`
- target conversion uses `target_mean/target_std`

## 3. fast_jtnn/datautils_prop.py
Replace the current file.

Fixes the deployment circular import:
`fast_jtnn -> datautils_prop -> fast_molopt.preprocess_prop -> fast_jtnn`

`fast_molopt` is now imported lazily only when dataset preprocessing/training is actually used. JT-VAE inference and latent encoding do not import `fast_molopt`.

## Important
Do not retrain any model. The errors are code/import/signature errors, not model errors.
