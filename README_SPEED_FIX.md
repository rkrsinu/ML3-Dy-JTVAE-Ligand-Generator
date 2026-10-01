# ML3 Streamlit Speed Fix

This is an update to `ML3_STREAMLIT_DEPLOYMENT_FIX.zip`.

## Main problem fixed
The previous `pair_screen()` recomputed Morgan fingerprints for every candidate pair against every observed ligand pair. With 104 ligands this creates 5,460 candidate pairs and 2,738 observed ordered ligand pairs, causing tens of millions of repeated RDKit fingerprint operations and deprecation-warning output.

The new version:
- caches all fingerprints;
- uses RDKit `MorganGenerator` and bulk Tanimoto similarity;
- vectorizes nearest observed CN-environment lookup;
- parses each candidate ligand graph only once;
- encodes each unique ligand once per screening call;
- evaluates only the small pair-level neural-network heads afterward;
- reports pair-screening progress batch-by-batch.

A local benchmark with 104 ligands and the supplied geometry/property models reduced pair screening from a process that exceeded 120 s to about 0.56 s on the test environment, while the nearest-CN assignments matched the previous implementation for 100 checked novel pairs.

## Replace
Copy these files into the existing Streamlit repository:
- `memory_engine.py`
- `app.py`
- `geometry_model.py`
- `fast_jtnn/datautils_prop.py`

No model retraining is required.
