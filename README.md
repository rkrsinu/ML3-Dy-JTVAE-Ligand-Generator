# ML3 Memory-Augmented JT-VAE — Iterative Target-Directed Version

## What changed

This version implements the workflow requested for the ML3 project:

1. Start from experimentally known ligands near the requested target.
2. Iteration 1 generates/optimizes new ligands around an intermediate target.
3. Screen ligand pairs with the trained PairGNN.
4. Keep the best pair/litigand combinations as memory.
5. Iteration 2 uses those winners as JT-VAE latent seeds.
6. H→Me / H→Et / additional ligand-dependent alkyl modifications are generated automatically.
7. Repeat the cycle.
8. The intermediate target moves progressively toward the final target.
9. Final ranking is performed against the user's actual target.

For a 3000 K Ueff target, the search therefore behaves as a progressive trajectory rather than forcing every first-generation molecule directly toward 3000 K.

## CN handling

CN is NOT a user input.

The application uses this priority:

- exact observed ligand pair -> exact observed CN1/CN2
- exact observed ligand in its L1/L2 role -> observed role-specific CN
- steric derivative -> inherits parent ligand's observed CN
- genuinely novel ligand -> CN inferred only from structurally similar observed ligands in the same role
- if no defensible CN exists -> the pair is not sent to the PairGNN

The application does NOT enumerate every possible CN1/CN2 combination and choose whichever gives the best target score.

CN is also NOT treated as donor-atom count.

## Generated-ligand validation

Before PairGNN screening, generated ligands must:

- be valid RDKit molecules
- be a single connected molecular component
- contain only elements represented in the experimental ligand library
- remain within the project's one-ring generated-ligand space
- remain within a controlled heavy-atom range
- have at least modest similarity to the learned ligand chemical space
- pass a guard against structures that look like two known ligands fused together

This is specifically intended to prevent structures such as a decoded/fused combination from being silently accepted as a normal single ligand.

## Files required beside the code

The deployment directory should contain:

    app.py
    memory_engine.py
    cn_manager.py
    ligand_validation.py
    steric_augmentation.py
    requirements.txt

and these existing ML3 artifacts:

    ML3_Ucal_Ueff_tio_2.csv
    true_jtvae_model/
        best_model.pt
        config.json
    true_jtvae_vocab.txt
    latent_oracle/
        property_oracle.pt
        property_scaler.csv
    gnn_oracle/
        model.pt
    generated_candidates/
        generated_candidates.csv       # optional

The JT-VAE source package (fast_jtnn) must also be available in the deployment environment exactly as used to train `true_jtvae_model`.

## Important

The code does not manufacture a 3000 K prediction. If the trained models cannot extrapolate to 3000 K, the final table will show the closest model-supported result. The purpose of the iterative memory mechanism is to progressively move the search toward the target.
