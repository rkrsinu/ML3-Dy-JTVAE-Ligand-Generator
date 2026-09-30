# ML3 Memory-Augmented JT-VAE — corrected CN handling

This version fixes the CN assignment problem.

The previous implementation evaluated every Cartesian combination of all CN values in the training set. That could assign an unrelated CN to a ligand.

Corrected priority:
1. Exact observed ligand pair -> use the dataset's observed CN1/CN2.
2. Exact known ligand, but unseen pair -> use CN values observed for that ligand in its L1/L2 role.
3. Novel JT-VAE ligand -> infer CN only from structurally similar training ligands using Morgan similarity and weighted nearest-neighbour voting.
4. No sufficiently similar reference -> do not invent a CN; exclude the pair from GNN screening.

CN is not inferred from donor-atom count and is not selected merely because that CN exists somewhere in the dataset.

The GNN still receives CN1 and CN2 exactly as training inputs.
