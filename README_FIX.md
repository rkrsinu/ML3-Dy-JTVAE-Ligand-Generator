# ML3 Memory-Augmented JT-VAE — deployment update

This version is prepared for the minimal, paper-facing Streamlit interface.

## User-facing controls

Only two target controls and one search control are exposed:

- target property: Ueff / Ucal / Tor
- requested target value
- number of memory iterations

The following are intentionally hidden and fixed in the production configuration:

- latent starts per iteration
- latent optimization steps
- decodes per seed
- number of new ligands
- elite-pair count
- random seed
- steric-substitution checkboxes
- coordination-number selectors

## Coordination numbers

CN1 and CN2 are not selected by the user. The search automatically evaluates all
CN1/CN2 combinations represented in the curated training dataset and reports the
CN pair associated with each predicted candidate complex.

## Steric diversification

Steric diversification is automatic and ligand-dependent. The backend examines
the generated ligand's size, ring count, and available carbon-bound H sites and
selects an appropriate subset of Me/Et/nPr/iPr/tBu substitutions. The original
experimental library is not modified.

## Memory search

Each iteration:

1. encodes elite ligand memory into the JT-VAE latent space;
2. combines memory seeds with fresh latent starts;
3. optimizes latent vectors toward the requested target;
4. decodes new ligands;
5. applies adaptive ligand-specific steric diversification;
6. screens ligand pairs with the complex-level PairGNN over supported CN pairs;
7. retains target-ranked elite pairs as memory for the next iteration.

The final table contains the target-ranked ligand combinations and their predicted
magnetic properties, CN1/CN2 values, target error, and iteration provenance.

## Fixed production configuration

- seed = 42
- latent starts / iteration = 16
- latent optimization steps = 80
- decodes / seed = 4
- new ligands / iteration = 80
- elite pairs / iteration = 20
- maximum PairGNN evaluations / iteration = 30,000
- maximum adaptive steric variants / generated parent = 6
