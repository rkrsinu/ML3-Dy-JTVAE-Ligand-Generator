# ML3 Memory-Augmented JT-VAE — JACS-ready deployment

This deployment is a target-directed ligand-pair generation workflow.

User controls:
- Target property: Ueff, Ucal, or Tor
- Target value
- Number of memory iterations

The following are intentionally hidden from the UI and fixed internally:
- random seed
- latent starts
- latent optimization steps
- number of decodes
- number of new ligands per iteration
- elite-memory size
- steric-variant count
- pair-search limit
- coordination-number inputs

Workflow:
1. JT-VAE generates target-directed ligand candidates in latent space.
2. Generated ligands are automatically diversified when chemically suitable
   C-H sites are available; the user does not choose substituents.
3. Ligand pairs are evaluated by the two-ligand GNN across CN1/CN2
   combinations represented in the training dataset.
4. Target-ranked elite pairs are stored as memory.
5. Elite ligands are re-encoded into JT-VAE latent space and used as seeds
   for the next iteration together with fresh random starts.
6. Final output is the accumulated target-ranked ligand combinations.

No retraining is required for deployment.
