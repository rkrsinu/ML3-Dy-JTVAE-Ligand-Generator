# ML3 Memory-Augmented JT-VAE deployment fix

This package keeps the ML3 workflow:

Target -> 56-D JT-VAE latent optimization -> decoded novel ligands -> H-to-Me/Et/nPr/iPr/tBu augmentation -> L1/L2 pair generation -> PairGNN screening -> elite memory -> next iteration.

The critical fix is the latent property-oracle loader. `property_oracle.pt` may be either a raw PyTorch `state_dict` or a wrapped checkpoint containing `model_state_dict`; the app now supports both and loads the neural-network weights strictly.

`requirements.txt` also includes SciPy because the JT-VAE chemistry implementation imports `scipy.sparse`.
