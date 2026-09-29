"""Offline deployment smoke test. Does not require Streamlit."""
import json, sys
from pathlib import Path
import torch
from rdkit import Chem
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from fast_jtnn import JTNNVAE,Vocab
import torch.nn as nn
class PropNet(nn.Module):
    def __init__(self,d=56,h=256,out=3):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,out))
    def forward(self,x): return self.net(x)
cfg=json.load(open(BASE/'true_jtvae_model/config.json'))
# load the actual JT vocabulary from the deployment copy
vocab=Vocab([x.strip() for x in open(BASE/'true_jtvae_vocab.txt') if x.strip()])
model=JTNNVAE(vocab,cfg['hidden_size'],cfg['latent_size'],cfg['depthT'],cfg['depthG'])
model.load_state_dict(torch.load(BASE/'true_jtvae_model/best_model.pt',map_location='cpu',weights_only=False)); model.eval()
z=torch.randn(1,56); zt,zm=torch.chunk(z,2,1); s=model.decode(zt,zm,False)
assert Chem.MolFromSmiles(s) is not None
oracle=PropNet(); oracle.load_state_dict(torch.load(BASE/'latent_oracle/property_oracle.pt',map_location='cpu',weights_only=False)); oracle.eval(); y=oracle(z)
assert tuple(y.shape)==(1,3)
print('PASS: JT-VAE decode + latent property oracle load.')
print('Decoded:',s)
